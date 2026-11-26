"""CPU tests for neighborhood replacement and masked reconstruction."""
import ast
import hashlib
from pathlib import Path
import random
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def functions_from_script(relative, names, extra=None):
    source = ROOT / relative
    tree = ast.parse(source.read_text())
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    if {node.name for node in nodes} != set(names):
        raise AssertionError("Requested function not found in the experiment entry point")
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    scope = {"np": np, "random": random, "hashlib": hashlib,
             "TWO_HOP_PROTOCOL": "20261023_stratified_random_bijection_random_prefix"}
    scope.update(extra or {})
    exec(compile(ast.fix_missing_locations(ast.Module(body=[future] + nodes, type_ignores=[])), str(source), "exec"), scope)
    return scope


MICRO = functions_from_script("perturbation/microenvironment/microenv_perturbation.py", {
    "get_hop_neighbors", "get_sixteen_neighbors", "make_stratified_replacement_plan",
    "create_interpolated_perturbation",
})


class GridBank:
    def __init__(self):
        first = np.array([(x, y) for x in range(7) for y in range(7)], dtype=float)
        self.coords = np.concatenate([first, first + [20.0, 0.0]])
        self.ids = np.arange(len(self.coords), dtype=np.int64)
        squared = np.sum((self.coords[:, None] - self.coords[None, :]) ** 2, axis=-1)
        np.fill_diagonal(squared, np.inf)
        self.neighbors = np.argsort(squared, axis=1, kind="stable")[:, :8]

    def get_neighbors_for_spot(self, idx):
        return self.neighbors[idx].tolist()

    def get_spot_data(self, idx):
        return {"gene_ids": np.array([1, 2]), "raw_normed_values": np.array([idx, idx + 1], dtype=np.float32)}


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.bank = GridBank()
        self.source = MICRO["get_sixteen_neighbors"](self.bank, 24, self.bank.coords, self.bank.ids)
        self.target = MICRO["get_sixteen_neighbors"](self.bank, 73, self.bank.coords, self.bank.ids)
        self.plan = MICRO["make_stratified_replacement_plan"](self.source, self.target, "HBRC", "cancer_to_healthy", 0)

    def test_exact_sixteen_preserves_original_eight(self):
        self.assertEqual(len(set(self.source)), 16)
        self.assertEqual(self.source[:8], self.bank.get_neighbors_for_spot(24))
        self.assertNotIn(24, self.source)
        distances = np.sum((self.bank.coords - self.bank.coords[24]) ** 2, axis=1)
        expected_outer = [i for i in np.lexsort((self.bank.ids, distances))
                          if i != 24 and i not in self.source[:8]][:8]
        self.assertEqual(self.source[8:], expected_outer)

    def test_stratified_bijection_and_nested_prefixes(self):
        order = self.plan["source_order"]
        donors = self.plan["target_order"]
        self.assertEqual(set(order), set(self.source))
        self.assertEqual(set(donors), set(self.target))
        for s, t in zip(order, donors):
            self.assertEqual(s in self.source[:8], t in self.target[:8])
        for k in range(1, 17):
            for strength in (0.2, 1.0):
                edits = MICRO["create_interpolated_perturbation"](
                    self.bank, self.source, self.target, k, strength, 42,
                    replacement_plan=self.plan,
                )
                self.assertEqual(list(edits), order[:k])
                for s, t in zip(order[:k], donors[:k]):
                    expected = (1 - strength) * s + strength * t
                    np.testing.assert_allclose(edits[s]["raw_normed_values"], [expected, expected + 1])
        self.assertEqual(len(set(order[:16]).intersection(self.source[:8])), 8)

    def test_plan_is_repeatable_and_pair_specific(self):
        repeat = MICRO["make_stratified_replacement_plan"](self.source, self.target, "HBRC", "cancer_to_healthy", 0)
        other = MICRO["make_stratified_replacement_plan"](self.source, self.target, "HBRC", "cancer_to_healthy", 1)
        self.assertEqual(self.plan, repeat)
        self.assertNotEqual(self.plan["derived_seed_sha256_hex"], other["derived_seed_sha256_hex"])

    def test_one_hop_selection_is_unchanged(self):
        source, target = self.source[:8], self.target[:8]
        result = MICRO["create_interpolated_perturbation"](self.bank, source, target, 5, 1.0, 42)
        expected_sources = random.Random(42).sample(source, 5)
        self.assertEqual(list(result), expected_sources)
        self.assertEqual([float(result[s]["raw_normed_values"][0]) for s in result], target[:5])

    def test_overlap_is_rejected(self):
        with self.assertRaises(ValueError):
            MICRO["make_stratified_replacement_plan"](self.source, self.source, "HBRC", "cancer_to_healthy", 0)


class ExpressionBank:
    def __init__(self, hidden_value):
        self.rows = {
            0: {"gene_ids": np.array([1, 2]), "raw_normed_values": np.array([hidden_value, hidden_value + 1], dtype=np.float32)},
            1: {"gene_ids": np.array([1, 2]), "raw_normed_values": np.array([hidden_value + 2, hidden_value + 3], dtype=np.float32)},
            2: {"gene_ids": np.array([1, 2]), "raw_normed_values": np.array([3., 6.], dtype=np.float32)},
        }
        self.overrides = {}

    def clear_runtime_spot_overrides(self):
        self.overrides = {}

    def get_spot_data(self, index):
        return self.overrides.get(index, self.rows[index])

    def set_runtime_spot_overrides(self, values):
        self.overrides = values


class ReconstructionTests(unittest.TestCase):
    def run_masked_fixture(self, hidden_value):
        bank = ExpressionBank(hidden_value)
        calls = []

        def forward(current_bank, loader, model, device, **kwargs):
            calls.append(1)
            np.testing.assert_array_equal(current_bank.get_spot_data(0)["raw_normed_values"], [0, 0])
            np.testing.assert_array_equal(current_bank.get_spot_data(1)["raw_normed_values"], [0, 0])
            np.testing.assert_array_equal(current_bank.get_spot_data(2)["raw_normed_values"], [3, 6])

            return {i: {"gene_ids": np.array([1, 2]), "raw_normed_values":
                        (current_bank.get_spot_data(1-i)["raw_normed_values"] +
                         current_bank.get_spot_data(2)["raw_normed_values"]) / 2}
                    for i in (0, 1)}

        scope = functions_from_script("reconstruction/spatialgt_reconstruction_eval.py", {"run_masked_reconstruction"},
                                      {"run_single_pass_inference": forward})
        prediction = scope["run_masked_reconstruction"](bank, lambda: None, None, None, [0, 1, 2], {0, 1}, False)
        self.assertEqual(len(calls), 1)
        self.assertEqual(bank.overrides, {})
        return prediction

    def test_hidden_neighbor_values_cannot_change_first_prediction(self):
        first = self.run_masked_fixture(1.)
        poisoned = self.run_masked_fixture(1e6)
        for i in first:
            np.testing.assert_array_equal(first[i]["raw_normed_values"], poisoned[i]["raw_normed_values"])
            np.testing.assert_array_equal(first[i]["raw_normed_values"], [1.5, 3.])

    def test_missing_masked_prediction_is_an_error(self):
        scope = functions_from_script("reconstruction/spatialgt_reconstruction_eval.py", {"run_masked_reconstruction"},
                                      {"run_single_pass_inference": lambda *args, **kwargs: {}})
        bank = ExpressionBank(1.)
        with self.assertRaises(RuntimeError):
            scope["run_masked_reconstruction"](bank, lambda: None, None, None, [0, 1, 2], {0}, False)
        self.assertEqual(bank.overrides, {})

    def test_export_names_follow_token_order_and_preserve_section_case(self):
        scope = functions_from_script("reconstruction/spatialgt_reconstruction_eval.py", {"align_output_gene_names"})
        names = scope["align_output_gene_names"]([3, 1, 2], {"A": 1, "B": 2, "C": 3}, ["a", "c"])
        self.assertEqual(names, ["c", "a", "B"])

    def test_unknown_export_token_is_an_error(self):
        scope = functions_from_script("reconstruction/spatialgt_reconstruction_eval.py", {"align_output_gene_names"})
        with self.assertRaises(ValueError):
            scope["align_output_gene_names"]([9], {"A": 1}, ["A"])

    def test_oversized_mask_is_not_silently_reduced(self):
        scope = functions_from_script("reconstruction/spatialgt_reconstruction_eval.py", {"select_random_spots"})
        with self.assertRaises(ValueError):
            scope["select_random_spots"]([0, 1, 2], 4, 42)


if __name__ == "__main__":
    unittest.main()

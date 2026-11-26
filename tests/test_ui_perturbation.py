"""UI perturbation tests on small CPU fixtures."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CORE_DEPS = ("numpy", "pandas", "torch", "tqdm")
MISSING = [name for name in CORE_DEPS if importlib.util.find_spec(name) is None]
if not MISSING:
    import numpy as np
    import pandas as pd
    from spatialgt_agent_ui.core import perturber


class FakeModel:
    def __init__(self):
        self.eval_called = False

    def eval(self):
        self.eval_called = True
        return self


class TwoSpotBank:
    def __init__(self, source=None, remote=None):
        self.rows = {
            0: {"gene_ids": np.array([1, 2]), "raw_normed_values": np.array(source if source is not None else [0.5, 1.], dtype=np.float32)},
            1: {"gene_ids": np.array([1, 2]), "raw_normed_values": np.array(remote if remote is not None else [0., 0.], dtype=np.float32)},
        }
        self.overrides = {}
        self.loader_calls = 0

    def get_spot_data(self, index):
        return self.overrides.get(index, self.rows[index])

    def set_runtime_spot_overrides(self, values):
        self.overrides = values

    def get_data_loader(self, **kwargs):
        self.loader_calls += 1
        return None


@unittest.skipIf(bool(MISSING), "Install the runtime dependencies: " + ", ".join(MISSING))
class GeneEditTests(unittest.TestCase):
    def test_gene_case_and_log_normalized_formula(self):
        with tempfile.TemporaryDirectory() as directory:
            vocab_path = Path(directory) / "vocab.json"
            vocab_path.write_text(json.dumps({"Gbp6": 1, "OTHER": 2}))
            vocab = perturber.load_vocab(vocab_path)
        self.assertEqual(vocab, {"gbp6": 1, "other": 2})
        row = {"gene_ids": np.array([1, 2]), "raw_normed_values": np.array([np.log(2), 0.3], dtype=np.float32)}
        deg = pd.DataFrame({"gene": ["GBP6"], "avg_logFC": [1.], "p_val_adj": [0.01]})
        expected = np.log(3.)  # One linear-normalized count doubled, then log1p.
        edited = perturber.apply_deg_perturbation(row, deg, vocab)
        manual = perturber.apply_custom_gene_edits(row, {"GbP6": 1.}, vocab)
        self.assertAlmostEqual(float(edited["raw_normed_values"][0]), expected, places=6)
        self.assertAlmostEqual(float(manual["raw_normed_values"][0]), expected, places=6)
        np.testing.assert_array_equal(edited["raw_normed_values"], manual["raw_normed_values"])
        np.testing.assert_allclose(row["raw_normed_values"], [np.log(2), 0.3])

    def test_clipping_significance_and_strength(self):
        row = {"gene_ids": np.array([1, 2, 3]), "raw_normed_values": np.array([np.log(2), -0.1, 0.7], dtype=np.float32)}
        deg = pd.DataFrame({"gene": ["A", "C"], "avg_logFC": [20., 2.], "p_val_adj": [0.01, 0.08]})
        edited = perturber.apply_deg_perturbation(row, deg, {"a": 1, "b": 2, "c": 3}, strength=0.4, weight=0.5)
        # Clipped logFC=5; 5*0.4*0.5=1. The entire directly edited row is nonnegative.
        np.testing.assert_allclose(edited["raw_normed_values"], [np.log(3), 0., 0.7], rtol=1e-6)
        negative = perturber.apply_custom_gene_edits(row, {"B": 1.}, {"a": 1, "b": 2, "c": 3})
        self.assertEqual(float(negative["raw_normed_values"][1]), 0.)


@unittest.skipIf(bool(MISSING), "Install the runtime dependencies: " + ", ".join(MISSING))
class DualLineTests(unittest.TestCase):
    def run_trajectory(self, remote_states, **options):
        bank = TwoSpotBank()
        model = FakeModel()
        calls = []
        baseline = {
            0: {"gene_ids": np.array([1, 2]), "raw_normed_values": np.array([np.log(2), -0.1], dtype=np.float32)},
            1: {"gene_ids": np.array([1, 2]), "raw_normed_values": np.zeros(2, dtype=np.float32)},
        }
        fixed = {
            0: {"gene_ids": np.array([1, 2]), "raw_normed_values": np.array([5., 5.], dtype=np.float32)},
            1: {"gene_ids": np.array([1, 2]), "raw_normed_values": np.array([2., 2.], dtype=np.float32)},
        }

        def predict(data_bank, loader, passed_model, device, config, description):
            self.assertTrue(passed_model.eval_called)
            calls.append(description)
            if description == "X_0 baseline":
                return baseline
            if description == "X fixed baseline":
                np.testing.assert_array_equal(data_bank.overrides[1]["raw_normed_values"], [0., 0.])
                return fixed
            step = int(description.rsplit(" ", 1)[-1])
            np.testing.assert_allclose(data_bank.overrides[0]["raw_normed_values"], [np.log(3), 0.], rtol=1e-6)
            # The fixed-control residual gives the requested remote state.
            return {
                0: {"gene_ids": np.array([1, 2]), "raw_normed_values": np.array([100., -100.], dtype=np.float32)},
                1: {"gene_ids": np.array([1, 2]), "raw_normed_values": np.array([2. + remote_states[step]] * 2, dtype=np.float32)},
            }

        kwargs = dict(max_steps=len(remote_states), patience=10, emergency_div_factor=100.,
                      device="cpu", num_workers=0, custom_gene_edits={"A": 1.})
        kwargs.update(options)
        with mock.patch.object(perturber, "load_vocab", return_value={"a": 1, "b": 2}), \
             mock.patch.object(perturber, "_run_single_pass", side_effect=predict):
            result = perturber.run_perturbation(model, bank, SimpleNamespace(vocab_file="unused"), [0], [0, 1], **kwargs)
        self.assertEqual(calls.count("X_0 baseline"), 1)
        self.assertEqual(calls.count("X fixed baseline"), 1)
        self.assertEqual(len(calls), result.total_steps + 2)
        self.assertEqual(bank.loader_calls, len(calls))
        for state in result.step_expressions.values():
            np.testing.assert_allclose(state[0]["raw_normed_values"], [np.log(3), 0.], rtol=1e-6)
        return result

    def test_selects_consecutive_mse_not_effect_mse(self):
        result = self.run_trajectory([0.1, 0.19, 0.27])
        np.testing.assert_allclose(result.step_mse, [0.01, 0.0081, 0.0064], rtol=1e-5)
        np.testing.assert_array_equal(result.step_mse, result.update_mse)
        self.assertEqual(result.selected_step, 2)  # Saved steps are zero-indexed in memory.
        self.assertEqual(int(np.argmin(result.effect_mse)), 0)
        self.assertAlmostEqual(result.best_mse, 0.0064, places=6)

    def test_nonzero_threshold_does_not_discard_true_argmin(self):
        result = self.run_trajectory([0.1, 0.19, 0.27], convergence_threshold=0.005, patience=2)
        self.assertEqual(result.selected_step, 2)
        self.assertEqual(result.total_steps, 3)
        self.assertTrue(result.stop_reason.startswith("auto_best_patience"))
        self.assertEqual(result.selected_step, int(np.argmin(result.update_mse)))

    def test_fixed_step_selects_requested_state(self):
        result = self.run_trajectory([0.1, 0.19, 0.27], stopping_mode="fixed_steps", fixed_step=2)
        self.assertEqual(result.selected_step, 1)
        np.testing.assert_allclose(result.step_expressions[result.selected_step][1]["raw_normed_values"], [0.19, 0.19], rtol=1e-6)

    def test_unfrozen_negative_updates_are_preserved(self):
        result = self.run_trajectory([-0.1, -0.18, -0.24])
        for i, expected in enumerate([-0.1, -0.18, -0.24]):
            np.testing.assert_allclose(result.step_expressions[i][1]["raw_normed_values"], [expected] * 2, rtol=1e-6)
        self.assertEqual(result.selected_step, 2)

    def test_no_edit_preserves_reconstructed_baseline(self):
        bank = TwoSpotBank(source=[1., 2.], remote=[3., 4.])
        model = FakeModel()
        calls = []

        def affine_predict(data_bank, loader, passed_model, device, config, description):
            calls.append(description)
            return {index: {"gene_ids": row["gene_ids"].copy(),
                            "raw_normed_values": 0.5 * row["raw_normed_values"] + 0.2}
                    for index, row in data_bank.overrides.items()}

        with mock.patch.object(perturber, "load_vocab", return_value={"a": 1, "b": 2}), \
             mock.patch.object(perturber, "_run_single_pass", side_effect=affine_predict):
            result = perturber.run_perturbation(model, bank, SimpleNamespace(vocab_file="unused"), [], [0, 1],
                                               max_steps=3, stopping_mode="fixed_steps", device="cpu", num_workers=0)
        self.assertEqual(len(calls), 5)
        np.testing.assert_array_equal(result.step_mse, [0., 0., 0.])
        for state in result.step_expressions.values():
            for index in (0, 1):
                np.testing.assert_array_equal(state[index]["raw_normed_values"], result.baseline_expressions[index]["raw_normed_values"])


@unittest.skipIf(bool(MISSING) or importlib.util.find_spec("anndata") is None,
                 "DEG CSV integration requires the complete runtime including anndata")
class DegCsvTests(unittest.TestCase):
    def test_direction_only_csv_has_unit_strength(self):
        from spatialgt_agent_ui.ui_utils.data_io import parse_deg_csv
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "directions.csv"
            path.write_text("gene,direction\nA,up\nB,down\n")
            parsed = parse_deg_csv(path)
        np.testing.assert_array_equal(parsed["avg_logFC"].to_numpy(), [1., -1.])


if __name__ == "__main__":
    unittest.main()

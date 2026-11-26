"""Source sampling and inference configuration checks."""

import argparse
import ast
import contextlib
import io
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np


SOURCE = Path(__file__).resolve().parents[1] / "perturbation/mouse_stroke/spatialgt_perturbation.py"


def load_functions():
    tree = ast.parse(SOURCE.read_text())
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef)
             and node.name in {"select_random_spots", "main"}]
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    scope = {"argparse": argparse, "random": random, "np": np, "Path": Path}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[future] + nodes, type_ignores=[])),
                 str(SOURCE), "exec"), scope)
    return scope


class ConfigCaptured(Exception):
    pass


class MouseProtocolTests(unittest.TestCase):
    def setUp(self):
        self.scope = load_functions()
        self.captured_config = None

    def test_source_selection_matches_expected_draw(self):
        ids = [129, 1, 65, 33, 97, 161, 193, 225, 257, 289]
        chosen = self.scope["select_random_spots"](ids, 4, 17)
        self.assertEqual(chosen, [257, 193, 33, 97])
        self.assertNotEqual(chosen, random.Random(17).sample(sorted(set(ids)), 4))

    def test_duplicate_ids_are_sampled_without_replacement(self):
        ids = [3, 5, 7, 3, 5, 11]
        chosen = self.scope["select_random_spots"](ids, 3, 17)
        self.assertEqual(len(chosen), len(set(chosen)))
        self.assertTrue(set(chosen).issubset(ids))

    def test_all_spots_selection(self):
        ids = [11, 3, 3, 7, 5]
        for count in (0, -1, 4, 9):
            self.assertEqual(set(self.scope["select_random_spots"](ids, count, 17)), {3, 5, 7, 11})

    def run_until_databank(self, arguments):
        self.scope["torch"] = SimpleNamespace(
            manual_seed=lambda seed: None,
            cuda=SimpleNamespace(is_available=lambda: False),
            device=lambda name: name,
        )
        self.scope["Config"] = lambda: SimpleNamespace(max_seq_len=4000)
        self.scope["_setup_config_for_cache"] = lambda config, **kwargs: config

        def capture(**kwargs):
            self.captured_config = kwargs["config"]
            raise ConfigCaptured

        self.scope["SpatialDataBank"] = capture
        with tempfile.TemporaryDirectory() as directory:
            argv = [str(SOURCE), "--ckpt_control", "weights", "--cache_sham", "cache",
                    "--roi_manifest", "regions.json", "--deg_csv", "genes.csv",
                    "--out_dir", directory] + arguments
            with mock.patch("sys.argv", argv):
                self.scope["main"]()

    def test_seed_is_required(self):
        with contextlib.redirect_stderr(io.StringIO()) as error:
            with self.assertRaises(SystemExit) as raised:
                self.run_until_databank([])
        self.assertEqual(raised.exception.code, 2)
        self.assertIn("--seed", error.getvalue())
        self.assertIsNone(self.captured_config)

    def test_default_padding_reaches_databank(self):
        with self.assertRaises(ConfigCaptured):
            self.run_until_databank(["--seed", "17"])
        self.assertEqual(self.captured_config.max_seq_len, 3000)

    def test_custom_padding_reaches_databank(self):
        with self.assertRaises(ConfigCaptured):
            self.run_until_databank(["--seed", "17", "--max_seq_len", "512"])
        self.assertEqual(self.captured_config.max_seq_len, 512)

    def test_invalid_padding_fails_before_data_access(self):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as raised:
                self.run_until_databank(["--seed", "17", "--max_seq_len", "0"])
        self.assertEqual(raised.exception.code, 2)
        self.assertIsNone(self.captured_config)


if __name__ == "__main__":
    unittest.main()

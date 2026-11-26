"""CPU checks for within-niche convergence and expression metrics."""

import contextlib
import io
from types import SimpleNamespace
import unittest

import numpy as np

from test_experiment_protocols import functions_from_script


SCRIPT = "perturbation/microenvironment/microenv_perturbation.py"


def state(values):
    return {i: {"gene_ids": np.array([1, 2]),
                "raw_normed_values": np.full(2, v, dtype=np.float32)}
            for i, v in enumerate(values)}


class WithinConvergenceTests(unittest.TestCase):
    def run_trajectory(self, values, patience):
        predictions = iter(state(row) for row in values)
        bank = SimpleNamespace(clear_runtime_spot_overrides=lambda: None,
                               set_runtime_spot_overrides=lambda values: None)
        torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
        scope = functions_from_script(SCRIPT,
            {"compute_step_mse", "run_dual_line_inference"},
            {"torch": torch, "run_full_inference": lambda *args, **kwargs: next(predictions)})
        with contextlib.redirect_stdout(io.StringIO()):
            return scope["run_dual_line_inference"](
                bank, lambda: None, None, None, max_steps=len(values),
                frozen_indices={2}, x0_state=state([0, 0, 7]),
                x0_prime_state=state([0, 0, 7]),
                precomputed_x_line=state([0, 0, 0]), patience=patience,
            )

    def test_mean_mse_selects_step_and_excludes_frozen_spots(self):
        steps, _, info = self.run_trajectory(
            [[0, 4, 1000], [2.5, 6.5, -1000], [5.5, 9.5, 2000]], patience=1)
        self.assertEqual(info["best_step"], 2)
        self.assertEqual(info["actual_steps"], 3)
        self.assertEqual(info["final_status"], "stopped_patience")
        self.assertEqual(info["best_mse"], 6.25)
        self.assertEqual([row["mse"] for row in info["mse_records"]], [8.0, 6.25, 9.0])
        self.assertTrue(all(row["n_spots_compared"] == 2 for row in info["mse_records"]))
        for prediction in steps.values():
            np.testing.assert_array_equal(prediction[2]["raw_normed_values"], [7, 7])

    def test_emergency_stop_uses_mse_ratio(self):
        _, _, info = self.run_trajectory([[1, 1, 7], [2.8, 2.8, 7], [3, 3, 7]], patience=10)
        self.assertEqual(info["best_step"], 1)
        self.assertEqual(info["actual_steps"], 2)
        self.assertEqual(info["final_status"], "emergency_stop")
        self.assertGreater(info["mse_records"][-1]["mse"], 3 * info["best_mse"])

    def test_expression_evaluation_reports_rmse(self):
        scope = functions_from_script(SCRIPT, {"compute_spot_similarity"})
        ids = np.arange(1, 11)
        target = np.arange(10, dtype=np.float32)
        result = scope["compute_spot_similarity"](target + 2, ids, target, ids)
        self.assertEqual(result["rmse"], 2.0)
        self.assertAlmostEqual(result["pearson"], 1.0, places=6)

    def test_center_convergence_reports_squared_change(self):
        scope = functions_from_script(SCRIPT, {"compute_step_convergence"})
        result = scope["compute_step_convergence"]({1: state([1]), 2: state([3])}, 0)
        self.assertEqual(result, [{"step_from": 1, "step_to": 2, "mse": 4.0}])


if __name__ == "__main__":
    unittest.main()

"""Step-to-step MSE convergence detection."""

from __future__ import annotations
from typing import List, Tuple


def check_convergence(
    step_mse: List[float],
    threshold: float = 1e-5,
    patience: int = 2,
) -> Tuple[bool, int]:
    """
    Check if MSE has converged.

    Returns (converged: bool, best_step: int).
    Convergence = the absolute change in MSE between consecutive steps
    has been below `threshold` for `patience` consecutive steps.
    """
    if len(step_mse) < patience + 1:
        return False, len(step_mse) - 1

    streak = 0
    for i in range(1, len(step_mse)):
        delta = abs(step_mse[i] - step_mse[i - 1])
        if delta < threshold:
            streak += 1
            if streak >= patience:
                return True, i
        else:
            streak = 0

    return False, len(step_mse) - 1


def find_best_step(step_mse: List[float]) -> int:
    """Find the step with the lowest MSE."""
    if not step_mse:
        return 0
    return int(min(range(len(step_mse)), key=lambda i: step_mse[i]))

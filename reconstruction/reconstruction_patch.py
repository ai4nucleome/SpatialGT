#!/usr/bin/env python3
"""Seeded patch selection for reconstruction evaluation."""

from __future__ import annotations

import random
from collections import deque
from typing import Any, Dict, List, Optional, Set, Tuple


def select_patch_spots(
    neighbors: Dict[int, List[int]],
    seed_spot: int,
    patch_size: int,
) -> List[int]:
    """Select a contiguous patch of spots via breadth-first expansion."""
    visited = {seed_spot}
    queue = deque([seed_spot])
    selected = [seed_spot]

    while queue and len(selected) < patch_size:
        current = queue.popleft()
        for neighbor in neighbors.get(current, []):
            if neighbor not in visited:
                visited.add(neighbor)
                selected.append(neighbor)
                queue.append(neighbor)
                if len(selected) >= patch_size:
                    break

    return selected[:patch_size]


def _directed_reachable_vertices(
    neighbors: Dict[int, List[int]],
    seed: int,
    allowed: Set[int],
) -> List[int]:
    """Return breadth-first visit order within the allowed vertices."""
    if seed not in allowed:
        return []
    visited: List[int] = []
    seen = {seed}
    queue = deque([seed])
    while queue:
        current = queue.popleft()
        visited.append(current)
        for neighbor in neighbors.get(current, []):
            if neighbor in allowed and neighbor not in seen:
                seen.add(neighbor)
                queue.append(neighbor)
    return visited


def _subgraph(neighbors: Dict[int, List[int]], allowed: Set[int]) -> Dict[int, List[int]]:
    """Restrict neighbor lists to edges whose targets stay inside allowed."""
    return {vertex: [target for target in neighbors.get(vertex, []) if target in allowed] for vertex in allowed}


def select_seeded_patch_center_and_mask(
    neighbors: Dict[int, List[int]],
    n_spots_requested: int,
    rng_seed: int,
    allowed_vertices: Optional[Set[int]] = None,
) -> Tuple[int, List[int], Dict[str, Any]]:
    """Pick a patch center using a seed and return the selected mask indices."""
    rng = random.Random(int(rng_seed))
    allowed: Set[int] = set(allowed_vertices) if allowed_vertices is not None else set(neighbors.keys())
    if not allowed:
        raise ValueError("select_seeded_patch_center_and_mask: empty allowed_vertices")

    neighbor_subgraph = _subgraph(neighbors, allowed)
    n_total = len(allowed)
    meta: Dict[str, Any] = {
        "n_spots_requested": int(n_spots_requested),
        "n_vertices_in_slice": int(n_total),
        "patch_center_mode": "seeded",
    }

    if n_spots_requested >= n_total:
        vertices = sorted(allowed)
        center = vertices[rng.randrange(len(vertices))]
        meta["n_spots_effective"] = len(vertices)
        meta["warning"] = "n_spots_requested >= slice size; masking all spots"
        meta["center_reachable_size"] = n_total
        return center, vertices, meta

    reach_sizes = {
        vertex: len(_directed_reachable_vertices(neighbor_subgraph, vertex, allowed))
        for vertex in allowed
    }
    feasible = [vertex for vertex in allowed if reach_sizes[vertex] >= n_spots_requested]

    if feasible:
        feasible.sort()
        center = feasible[rng.randrange(len(feasible))]
        mask = select_patch_spots(neighbor_subgraph, center, n_spots_requested)
        meta["n_spots_effective"] = len(mask)
        meta["center_reachable_size"] = reach_sizes[center]
        meta["warning"] = None
        return center, mask, meta

    max_reachable = max(reach_sizes.values()) if reach_sizes else 0
    if max_reachable <= 0:
        raise RuntimeError("select_seeded_patch_center_and_mask: degenerate graph")

    candidates = sorted(vertex for vertex in allowed if reach_sizes[vertex] == max_reachable)
    center = candidates[rng.randrange(len(candidates))]
    n_effective = min(int(n_spots_requested), int(max_reachable))
    mask = select_patch_spots(neighbor_subgraph, center, n_effective)
    meta["n_spots_effective"] = len(mask)
    meta["center_reachable_size"] = max_reachable
    meta["warning"] = (
        f"No seed reaches n_spots_requested={n_spots_requested}; "
        f"using max reachable={max_reachable} (effective n_spots={len(mask)})"
    )
    return center, mask, meta

"""
Wraps preprocess_v3.py logic into a callable module for single h5ad files.

Main entry: preprocess_h5ad(h5ad_path, cache_dir, config)
"""

from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pretrain.Config import Config


def preprocess_h5ad(
    h5ad_path: str | Path,
    cache_dir: str | Path,
    config: Optional[Config] = None,
    max_neighbors: int = 8,
    force_rebuild: bool = False,
    progress_callback=None,
) -> Path:
    """
    Preprocess a single h5ad file for SpatialGT inference.

    Steps:
      1. Copy h5ad into cache layout
      2. Compute spatial neighbors (k-NN)
      3. Build sparse adjacency matrix
      4. Preprocess spot data (gene embeddings, etc.)
      5. Write metadata.json

    Returns the cache_dir path.
    """
    h5ad_path = Path(h5ad_path).resolve()
    cache_dir = Path(cache_dir).resolve()
    dataset_name = h5ad_path.stem

    if config is None:
        from .model_manager import make_config
        config = make_config(cache_dir=str(cache_dir))

    _report(progress_callback, 0.0, "Loading data...")

    import anndata as ad
    from scipy.sparse import issparse

    adata = ad.read_h5ad(str(h5ad_path))
    n_spots = adata.n_obs
    n_genes = adata.n_vars

    # Step 1: set up cache directory layout
    # SpatialDataBank derives dataset name from the h5ad filename stem,
    # so we must use <dataset_name>.h5ad (not processed.h5ad)
    ds_dir = cache_dir / dataset_name
    ds_dir.mkdir(parents=True, exist_ok=True)
    processed_path = ds_dir / f"{dataset_name}.h5ad"
    if not processed_path.exists() or force_rebuild:
        shutil.copy2(str(h5ad_path), str(processed_path))

    _report(progress_callback, 0.15, f"Data loaded: {n_spots} spots, {n_genes} genes")

    # Step 2: compute spatial neighbors
    _report(progress_callback, 0.2, "Computing spatial neighbors...")
    spatial = np.array(adata.obsm["spatial"], dtype=np.float64)
    all_neighbors, all_distances = _compute_neighbors(spatial, max_neighbors)

    # Step 3: save neighbors
    _report(progress_callback, 0.4, "Saving neighbor graph...")
    neighbors_dir = cache_dir / "neighbors"
    neighbors_dir.mkdir(parents=True, exist_ok=True)
    _save_neighbors_hdf5(
        neighbors_dir, dataset_name, max_neighbors,
        all_neighbors, all_distances, n_spots,
    )

    # Step 4: build adjacency matrix
    _report(progress_callback, 0.5, "Building adjacency matrix...")
    _save_adjacency(ds_dir, all_neighbors, n_spots)

    # Step 5: preprocess spots via SpatialDataBank
    _report(progress_callback, 0.6, "Preprocessing spot data (gene embeddings)...")

    config.cache_dir = str(cache_dir)
    config.dataset_list = None
    config.dataset_paths = [str(processed_path)]
    config.max_neighbors = max_neighbors

    spots_file = cache_dir / dataset_name / "spots.h5"
    need_spots = not spots_file.exists() or force_rebuild

    if need_spots:
        # Delete stale metadata so SpatialDataBank runs full init
        stale_meta = cache_dir / "metadata.json"
        if stale_meta.exists():
            stale_meta.unlink()

        prev_strict = getattr(config, "strict_cache_only", False)
        config.strict_cache_only = False

        from pretrain.spatial_databank import SpatialDataBank
        databank = SpatialDataBank(
            dataset_paths=[str(processed_path)],
            cache_dir=str(cache_dir),
            config=config,
            force_rebuild=True,
        )

        if not spots_file.exists():
            _report(progress_callback, 0.7, "Building spots.h5 cache...")
            databank.preprocess_dataset_spots(0)

        config.strict_cache_only = prev_strict

        if not spots_file.exists():
            raise RuntimeError(f"spots.h5 was not created at {spots_file}.")
    else:
        _report(progress_callback, 0.8, "spots.h5 already exists, skipping spot preprocessing.")

    _report(progress_callback, 1.0, "Preprocessing complete (h5 mode)!")
    return cache_dir


def build_lmdb(
    cache_dir: str | Path,
    config: Optional[Config] = None,
    progress_callback=None,
) -> tuple:
    """
    Build LMDB cache from an already-preprocessed h5 cache.
    Returns (lmdb_path, manifest_path).
    """
    cache_dir = Path(cache_dir).resolve()
    lmdb_dir = cache_dir / "lmdb"
    lmdb_dir.mkdir(parents=True, exist_ok=True)
    lmdb_path = lmdb_dir / "spatial_cache.lmdb"
    manifest_path = lmdb_dir / "spatial_cache.manifest.json"

    if config is None:
        from .model_manager import make_config
        config = make_config(cache_dir=str(cache_dir), cache_mode="h5")

    _report(progress_callback, 0.0, "Building LMDB cache...")

    config.cache_mode = "h5"
    config.cache_dir = str(cache_dir)
    config.strict_cache_only = True

    h5ad_files = list(cache_dir.glob("*/*.h5ad"))
    if not h5ad_files:
        raise FileNotFoundError(f"No .h5ad files found in {cache_dir}")

    config.dataset_paths = [str(p) for p in h5ad_files]

    # Verify spots.h5 exists before trying to build LMDB
    for h5ad_path in h5ad_files:
        ds_name = h5ad_path.parent.name
        spots_file = cache_dir / ds_name / "spots.h5"
        if not spots_file.exists():
            raise FileNotFoundError(
                f"spots.h5 not found at {spots_file}. "
                f"Run h5 preprocessing first before building LMDB."
            )

    from pretrain.spatial_databank import SpatialDataBank

    _report(progress_callback, 0.1, "Initializing databank (h5 mode)...")
    databank = SpatialDataBank(
        dataset_paths=config.dataset_paths,
        cache_dir=str(cache_dir),
        config=config,
        force_rebuild=False,
    )

    total_spots = databank.metadata.get("total_spots", 0)
    if total_spots == 0:
        for ds in databank.metadata.get("datasets", []):
            total_spots += ds.get("n_spots", 0)
    _report(progress_callback, 0.2, f"Found {total_spots} spots, writing to LMDB...")

    import pickle
    import lmdb

    max_seq_len = int(getattr(config, "max_seq_len", 3000) or 3000)
    approx_per_record = max_seq_len * 4 * 2 + 4096
    map_size = int(min(max(approx_per_record * max(total_spots, 1) * 2, 32 * 1024 ** 3), 4 * 1024 ** 4))
    env = lmdb.open(
        str(lmdb_path),
        map_size=map_size,
        max_readers=128,
        subdir=False,
        lock=False,
        readahead=True,
        meminit=False,
    )

    started = time.time()
    commit_interval = 2048
    skipped = []

    txn = env.begin(write=True)
    for idx in range(total_spots):
        try:
            center_data = databank.get_spot_data(idx)
            try:
                neighbor_indices = databank.get_neighbors_for_spot(idx)
            except Exception:
                neighbor_indices = []

            record = {
                "gene_ids": np.asarray(center_data["gene_ids"], dtype=np.int32),
                "raw_normed_values": np.asarray(center_data["raw_normed_values"], dtype=np.float32),
            }
            meta_fields = ["celltype", "celltype_id", "batch_id", "platform", "organ", "species"]
            record["meta"] = {
                "global_idx": idx,
                **{k: center_data.get(k, "") for k in meta_fields},
            }
            if neighbor_indices:
                record["neighbor_indices"] = np.asarray(neighbor_indices, dtype=np.int32)

            key = f"{idx:012d}".encode("ascii")
            txn.put(key, pickle.dumps(record, protocol=pickle.HIGHEST_PROTOCOL))

            if (idx + 1) % commit_interval == 0:
                txn.commit()
                txn = env.begin(write=True)
                pct = 0.2 + 0.7 * (idx + 1) / total_spots
                _report(progress_callback, pct, f"LMDB: {idx + 1}/{total_spots} spots...")

        except Exception as e:
            skipped.append({"idx": idx, "reason": f"error:{e}"})

    txn.commit()
    env.sync()
    env.close()

    manifest = {
        "schema_version": 1,
        "lmdb_path": str(lmdb_path.resolve()),
        "map_size": map_size,
        "total_spots": total_spots,
        "datasets": databank.metadata.get("datasets", []),
        "metadata": databank.metadata,
        "build_duration_sec": time.time() - started,
        "skipped_records": skipped,
        "config": {
            "cache_dir": str(cache_dir),
            "max_seq_len": max_seq_len,
            "mask_value": getattr(config, "mask_value", None),
        },
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    _report(progress_callback, 1.0,
            f"LMDB built: {total_spots - len(skipped)} spots, {len(skipped)} skipped")
    return str(lmdb_path), str(manifest_path)


def _report(cb, progress: float, message: str):
    if cb:
        cb(progress, message)
    else:
        print(f"[Preprocess {progress:.0%}] {message}")


def _compute_neighbors(spatial: np.ndarray, k: int):
    """Compute k nearest neighbors for each spot."""
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    coords = torch.tensor(spatial, dtype=torch.float32, device=device)
    n = coords.shape[0]

    all_neighbors = {}
    all_distances = {}
    batch_size = min(5000, n)

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch = coords[start:end]
        dists = torch.cdist(batch, coords, p=2)
        for j in range(end - start):
            dists[j, start + j] = float("inf")
        _, indices = torch.topk(dists, k, dim=1, largest=False)
        top_dists = torch.gather(dists, 1, indices)

        idx_np = indices.cpu().numpy()
        dist_np = top_dists.cpu().numpy()
        for j in range(end - start):
            gidx = start + j
            all_neighbors[gidx] = idx_np[j].tolist()
            all_distances[gidx] = dist_np[j].tolist()

    return all_neighbors, all_distances


def _save_neighbors_hdf5(
    neighbors_dir: Path, dataset_name: str, k: int,
    all_neighbors: dict, all_distances: dict, n_spots: int,
):
    import h5py
    fname = neighbors_dir / f"neighbors_{dataset_name}_n{k}_v3.h5"
    with h5py.File(str(fname), "w") as f:
        # SpatialDataBank.get_neighbors_for_spot expects one dataset per spot
        # under a top-level "neighbors" group.
        neighbors_grp = f.create_group("neighbors")
        distances_grp = f.create_group("neighbor_distances")
        nb_arr = np.zeros((n_spots, k), dtype=np.int64)
        dist_arr = np.zeros((n_spots, k), dtype=np.float32)
        for i in range(n_spots):
            nbs = all_neighbors.get(i, [])
            ds = all_distances.get(i, [])
            nb_arr[i, :] = -1
            dist_arr[i, :] = np.inf
            nb_arr[i, :len(nbs)] = nbs
            dist_arr[i, :len(ds)] = ds
            neighbors_grp.create_dataset(str(i), data=nb_arr[i])
            distances_grp.create_dataset(str(i), data=dist_arr[i])
        # Keep matrix-style datasets too for tools that inspect the file directly.
        f.create_dataset("neighbor_indices", data=nb_arr)
        f.create_dataset("neighbor_distances_matrix", data=dist_arr)
        f.attrs["dataset_name"] = dataset_name
        f.attrs["max_neighbors"] = k
        f.attrs["n_spots"] = n_spots

    index_path = neighbors_dir / "neighbors_index.json"
    index_data = {}
    if index_path.exists():
        with open(index_path) as fj:
            index_data = json.load(fj)
    index_data["0"] = str(fname.name)
    index_data[dataset_name] = str(fname.name)
    with open(index_path, "w") as fj:
        json.dump(index_data, fj, indent=2)


def _save_adjacency(ds_dir: Path, all_neighbors: dict, n_spots: int):
    from scipy.sparse import lil_matrix, save_npz
    adj = lil_matrix((n_spots, n_spots), dtype=np.float32)
    for i, nbs in all_neighbors.items():
        for nb in nbs:
            adj[i, nb] = 1.0
            adj[nb, i] = 1.0
    save_npz(str(ds_dir / "adjacency_sparse.npz"), adj.tocsr())


def _write_metadata(cache_dir: Path, dataset_name: str, n_spots: int, n_genes: int, k: int):
    """Write metadata.json only if SpatialDataBank hasn't created one."""
    meta_path = cache_dir / "metadata.json"
    if meta_path.exists():
        return
    meta = {
        "datasets": [{"name": dataset_name, "n_spots": n_spots, "n_genes": n_genes,
                       "cache_dir": str(cache_dir / dataset_name)}],
        "dataset_indices": [{"dataset_idx": 0, "start_idx": 0, "end_idx": n_spots}],
        "total_spots": n_spots,
        "max_neighbors": k,
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

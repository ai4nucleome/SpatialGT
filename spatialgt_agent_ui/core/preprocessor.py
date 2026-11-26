"""Single-section expression preprocessing, spatial graphs and runtime caches."""

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
    """Preprocess one section, then build its graph in the retained barcode order."""
    import anndata as ad
    import h5py

    h5ad_path = Path(h5ad_path).resolve()
    cache_dir = Path(cache_dir).resolve()
    dataset_name = h5ad_path.stem
    if config is None:
        from .model_manager import make_config
        config = make_config(cache_dir=str(cache_dir), cache_mode="h5")

    _report(progress_callback, 0.0, "Loading data...")
    adata = ad.read_h5ad(str(h5ad_path))
    if not adata.obs_names.is_unique:
        raise ValueError("Spot barcodes must be unique before preprocessing.")
    if "spatial" not in adata.obsm:
        raise ValueError("The input section needs obsm['spatial'] coordinates.")
    source_barcodes = adata.obs_names.astype(str).tolist()
    ds_dir = cache_dir / dataset_name
    ds_dir.mkdir(parents=True, exist_ok=True)
    input_copy = ds_dir / f"{dataset_name}.h5ad"
    if (not input_copy.exists() or force_rebuild) and input_copy != h5ad_path:
        shutil.copy2(str(h5ad_path), str(input_copy))
    processed_path = ds_dir / "processed.h5ad"
    spots_file = ds_dir / "spots.h5"
    metadata_file = cache_dir / "metadata.json"
    need_spots = (force_rebuild or not spots_file.exists() or
                  not processed_path.exists() or not metadata_file.exists())

    config.cache_dir = str(cache_dir)
    config.cache_mode = "h5"
    config.dataset_list = None
    config.dataset_paths = [str(input_copy)]
    config.max_neighbors = int(max_neighbors)
    databank = None
    prev_strict = getattr(config, "strict_cache_only", False)
    try:
        if need_spots:
            _report(progress_callback, 0.2, "Preprocessing section expression...")
            if metadata_file.exists():
                metadata_file.unlink()
            # The row ordering may change after quality filtering.
            for split_path in cache_dir.glob("split_v*_s*_r*.json"):
                split_path.unlink()
            config.strict_cache_only = False
            from pretrain.spatial_databank import SpatialDataBank
            databank = SpatialDataBank(
                dataset_paths=[str(input_copy)], cache_dir=str(cache_dir),
                config=config, force_rebuild=True,
            )
        if not processed_path.exists():
            raise RuntimeError("Preprocessing produced no valid section. Check the input expression and gene vocabulary.")

        processed = ad.read_h5ad(str(processed_path), backed="r")
        try:
            retained_barcodes = processed.obs_names.astype(str).tolist()
            spatial = np.asarray(processed.obsm["spatial"], dtype=np.float64)
        finally:
            processed.file.close()
        source_lookup = {barcode: i for i, barcode in enumerate(source_barcodes)}
        if len(set(retained_barcodes)) != len(retained_barcodes) or any(
            barcode not in source_lookup for barcode in retained_barcodes
        ):
            raise ValueError("Cached section barcodes do not match this input; use a new cache directory or force rebuild.")
        n_retained = len(retained_barcodes)
        with open(metadata_file) as handle:
            metadata = json.load(handle)
        if len(metadata.get("datasets", [])) != 1 or metadata.get("total_spots") != n_retained:
            raise ValueError("The single-section cache metadata does not match the processed barcodes; force rebuild the cache.")

        _report(progress_callback, 0.5, f"Building the graph for {n_retained} retained spots...")
        all_neighbors, all_distances = _compute_neighbors(spatial, int(max_neighbors))
        neighbors_dir = cache_dir / "neighbors"
        neighbors_dir.mkdir(parents=True, exist_ok=True)
        _save_neighbors_hdf5(neighbors_dir, dataset_name, int(max_neighbors),
                            all_neighbors, all_distances, n_retained)
        _save_adjacency(ds_dir, all_neighbors, n_retained)

        if need_spots:
            _report(progress_callback, 0.7, "Building spot expression cache...")
            databank.preprocess_dataset_spots(0)
        if not spots_file.exists():
            raise RuntimeError(f"spots.h5 was not created at {spots_file}.")
        with h5py.File(spots_file, "r") as handle:
            spot_ids = {int(key) for key in handle["spots"].keys()}
        if spot_ids != set(range(n_retained)):
            raise ValueError("Spot cache indices do not match the processed section; force rebuild the cache.")
        mapping = {
            "dataset_name": dataset_name,
            "source_spots": len(source_barcodes),
            "retained_spots": n_retained,
            "filtered_spots": len(source_barcodes) - n_retained,
            "retained_barcodes": retained_barcodes,
            "source_indices": [source_lookup[barcode] for barcode in retained_barcodes],
        }
        with open(ds_dir / "spot_index_mapping.json", "w") as handle:
            json.dump(mapping, handle, indent=2)
    finally:
        config.strict_cache_only = prev_strict
        if databank is not None:
            databank.cleanup_resources()

    _report(progress_callback, 1.0,
            f"Preprocessing complete: {n_retained} retained spots, {len(source_barcodes) - n_retained} filtered.")
    return cache_dir


def align_session_to_cache(session, cache_dir: str | Path, dataset_name: str) -> dict:
    """Align displayed rows and selected spots to the processed barcode order."""
    import anndata as ad

    adata = session.get("adata")
    if adata is None:
        raise ValueError("Load the section before aligning its processed spots.")
    if not adata.obs_names.is_unique:
        raise ValueError("Spot barcodes must be unique to align the processed section.")
    processed_path = Path(cache_dir) / dataset_name / "processed.h5ad"
    processed = ad.read_h5ad(str(processed_path), backed="r")
    try:
        retained = processed.obs_names.astype(str).tolist()
    finally:
        processed.file.close()
    source = adata.obs_names.astype(str)
    positions = source.get_indexer(retained)
    if len(set(retained)) != len(retained) or np.any(positions < 0):
        raise ValueError("The displayed section and processed cache have different spot barcodes; reload the matching input section.")
    selected = session.get("selected_spots") or []
    if any(int(i) < 0 or int(i) >= adata.n_obs for i in selected):
        raise ValueError("A selected spot index is outside the displayed section.")
    lookup = {barcode: i for i, barcode in enumerate(retained)}
    aligned_selection = [lookup[source[int(i)]] for i in selected if source[int(i)] in lookup]
    changed = list(source) != retained
    aligned = adata[positions].copy() if changed else adata
    session["adata"] = aligned
    session["selected_spots"] = aligned_selection
    session["inference_results"] = None
    session["databank"] = None
    label_column = session.get("label_column")
    if label_column and label_column in aligned.obs:
        session["label_counts"] = {
            str(key): int(value) for key, value in aligned.obs[label_column].value_counts(dropna=False).items()
        }
    return {"retained_spots": aligned.n_obs, "removed_spots": adata.n_obs - aligned.n_obs,
            "removed_selected_spots": len(selected) - len(aligned_selection)}


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
    if k < 1 or n <= k:
        raise ValueError(f"At least {k + 1} retained spots are required for {k} neighbors; got {n}.")
    if not torch.isfinite(coords).all():
        raise ValueError("Spatial coordinates must be finite.")

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

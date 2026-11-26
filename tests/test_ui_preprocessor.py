"""Single-section preprocessing and barcode-alignment tests."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

DEPS = ("anndata", "scanpy", "h5py", "lmdb", "pandas", "torch", "numpy")
MISSING = [name for name in DEPS if importlib.util.find_spec(name) is None]
if not MISSING:
    import anndata as ad
    import h5py
    import numpy as np
    import pandas as pd
    from pretrain.Config import Config
    from pretrain.spatial_databank import SpatialDataBank
    from spatialgt_agent_ui.core.preprocessor import (
        preprocess_h5ad, align_session_to_cache, build_lmdb,
    )


@unittest.skipIf(bool(MISSING), "Full preprocessing dependencies required: " + ", ".join(MISSING))
class PreprocessingAlignmentTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        rng = np.random.default_rng(42)
        counts = rng.poisson(3., size=(12, 120)).astype(np.float32)
        counts[3] = 0
        counts[3, :4] = 2  # Removed by the native min_genes=10 filter.
        counts[9] = 0
        names = [f"spot{i:02d}" for i in range(12)]
        self.adata = ad.AnnData(counts, obs=pd.DataFrame({"Region": names}, index=names),
                              var=pd.DataFrame(index=[f"G{i}" for i in range(120)]))
        self.adata.obsm["spatial"] = rng.uniform(0, 1000, size=(12, 2))
        self.input = self.root / "tissue.h5ad"
        self.adata.write_h5ad(self.input)
        self.cache = self.root / "cache"
        self.cfg = Config()
        self.cfg.cache_mode = "h5"
        self.cfg.device = "cpu"
        self.cfg.max_neighbors = 8
        self.cfg.subset_hvg = 0  # This fixture isolates native QC and indexing.
        self.cfg.vocab_file = str(self.root / "vocab.json")
        self.cfg.max_seq_len = 128
        vocab = {"[PAD]": 0, **{f"G{i}": i + 1 for i in range(120)}}
        Path(self.cfg.vocab_file).write_text(json.dumps(vocab))

    def tearDown(self):
        self.tmp.cleanup()

    def preprocess(self):
        preprocess_h5ad(self.input, self.cache, self.cfg, max_neighbors=8,
                        force_rebuild=True, progress_callback=lambda *args: None)

    def test_filtered_barcodes_graph_and_ui_selection_stay_aligned(self):
        self.preprocess()
        expected = [i for i in range(12) if i not in (3, 9)]
        processed = ad.read_h5ad(self.cache / "tissue" / "processed.h5ad")
        self.assertEqual(processed.obs_names.tolist(), self.adata.obs_names[expected].tolist())
        mapping = json.loads((self.cache / "tissue" / "spot_index_mapping.json").read_text())
        self.assertEqual(mapping["source_indices"], expected)
        self.assertEqual(mapping["filtered_spots"], 2)
        with h5py.File(self.cache / "neighbors" / "neighbors_tissue_n8_v3.h5") as handle:
            graph = handle["neighbor_indices"][:]
        coords = self.adata.obsm["spatial"][expected]
        distance = np.sum((coords[:, None, :] - coords[None, :, :]) ** 2, axis=2)
        np.fill_diagonal(distance, np.inf)
        np.testing.assert_array_equal(graph, np.argsort(distance, axis=1)[:, :8])
        with h5py.File(self.cache / "tissue" / "spots.h5") as handle:
            self.assertEqual(len(handle["spots"]), 10)
            for i in range(10):
                np.testing.assert_allclose(handle["spots"][str(i)]["spatial_coords"][:], coords[i])
        state = {"adata": self.adata.copy(), "selected_spots": [2, 3, 4, 9, 11],
                 "label_column": "Region", "inference_results": object()}
        report = align_session_to_cache(state, self.cache, "tissue")
        self.assertEqual(state["selected_spots"], [2, 3, 9])
        self.assertEqual(state["adata"].obs_names.tolist(), processed.obs_names.tolist())
        self.assertEqual(report, {"retained_spots": 10, "removed_spots": 2, "removed_selected_spots": 2})
        self.assertIsNone(state["inference_results"])
        # Re-entering preprocessing must not move already aligned selections.
        again = align_session_to_cache(state, self.cache, "tissue")
        self.assertEqual(state["selected_spots"], [2, 3, 9])
        self.assertEqual(again["removed_spots"], 0)

    def test_lmdb_uses_filtered_indices_and_rebuilt_neighbors(self):
        self.preprocess()
        lmdb_path, manifest = build_lmdb(self.cache, self.cfg, progress_callback=lambda *args: None)
        saved = json.loads(Path(manifest).read_text())
        self.assertEqual(saved["total_spots"], 10)
        self.assertEqual(saved["skipped_records"], [])
        self.cfg.cache_mode = "lmdb"
        self.cfg.lmdb_path = lmdb_path
        self.cfg.lmdb_manifest_path = manifest
        bank = SpatialDataBank([], str(self.cache), self.cfg)
        try:
            for i in range(10):
                row = bank.get_spot_data(i)
                neighbors = list(row["neighbor_indices"])
                self.assertEqual(len(neighbors), 8)
                self.assertNotIn(i, neighbors)
                self.assertTrue(all(0 <= index < 10 for index in neighbors))
                self.assertEqual(row["global_idx"], i)
        finally:
            bank.reset_lmdb_env()

    def test_unfiltered_section_keeps_row_identity(self):
        self.adata.X[3] = 2
        self.adata.X[9] = 3
        self.adata.write_h5ad(self.input)
        self.preprocess()
        state = {"adata": self.adata, "selected_spots": [3, 9]}
        report = align_session_to_cache(state, self.cache, "tissue")
        self.assertIs(state["adata"], self.adata)
        self.assertEqual(state["selected_spots"], [3, 9])
        self.assertEqual(report["removed_spots"], 0)


if __name__ == "__main__":
    unittest.main()

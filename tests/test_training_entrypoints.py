"""Training entrypoint, configuration, vocabulary and sampling tests.

Run with: python -m unittest discover -s tests -p 'test_training_entrypoints.py'
"""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pretrain.Config import Config
from finetune.Config import FinetuneConfig
from pretrain._vocab import Vocab
from spatialgt_agent_ui.core import finetuner

HAS_TRAINING_STACK = all(importlib.util.find_spec(name) is not None
                         for name in ("scanpy", "anndata", "pandas", "accelerate"))


class TrainingEntryTests(unittest.TestCase):
    def test_configs_cover_data_contract(self):
        required = ("filter_gene_by_counts", "subset_hvg", "input_style",
                    "include_zero_gene", "max_seq_len", "mask_ratio",
                    "mask_value", "pad_value", "pad_token", "cls_token")
        for config in (Config(), FinetuneConfig()):
            for name in required:
                self.assertTrue(hasattr(config, name), name)
            self.assertEqual(config.pad_token, "[PAD]")
            self.assertEqual(config.max_seq_len, 4000)
            self.assertTrue(config.include_zero_gene)
            self.assertEqual(config.gene_embedding_dim, 3072)

    def test_ui_launches_existing_cli(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(finetuner.subprocess, "Popen") as popen:
                finetuner.run_finetune("model", "cache", directory, "slice", device="cpu")
                cmd = popen.call_args[0][0]
                self.assertTrue(Path(cmd[1]).is_file())
                self.assertEqual(Path(cmd[1]), ROOT / "finetune" / "finetune.py")
                self.assertNotIn("--use_existing_cache", cmd)
                self.assertEqual(cmd[cmd.index("--device") + 1], "cpu")

    def test_vocab_keeps_indices_and_default(self):
        vocab = Vocab(["[PAD]", "Mbp", "PCP4"])
        vocab.set_default_index(0)
        self.assertEqual(vocab(["PCP4", "Mbp", "missing"]), [2, 1, 0])
        self.assertEqual(Vocab(vocab.vocab).get_stoi(), vocab.get_stoi())
        with self.assertRaises(ValueError):
            vocab.insert_token("bad", 20)

    def test_ui_prefers_completed_training_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for step in (100, 200):
                checkpoint = root / f"checkpoint-{step}"
                checkpoint.mkdir()
                (checkpoint / "model.safetensors").touch()
            self.assertEqual(finetuner.find_latest_checkpoint(root), root / "checkpoint-200")
            (root / "model.safetensors").touch()
            self.assertEqual(finetuner.find_latest_checkpoint(root), root)

    @unittest.skipUnless(HAS_TRAINING_STACK, "full training dependencies required")
    def test_explicit_cuda_device_maps_visible_index(self):
        from finetune.finetune import _configure_device_environment
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "2,7", "WORLD_SIZE": "1"}):
            self.assertEqual(_configure_device_environment(["--device", "cuda:1"]), "cuda:0")
            self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "7")
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "2,7", "WORLD_SIZE": "2"}):
            self.assertEqual(_configure_device_environment(["--device", "cuda:1"]), "cuda")
            self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "2,7")

    @unittest.skipUnless(HAS_TRAINING_STACK, "full training dependencies required")
    def test_cli_help_in_fresh_process(self):
        for script in ("finetune/finetune.py", "pretrain/run_pretrain.py", "pretrain/preprocess.py"):
            with tempfile.TemporaryDirectory() as directory:
                result = subprocess.run([sys.executable, str(ROOT / script), "--help"],
                                        cwd=directory, capture_output=True, text=True, timeout=120)
            self.assertEqual(result.returncode, 0, script + "\n" + result.stderr)
            self.assertIn("usage:", result.stdout)

    @unittest.skipUnless(HAS_TRAINING_STACK, "full training dependencies required")
    def test_preprocessing_accepts_files_and_directories(self):
        from pretrain.preprocess import read_dataset_list
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "sections").mkdir()
            section = root / "sections" / "slice.h5ad"
            section.touch()
            listing = root / "datalist.txt"
            listing.write_text("# inputs\nsections/slice.h5ad\nsections\n")
            self.assertEqual(read_dataset_list(listing), [str(section)])
            listing.write_text("missing.h5ad\n")
            with self.assertRaises(FileNotFoundError):
                read_dataset_list(listing)
            listing.write_text("# no inputs\n")
            with self.assertRaises(ValueError):
                read_dataset_list(listing)

    @unittest.skipUnless(HAS_TRAINING_STACK, "full training dependencies required")
    def test_gene_vocab_json_roundtrip(self):
        from pretrain.gene_tokenizer import GeneVocab
        expected = {"[PAD]": 0, "G2": 1, "G1": 2}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "vocab.json"
            GeneVocab.from_dict(expected, default_token="[PAD]").save_json(path)
            self.assertEqual(GeneVocab.from_file(path).get_stoi(), expected)

    @unittest.skipUnless(HAS_TRAINING_STACK, "full training dependencies required")
    def test_distributed_sampler_uses_process_rank(self):
        from pretrain.spatial_databank import SpatialDataBank
        with tempfile.TemporaryDirectory() as directory:
            vocab_path = Path(directory) / "vocab.json"
            vocab_path.write_text(json.dumps({"[PAD]": 0, "G1": 1}))
            bank = SpatialDataBank.__new__(SpatialDataBank)
            bank.config = Config()
            bank.config.vocab_file = str(vocab_path)
            bank.cache_mode = "h5"
            bank.get_data_split = lambda **kw: (list(range(8)), [])
            samples = []
            for rank in (0, 1):
                with patch.dict(os.environ, {"WORLD_SIZE": "2", "RANK": str(rank)}):
                    loader = bank.get_data_loader(batch_size=2, num_workers=0,
                                                  use_distributed=True, shuffle=False)
                    self.assertEqual(loader.sampler.num_replicas, 2)
                    self.assertEqual(loader.sampler.rank, rank)
                    samples.append(set(loader.sampler))
                    loader.set_epoch(7)
                    self.assertEqual(loader.sampler.epoch, 7)
            self.assertFalse(samples[0] & samples[1])
            self.assertEqual(samples[0] | samples[1], set(range(8)))


if __name__ == "__main__":
    unittest.main()

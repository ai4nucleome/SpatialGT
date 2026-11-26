"""CPU checks for model initialization and checkpoint loading."""

import contextlib
import json
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import torch

from pretrain.model_spatialpt import SpatialNeighborTransformer
from pretrain.Config import Config
from finetune.Config import FinetuneConfig
from pretrain.checkpoint import load_model_checkpoint, resolve_checkpoint_file
from spatialgt_agent_ui.core import model_manager


def tiny_config(**overrides):
    options = dict(num_genes=32, gene_embedding_dim=16, use_gene_projection=True,
                   d_model=8, num_heads=2, num_layers=1, dropout=0.0)
    options.update(overrides)
    return SimpleNamespace(**options)


def example_batch():
    return dict(
        genes=torch.tensor([[5, 6, 7], [5, 6, 7], [5, 6, 7]]),
        input_values=torch.tensor([[.3, .2, .1], [.4, .5, .1], [.9, .7, .1]]),
        padding_attention_mask=torch.ones(3, 3, dtype=torch.bool),
        center_indices=[2], batch_to_global_map=[(0, 2)],
    )


class ModelCheckpointTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        torch.manual_seed(42)

    def tearDown(self):
        self.tmp.cleanup()

    def test_default_structure_matches_released_240m_checkpoint(self):
        cfg = tiny_config(num_genes=59483, gene_embedding_dim=3072,
                          d_model=512, num_heads=8, num_layers=8, dropout=.1)
        with torch.device("meta"):
            model = SpatialNeighborTransformer(cfg)
        self.assertEqual(sum(p.numel() for p in model.parameters()), 240040539)
        self.assertEqual(tuple(model.gene_embedding_layer.weight.shape), (59483, 3072))
        self.assertEqual(tuple(model.gene_pretrained_projection.weight.shape), (512, 3072))

    def test_model_initialization_uses_learnable_embeddings(self):
        cfg = tiny_config()
        with mock.patch("torch.load", side_effect=AssertionError("Unexpected external matrix read")):
            log = io.StringIO()
            with contextlib.redirect_stdout(log):
                model = SpatialNeighborTransformer(cfg)
        self.assertEqual(log.getvalue(), "")
        self.assertTrue(model.gene_embedding_layer.weight.requires_grad)
        self.assertTrue(torch.all(model.gene_embedding_layer.weight[0] == 0))
        pred, target, mask, _ = model(**example_batch())
        loss = (pred[mask] - target[mask]).square().mean()
        loss.backward()
        self.assertTrue(torch.isfinite(pred).all())
        self.assertGreater(model.gene_embedding_layer.weight.grad.abs().sum().item(), 0)

    def test_load_restores_outputs_and_infers_checkpoint_shapes(self):
        source = SpatialNeighborTransformer(tiny_config()).eval()
        checkpoint = self.root / "checkpoint"
        checkpoint.mkdir()
        torch.save({"state_dict": source.state_dict()}, checkpoint / "pytorch_model.bin")
        wrong_dimensions = tiny_config(num_genes=19, gene_embedding_dim=8, num_layers=2)
        restored = load_model_checkpoint(self.root, wrong_dimensions).eval()
        self.assertEqual(restored.config.gene_embedding_dim, 16)
        self.assertEqual(restored.config.num_genes, 32)
        self.assertEqual(restored.config.num_layers, 1)
        self.assertTrue(torch.equal(source(**example_batch())[0], restored(**example_batch())[0]))
        self.assertEqual(wrong_dimensions.num_genes, 19)  # Caller config is not mutated.
        for key, value in source.state_dict().items():
            self.assertTrue(torch.equal(value, restored.state_dict()[key]), key)

    def test_incomplete_checkpoint_is_rejected(self):
        state = SpatialNeighborTransformer(tiny_config()).state_dict()
        state.pop("reconstruction_head.0.bias")
        path = self.root / "finetuned_state_dict.pth"
        torch.save(state, path)
        with self.assertRaisesRegex(RuntimeError, "Missing key"):
            load_model_checkpoint(path, tiny_config())

    def test_missing_embedding_is_rejected_before_instantiation(self):
        state = SpatialNeighborTransformer(tiny_config()).state_dict()
        state.pop("gene_embedding_layer.weight")
        path = self.root / "finetuned_state_dict.pth"
        torch.save(state, path)
        with self.assertRaisesRegex(ValueError, "missing gene_embedding_layer"):
            load_model_checkpoint(path, tiny_config())

    def test_custom_vocab_is_used_without_network(self):
        path = self.root / "custom_vocab.json"
        mapping = {"[PAD]": 0, "PCP4": 1, "Mbp": 2}
        path.write_text(json.dumps(mapping))
        with mock.patch("huggingface_hub.hf_hub_download", side_effect=AssertionError), \
             mock.patch("huggingface_hub.snapshot_download", side_effect=AssertionError):
            self.assertEqual(model_manager.ensure_gene_vocabulary(path), path.resolve())
            config = model_manager.make_config(vocab_file=path, device="cpu")
        self.assertEqual(config.vocab_file, str(path.resolve()))
        self.assertEqual(json.loads(path.read_text()), mapping)

    def test_missing_vocab_reports_required_local_file(self):
        path = self.root / "missing_vocab.json"
        with mock.patch("huggingface_hub.hf_hub_download", side_effect=AssertionError), \
             mock.patch("huggingface_hub.snapshot_download", side_effect=AssertionError):
            with self.assertRaisesRegex(FileNotFoundError, "Clone the complete SpatialGT"):
                model_manager.make_config(vocab_file=path, device="cpu")
        self.assertFalse(path.exists())

    def test_default_vocab_is_bundled_and_matches_model_size(self):
        expected_path = Path(__file__).resolve().parents[1] / "vocab.json"
        with mock.patch("huggingface_hub.hf_hub_download", side_effect=AssertionError), \
             mock.patch("huggingface_hub.snapshot_download", side_effect=AssertionError):
            self.assertEqual(model_manager.ensure_gene_vocabulary(), expected_path)
            ui_config = model_manager.make_config(device="cpu")
        for config in (Config(), FinetuneConfig(), ui_config):
            self.assertEqual(Path(config.vocab_file), expected_path)
        mapping = json.loads(expected_path.read_text())
        self.assertEqual(len(mapping), Config().num_genes)
        self.assertEqual(set(mapping.values()), set(range(Config().num_genes)))
        self.assertEqual(mapping["[PAD]"], 0)

    def test_hf_download_returns_nested_checkpoint_directory(self):
        checkpoint = self.root / "checkpoint"
        checkpoint.mkdir()
        (checkpoint / "model.safetensors").touch()
        with mock.patch.object(model_manager, "download_from_hf", return_value=self.root) as download:
            actual = model_manager.download_pretrained(self.root)
        self.assertEqual(actual, checkpoint)
        patterns = download.call_args.kwargs["allow_patterns"]
        self.assertIn("checkpoint/model.safetensors", patterns)
        self.assertNotIn("*", patterns)
        self.assertEqual(resolve_checkpoint_file(self.root), checkpoint / "model.safetensors")

    def test_ui_load_local_checkpoint_does_not_fetch_gene_assets(self):
        source = SpatialNeighborTransformer(tiny_config()).eval()
        torch.save(source.state_dict(), self.root / "finetuned_state_dict.pth")
        with mock.patch.object(model_manager, "ensure_gene_vocabulary", side_effect=AssertionError):
            loaded = model_manager.load_model(self.root, config=tiny_config(), device="cpu")
        self.assertFalse(loaded.training)
        self.assertTrue(torch.equal(source(**example_batch())[0], loaded(**example_batch())[0]))


if __name__ == "__main__":
    unittest.main()

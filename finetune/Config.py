"""
SpatialGT Finetuning Configuration

This module defines the configuration class for SpatialGT finetuning.
Default settings are optimized for full fine-tuning with all 8 transformer layers.
"""

import json
from pathlib import Path
import torch


class FinetuneConfig:
    """Configuration class for SpatialGT finetuning."""
    
    def __init__(self):
        # ============ Path Configuration ============
        # Project root is the parent directory of finetune/
        self.project_root = Path(__file__).resolve().parent.parent
        
        # Pretrained model checkpoint (required)
        # Users should set this to their pretrained checkpoint path
        self.base_checkpoint = None
        
        # Gene embedding files (inherited from pretrain)
        self.vocab_file = str(self.project_root / "gene_embedding" / "vocab.json")
        self.pretrained_gene_embeddings_path = str(
            self.project_root / "gene_embedding" / "pretrained_gene_embeddings.pt"
        )
        
        # ============ Data Configuration ============
        # Dataset paths (list of h5ad files or folders)
        self.dataset_paths = []
        
        # Cache directory for preprocessed data
        self.cache_dir = None
        
        # Cache mode: "h5" (single-worker safe) or "lmdb" (multi-worker friendly)
        self.cache_mode = "h5"
        self.lmdb_path = None
        self.lmdb_manifest_path = None
        self.lmdb_map_size_gb = 1024
        self.lmdb_max_readers = 1024
        
        # ============ Output Configuration ============
        self.output_dir = str(self.project_root / "output" / "finetune")
        
        # ============ Model Architecture ============
        # These should match the pretrained model
        self.model_name = "spatialgt"
        self.d_model = 512
        self.num_heads = 8
        self.num_layers = 8  # Total transformer layers
        self.gene_encoder_layers = 6
        self.decoder_layers = 4
        self.dropout = 0.1
        
        # ============ Fine-tuning Strategy ============
        # Number of transformer layers to unfreeze (default: all 8 for full fine-tuning)
        self.unfreeze_last_n_layers = 8
        # Always unfreeze reconstruction head
        self.unfreeze_head = True
        
        # ============ Training Configuration ============
        self.batch_size = 64
        self.eval_batch_size = 64
        self.num_epochs = 100  # Default 100 epochs for fine-tuning
        self.learning_rate = 1e-4  # Slightly lower than pretraining
        self.weight_decay = 1e-5
        self.num_warmup_steps = 100
        self.max_steps = -1  # -1 means use num_epochs
        self.accumulation_steps = 1
        
        # ============ Validation ============
        self.validation_split = 0.0  # No validation by default for fine-tuning
        self.validation_interval = 500
        
        # ============ Spatial Configuration ============
        self.max_neighbors = 8
        self.distance_threshold = 1800
        self.adjacency_type = "value"
        self.stratify_by_batch = False
        
        # ============ DataLoader Configuration ============
        self.num_workers = 4
        self.pin_memory = True
        
        # ============ Logging & Checkpointing ============
        self.print_interval = 10
        self.checkpoint_interval = 500
        self.save_total_limit = 10  # Keep last 10 checkpoints
        
        # ============ Device & Precision ============
        self.device = "cuda"
        self.dtype = torch.float16
        self.precision = "fp16"
        
        # ============ Reproducibility ============
        self.random_seed = 42
        
        # ============ Internal Flags ============
        self.strict_cache_only = True

    def to_dict(self):
        """Convert config to dictionary for serialization."""
        def convert(obj):
            if isinstance(obj, (str, int, float, bool, type(None))):
                return obj
            elif isinstance(obj, Path):
                return str(obj)
            elif isinstance(obj, (list, tuple)):
                return [convert(i) for i in obj]
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            elif hasattr(obj, '__name__') and not isinstance(obj, type):
                return obj.__name__
            else:
                return str(obj)
        return {k: convert(v) for k, v in self.__dict__.items()}

    def to_json_string(self):
        """Return JSON string representation of config."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)
    
    def save(self, path: str):
        """Save configuration to a JSON file."""
        with open(path, 'w') as f:
            f.write(self.to_json_string())
    
    @classmethod
    def load(cls, path: str) -> 'FinetuneConfig':
        """Load configuration from a JSON file."""
        config = cls()
        with open(path, 'r') as f:
            data = json.load(f)
        for key, value in data.items():
            if hasattr(config, key):
                setattr(config, key, value)
        return config


# Alias for backward compatibility
Config = FinetuneConfig

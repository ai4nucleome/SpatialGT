"""
SpatialGT Pretraining Configuration

This module defines the configuration class for SpatialGT pretraining.
All paths are relative to the project root to ensure portability.
"""

import json
from pathlib import Path
import torch


class Config:
    """Configuration class for SpatialGT pretraining."""
    
    def __init__(self):
        # ============ Path Configuration ============
        # Project root is the parent directory of pretrain/
        self.project_root = Path(__file__).resolve().parent.parent
        
        # Gene embedding files (required for model initialization)
        self.vocab_file = str(self.project_root / "gene_embedding" / "vocab.json")
        self.pretrained_gene_embeddings_path = str(
            self.project_root / "gene_embedding" / "pretrained_gene_embeddings.pt"
        )
        
        # ============ Data Configuration ============
        # Dataset list file containing paths to .h5ad files (one folder path per line)
        self.dataset_list = str(self.project_root / "example_data" / "HBRC" / "datalist.txt")
        
        # Cache directory for preprocessed data
        self.cache_dir = str(self.project_root / "example_data" / "HBRC" / "HBRC_preprocessed")
        
        # LMDB cache configuration (for efficient data loading)
        self.cache_mode = "lmdb"  # Options: "h5", "lmdb"
        self.lmdb_path = str(
            self.project_root / "example_data" / "HBRC" / "HBRC_lmdb" / "spatial_cache.lmdb"
        )
        self.lmdb_manifest_path = str(
            self.project_root / "example_data" / "HBRC" / "HBRC_lmdb" / "spatial_cache.manifest.json"
        )
        self.lmdb_map_size_gb = 1024
        self.lmdb_max_readers = 1024
        self.copy_lmdb_to_shm = False  # Copy LMDB to shared memory for faster access
        self.shared_cache_dir = "/dev/shm/spatialgt_lmdb"
        
        # ============ Output Configuration ============
        self.output_dir = str(self.project_root / "output" / "pretrain")
        self.resume_from_checkpoint = None  # Path to checkpoint for resuming training
        
        # ============ Model Architecture ============
        self.model_name = "spatialgt"
        self.d_model = 512              # Transformer hidden dimension
        self.num_heads = 8              # Number of attention heads
        self.num_layers = 8             # Number of transformer encoder layers
        self.gene_encoder_layers = 6    # Number of gene encoder layers
        self.decoder_layers = 4         # Number of decoder layers
        self.dropout = 0.1              # Dropout rate
        
        # ============ Training Configuration ============
        self.batch_size = 64            # Training batch size
        self.eval_batch_size = 64       # Evaluation batch size
        self.num_epochs = 4             # Number of training epochs
        self.learning_rate = 0.0001     # Initial learning rate
        self.weight_decay = 0.00001     # Weight decay for AdamW
        self.num_warmup_steps = 10      # Learning rate warmup steps
        self.max_steps = None           # Maximum training steps (None for epoch-based)
        self.accumulation_steps = 2     # Gradient accumulation steps
        self.eval_accumulation_steps = 1
        
        # ============ Masking & Loss ============
        self.mask_ratio = 0.4           # Ratio of genes to mask during pretraining
        self.recon_loss_weight = 1.0    # Weight for reconstruction loss
        self.contrast_weight = 1.0      # Weight for contrastive loss
        
        # ============ Spatial Configuration ============
        self.max_neighbors = 8          # Maximum number of spatial neighbors
        self.distance_threshold = 1800  # Distance threshold for neighbor detection
        self.adjacency_type = "value"   # Options: "value", "binary"
        
        # ============ Data Processing ============
        self.input_style = "log1p"      # Input normalization style
        self.validation_split = 0.05    # Fraction of data for validation
        self.stratify_by_batch = False  # Stratify validation split by batch
        
        # ============ DataLoader Configuration ============
        self.num_workers = 4            # Number of data loading workers
        self.dataloader_prefetch_factor = 4
        self.pin_memory = True          # Pin memory for faster GPU transfer
        
        # ============ Logging & Checkpointing ============
        self.print_interval = 10        # Log training status every N steps
        self.validation_interval = 100  # Run validation every N steps
        self.checkpoint_interval = 100  # Save checkpoint every N steps
        self.resource_monitor_interval = 10
        
        # ============ Device & Precision ============
        self.device = "cuda"
        self.dtype = torch.float16
        self.precision = "fp16"         # Options: "fp16", "bf16", "fp32"
        
        # ============ Reproducibility ============
        self.random_seed = 42

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
    def load(cls, path: str) -> 'Config':
        """Load configuration from a JSON file."""
        config = cls()
        with open(path, 'r') as f:
            data = json.load(f)
        for key, value in data.items():
            if hasattr(config, key):
                setattr(config, key, value)
        return config

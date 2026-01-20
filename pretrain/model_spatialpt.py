"""
SpatialGT Model for Spatial Transcriptomics

This module implements the SpatialNeighborTransformer model that uses 
GPT-style causal attention to predict gene expression of center spots
based on their spatial neighbors.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from Config import Config


class SpatialNeighborTransformer(nn.Module):
    """
    Spatial Transcriptomics Foundation Model.
    
    Uses GPT-style causal attention to predict the gene expression of 
    a center spot based on its spatial neighbors.
    
    Architecture:
        1. Gene Embedding: Maps gene IDs to embeddings (optionally pretrained)
        2. Spot Embedding: Aggregates gene embeddings weighted by expression
        3. Transformer Encoder: Processes neighbor-center sequences
        4. Reconstruction Head: Predicts gene expression from hidden states
    """
    
    def __init__(self, config: Config):
        super(SpatialNeighborTransformer, self).__init__()
        self.config = config
        
        # Basic configuration
        self.d_model = config.d_model
        self.num_heads = config.num_heads
        self.num_layers = config.num_layers
        self.dropout = config.dropout
        self.num_genes = 59483  # Total number of genes (including padding at index 0)
        
        # Gene embedding layer
        # Load pretrained gene embeddings if available
        if (hasattr(config, "pretrained_gene_embeddings_path") and 
            os.path.exists(config.pretrained_gene_embeddings_path)):
            # Load pretrained embedding matrix
            pretrained_weights = torch.load(config.pretrained_gene_embeddings_path)
            self.gene_embedding_dim = pretrained_weights.size(1)
            self.num_total_genes = pretrained_weights.size(0)
            padding_idx = 0  # Index 0 is used for padding
            
            # Create embedding layer from pretrained weights
            self.gene_embedding_layer = nn.Embedding.from_pretrained(
                pretrained_weights,
                freeze=False,  # Allow fine-tuning
                padding_idx=padding_idx
            )

            # Project pretrained gene embeddings to model dimension
            self.gene_pretrained_projection = nn.Linear(
                self.gene_embedding_dim, self.d_model
            )
            print(f"Loaded pretrained gene embeddings, dimension: {self.gene_embedding_dim}")
            
        else:
            # Use random initialization if no pretrained embeddings
            self.gene_embedding_layer = nn.Embedding(
                num_embeddings=self.num_genes,
                embedding_dim=self.d_model,
                padding_idx=0
            )
            self.gene_pretrained_projection = None
            print("No pretrained gene embeddings found, using random initialization")
        
        # Learnable mask token (for center spot)
        self.mask_token = nn.Parameter(torch.randn(self.d_model))
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.num_heads,
            dim_feedforward=self.d_model * 4,
            dropout=self.dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True  # Pre-LN for better training stability
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.num_layers,
            norm=nn.LayerNorm(self.d_model)
        )
        
        # Reconstruction head: predict gene expression from hidden states
        self.reconstruction_head = nn.Sequential(
            nn.Linear(self.d_model, self.num_genes)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        # Gene embedding with normal distribution
        nn.init.normal_(self.gene_embedding_layer.weight, mean=0.0, std=0.02)
        
        # Set padding position to zeros
        with torch.no_grad():
            self.gene_embedding_layer.weight[0].fill_(0)
        
        # Mask token initialization
        nn.init.normal_(self.mask_token, mean=0.0, std=0.02)
        
        # Reconstruction head with small initialization
        nn.init.normal_(self.reconstruction_head[-1].weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.reconstruction_head[-1].bias)
    
    def get_spot_embedding(self, gene_ids, expression_values, padding_mask):
        """
        Convert a spot's gene expression to a spot embedding.
        
        Args:
            gene_ids: [batch_size, seq_len] - Gene token IDs
            expression_values: [batch_size, seq_len] - Normalized expression values
            padding_mask: [batch_size, seq_len] - True indicates valid positions
        
        Returns:
            spot_embedding: [batch_size, d_model] - Aggregated spot representation
        """
        # Get gene embeddings: [batch_size, seq_len, gene_embedding_dim]
        gene_embeds = self.gene_embedding_layer(gene_ids)
        
        # Project to model dimension if using pretrained embeddings
        if self.gene_pretrained_projection is not None:
            gene_embeds = self.gene_pretrained_projection(gene_embeds)
        
        # Use expression values as weights for weighted sum
        # Set padding positions to zero weight
        expr_weights = expression_values.clone()
        expr_weights[~padding_mask] = 0.0
        
        # Weighted sum to get spot embedding
        # [batch_size, seq_len, d_model] * [batch_size, seq_len, 1] -> [batch_size, d_model]
        spot_embed = (gene_embeds * expr_weights.unsqueeze(-1)).sum(dim=1)
        
        return spot_embed
    
    def create_causal_mask(self, num_neighbors):
        """
        Create GPT-style causal attention mask.
        
        Args:
            num_neighbors: int - Number of neighbors
        
        Returns:
            mask: [num_neighbors+1, num_neighbors+1] - True means masked (cannot attend)
        
        Mask structure:
               [n1  n2  n3  ... nK  center]
        n1     [0   0   0   ... 0   1    ]  # Neighbors see all neighbors, not center
        n2     [0   0   0   ... 0   1    ]
        ...
        nK     [0   0   0   ... 0   1    ]
        center [0   0   0   ... 0   1    ]  # Center sees all neighbors, not itself
        
        Note: In PyTorch, True = masked, False = can attend
        """
        seq_len = num_neighbors + 1
        mask = torch.zeros(seq_len, seq_len, dtype=torch.bool)
        
        # All positions cannot see the last position (center)
        mask[:, -1] = True
        
        return mask

    def forward(
        self,
        genes,
        input_values,
        padding_attention_mask,
        center_indices=None,
        batch_to_global_map=None,
        **kwargs
    ):
        """
        Forward pass - fully batched version.
        
        Args:
            genes: [B, L] - Gene IDs, B is total spots (centers + all neighbors)
            input_values: [B, L] - Expression values
            padding_attention_mask: [B, L] - True indicates valid positions
            center_indices: list[int] - Row indices of center spots
            batch_to_global_map: list[tuple] - Neighbor range for each center
        
        Returns:
            predictions: [num_centers, num_genes] - Predicted gene expression
            targets: [num_centers, num_genes] - Ground truth gene expression
            valid_masks: [num_centers, num_genes] - Which genes to compute loss on
            center_hiddens: [num_centers, d_model] - Center spot hidden states
        """
        device = genes.device
        num_centers = len(center_indices)
        
        # 1. Get all spot embeddings
        all_spot_embeds = self.get_spot_embedding(
            genes, input_values, padding_attention_mask
        )  # [B, d_model]
        
        # 2. Save center's true expression for loss computation
        center_genes = genes[center_indices]
        center_expressions = input_values[center_indices]
        center_padding_masks = padding_attention_mask[center_indices]
        
        # 3. Batched transformer input construction
        neighbor_counts = [end - start for start, end in batch_to_global_map]
        num_neighbors = neighbor_counts[0]
        assert all(c == num_neighbors for c in neighbor_counts), \
            "All centers must have the same number of neighbors"
        
        # 3.1 Extract all neighbor embeddings at once using advanced indexing
        neighbor_indices = []
        for start, end in batch_to_global_map:
            neighbor_indices.append(torch.arange(start, end, device=device))
        neighbor_indices = torch.stack(neighbor_indices)  # [num_centers, K]
        
        # Index once: [num_centers, K, d_model]
        batched_neighbor_embeds = all_spot_embeds[neighbor_indices]
        
        # 3.2 Center embeddings (use mask token)
        batched_center_embeds = self.mask_token.view(1, 1, -1).expand(
            num_centers, 1, -1
        )
        
        # 3.3 Concatenate: [neighbors..., center]
        batched_seq_embeds = torch.cat(
            [batched_neighbor_embeds, batched_center_embeds], 
            dim=1
        )  # [num_centers, K+1, d_model]
        
        # 4. Batched Transformer encoding
        attn_mask = self.create_causal_mask(num_neighbors).to(device)
        batched_hidden_states = self.transformer_encoder(
            batched_seq_embeds,
            mask=attn_mask
        )
        
        # 5. Batched prediction
        batched_center_hiddens = batched_hidden_states[:, -1, :]
        predictions = self.reconstruction_head(batched_center_hiddens)
        
        # 6. Vectorized target construction
        targets = torch.zeros(num_centers, self.num_genes, device=device)
        valid_masks = torch.zeros(num_centers, self.num_genes, dtype=torch.bool, device=device)
        
        # Use scatter operation
        batch_idx = torch.arange(num_centers, device=device).unsqueeze(1).expand_as(center_genes)
        
        # Only scatter at valid positions
        valid_batch_idx = batch_idx[center_padding_masks]
        valid_gene_idx = center_genes[center_padding_masks]
        valid_expr = center_expressions[center_padding_masks]
        
        targets.index_put_((valid_batch_idx, valid_gene_idx), valid_expr)
        valid_masks.index_put_(
            (valid_batch_idx, valid_gene_idx), 
            torch.ones(1, dtype=torch.bool, device=device)
        )
        
        return predictions, targets, valid_masks, batched_center_hiddens
    
    @torch.no_grad()
    def get_embeddings(
        self, 
        genes, 
        input_values, 
        padding_attention_mask, 
        center_indices,
        use_bidirectional=True
    ):
        """
        Inference mode: Get embeddings for all spots.
        
        Args:
            genes: [num_spots, seq_len] - Gene IDs
            input_values: [num_spots, seq_len] - Expression values
            padding_attention_mask: [num_spots, seq_len] - Valid position mask
            center_indices: Indices of center spots to return
            use_bidirectional: Whether to use bidirectional attention (recommended)
        
        Returns:
            embeddings: [len(center_indices), d_model] - Spot representations
        """
        self.eval()
        device = genes.device
        num_spots = genes.shape[0]
        
        # 1. Get initial embeddings for all spots (using real expression values)
        spot_embeds = self.get_spot_embedding(
            genes, input_values, padding_attention_mask
        )  # [num_spots, d_model]
        
        spot_embeds = spot_embeds.unsqueeze(0)  # [1, num_spots, d_model]
        
        # 2. Select attention mask
        if use_bidirectional:
            # Bidirectional attention: all spots can see each other
            attn_mask = None
        else:
            # Keep causal structure from training (not recommended)
            attn_mask = self.create_causal_mask(num_spots - 1).to(device)
        
        # 3. Transformer encoding
        hidden_states = self.transformer_encoder(
            spot_embeds,
            mask=attn_mask
        )  # [1, num_spots, d_model]
        
        embeddings = hidden_states.squeeze(0)[center_indices]  # [len(center_indices), d_model]
        
        return embeddings

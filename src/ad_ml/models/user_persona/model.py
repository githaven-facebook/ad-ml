"""User Persona Network: sequence encoder + clustering head."""

from __future__ import annotations

import math
from typing import Dict, List, NamedTuple, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class PersonaOutput(NamedTuple):
    """Outputs from a forward pass of UserPersonaNet."""

    user_embedding: Tensor       # (B, embedding_dim)
    cluster_logits: Tensor       # (B, num_segments) — pre-softmax
    cluster_probs: Tensor        # (B, num_segments) — soft assignment
    reconstructed: Tensor        # (B, seq_feature_dim) — reconstructed input for aux loss
    attention_weights: Tensor    # (B, T) — temporal attention over sequence


class MultiHeadTemporalAttention(nn.Module):
    """Multi-head self-attention for weighting sequence timesteps.

    Outputs a weighted sum over the time dimension (B, T, D) -> (B, D).
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads})")
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self._scale = math.sqrt(self.head_dim)

    def forward(
        self,
        x: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
            x: (B, T, D)
            mask: (B, T) bool, True = valid position

        Returns:
            context: (B, D) weighted sum
            attn_weights: (B, T) averaged over heads
        """
        B, T, D = x.shape
        Q = self.q_proj(x)  # (B, T, D)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # Split heads
        Q = Q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, T, dh)
        K = K.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / self._scale  # (B, H, T, T)

        if mask is not None:
            # Expand mask to (B, 1, 1, T) for broadcasting
            mask_expanded = mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(~mask_expanded, float("-inf"))

        attn = F.softmax(scores, dim=-1)  # (B, H, T, T)
        attn = self.dropout(attn)

        # Use [CLS]-style: attend from last valid position -> use mean over query positions
        out = torch.matmul(attn, V)  # (B, H, T, dh)
        out = out.transpose(1, 2).contiguous().view(B, T, D)  # (B, T, D)
        out = self.out_proj(out)

        # Pool to (B, D) using attention over time axis (mean of attn map rows)
        attn_weights = attn.mean(dim=1).mean(dim=-2)  # (B, T) — average over heads and query pos
        context = (out * attn_weights.unsqueeze(-1)).sum(dim=1)  # (B, D)

        return context, attn_weights


class UserPersonaNet(nn.Module):
    """Multi-task user persona network.

    Architecture:
        1. Categorical embedding lookup for user attributes.
        2. GRU encoder over the temporal action sequence.
        3. Multi-head self-attention to weight timestep importance.
        4. MLP projection head -> 128-dim user embedding.
        5. Auxiliary clustering head with Gumbel-Softmax -> soft persona assignment.
        6. Linear reconstruction head for auxiliary reconstruction loss.
    """

    def __init__(
        self,
        seq_feature_dim: int,
        user_feature_dim: int,
        embedding_dim: int = 128,
        hidden_dims: Optional[List[int]] = None,
        num_segments: int = 32,
        gru_hidden_size: int = 256,
        gru_num_layers: int = 2,
        attention_heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.seq_feature_dim = seq_feature_dim
        self.user_feature_dim = user_feature_dim
        self.embedding_dim = embedding_dim
        self.num_segments = num_segments
        hidden_dims = hidden_dims or [256, 128, 64]

        # --- Sequence encoder ---
        self.seq_encoder = nn.GRU(
            input_size=seq_feature_dim,
            hidden_size=gru_hidden_size,
            num_layers=gru_num_layers,
            batch_first=True,
            dropout=dropout if gru_num_layers > 1 else 0.0,
            bidirectional=False,
        )

        # --- Temporal attention ---
        self.attention = MultiHeadTemporalAttention(
            d_model=gru_hidden_size,
            num_heads=attention_heads,
            dropout=dropout,
        )

        # --- User feature projection ---
        self.user_feat_proj = nn.Sequential(
            nn.Linear(user_feature_dim, gru_hidden_size),
            nn.LayerNorm(gru_hidden_size),
            nn.GELU(),
        )

        # --- MLP projection head ---
        mlp_input_dim = gru_hidden_size * 2  # concat: attention_out + user_feat
        layers: List[nn.Module] = []
        prev_dim = mlp_input_dim
        for h_dim in hidden_dims:
            layers += [
                nn.Linear(prev_dim, h_dim),
                nn.LayerNorm(h_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ]
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, embedding_dim))
        self.mlp_head = nn.Sequential(*layers)

        # --- Clustering head ---
        self.cluster_head = nn.Linear(embedding_dim, num_segments)

        # --- Reconstruction head (aux task) ---
        self.reconstruction_head = nn.Linear(embedding_dim, seq_feature_dim)

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.GRU):
                for name, param in module.named_parameters():
                    if "weight" in name:
                        nn.init.orthogonal_(param)
                    elif "bias" in name:
                        nn.init.zeros_(param)

    def forward(
        self,
        sequences: Tensor,
        user_features: Tensor,
        seq_lengths: Tensor,
        attention_mask: Optional[Tensor] = None,
        gumbel_temperature: float = 1.0,
        hard_gumbel: bool = False,
    ) -> PersonaOutput:
        """Forward pass.

        Args:
            sequences: (B, T, seq_feature_dim) padded sequence tensor.
            user_features: (B, user_feature_dim) user-level feature tensor.
            seq_lengths: (B,) actual sequence lengths (for pack_padded_sequence).
            attention_mask: (B, T) bool mask, True = valid; inferred from seq_lengths if None.
            gumbel_temperature: Temperature for Gumbel-Softmax clustering.
            hard_gumbel: If True, use straight-through Gumbel-Softmax.

        Returns:
            PersonaOutput named tuple.
        """
        B, T, _ = sequences.shape

        if attention_mask is None:
            # Build mask from lengths
            idx = torch.arange(T, device=sequences.device).unsqueeze(0)  # (1, T)
            attention_mask = idx < seq_lengths.unsqueeze(1)  # (B, T)

        # --- GRU encoding ---
        packed = nn.utils.rnn.pack_padded_sequence(
            sequences,
            seq_lengths.cpu().clamp(min=1),
            batch_first=True,
            enforce_sorted=False,
        )
        gru_out, _ = self.seq_encoder(packed)
        gru_out, _ = nn.utils.rnn.pad_packed_sequence(
            gru_out, batch_first=True, total_length=T
        )  # (B, T, gru_hidden)

        # --- Temporal attention ---
        attn_context, attn_weights = self.attention(gru_out, attention_mask)  # (B, gru_hidden)

        # --- User features ---
        user_proj = self.user_feat_proj(user_features)  # (B, gru_hidden)

        # --- Fuse and project to embedding ---
        fused = torch.cat([attn_context, user_proj], dim=-1)  # (B, gru_hidden*2)
        user_emb = self.mlp_head(fused)  # (B, embedding_dim)
        user_emb = F.normalize(user_emb, p=2, dim=-1)  # L2-normalize for contrastive

        # --- Clustering ---
        cluster_logits = self.cluster_head(user_emb)  # (B, num_segments)
        cluster_probs = F.gumbel_softmax(
            cluster_logits, tau=gumbel_temperature, hard=hard_gumbel, dim=-1
        )

        # --- Reconstruction (aux) ---
        # Reconstruct mean of input sequence as a proxy for autoencoder loss
        seq_mean = (sequences * attention_mask.unsqueeze(-1).float()).sum(1) / seq_lengths.unsqueeze(-1).float().clamp(min=1)
        reconstructed = self.reconstruction_head(user_emb)  # (B, seq_feature_dim)

        return PersonaOutput(
            user_embedding=user_emb,
            cluster_logits=cluster_logits,
            cluster_probs=cluster_probs,
            reconstructed=reconstructed,
            attention_weights=attn_weights,
        )


class PersonaLoss(nn.Module):
    """Combined loss for user persona training.

    Components:
    - Reconstruction loss (MSE): embedding reconstructs mean sequence input.
    - Clustering loss (KL divergence): cluster assignments vs. uniform target distribution.
    - Contrastive loss (InfoNCE): pull similar users together, push dissimilar apart.
    """

    def __init__(
        self,
        reconstruction_weight: float = 1.0,
        clustering_weight: float = 0.1,
        contrastive_weight: float = 0.05,
        temperature: float = 0.07,
    ) -> None:
        super().__init__()
        self.reconstruction_weight = reconstruction_weight
        self.clustering_weight = clustering_weight
        self.contrastive_weight = contrastive_weight
        self.temperature = temperature

    def forward(
        self,
        output: PersonaOutput,
        target_mean_seq: Tensor,
    ) -> Dict[str, Tensor]:
        """Compute the combined persona training loss.

        Args:
            output: PersonaOutput from UserPersonaNet.forward().
            target_mean_seq: (B, seq_feature_dim) mean sequence features as reconstruction target.

        Returns:
            Dict with keys 'total', 'reconstruction', 'clustering', 'contrastive'.
        """
        # Reconstruction loss
        recon_loss = F.mse_loss(output.reconstructed, target_mean_seq)

        # Clustering KL loss: encourage cluster utilization
        # Target: uniform distribution over clusters (soft)
        cluster_mean = output.cluster_probs.mean(dim=0)  # (K,)
        num_clusters = cluster_mean.shape[0]
        uniform_target = torch.full_like(cluster_mean, 1.0 / num_clusters)
        cluster_loss = F.kl_div(
            cluster_mean.log().clamp(min=-100),
            uniform_target,
            reduction="sum",
        )

        # Contrastive loss (in-batch InfoNCE)
        contrastive_loss = self._infonce_loss(output.user_embedding)

        total = (
            self.reconstruction_weight * recon_loss
            + self.clustering_weight * cluster_loss
            + self.contrastive_weight * contrastive_loss
        )

        return {
            "total": total,
            "reconstruction": recon_loss,
            "clustering": cluster_loss,
            "contrastive": contrastive_loss,
        }

    def _infonce_loss(self, embeddings: Tensor) -> Tensor:
        """In-batch InfoNCE: treats augmented view pairs or just uses all-pairs.

        With no explicit augmentation, uses a self-supervised approximation:
        embeddings are L2-normalized, cosine similarity computed for all pairs,
        and diagonal (self-pairs) treated as positives.
        """
        # embeddings already L2-normalized from model
        sim = torch.matmul(embeddings, embeddings.T) / self.temperature  # (B, B)
        B = sim.shape[0]
        labels = torch.arange(B, device=sim.device)
        # Zero out diagonal for numerical stability before softmax
        sim.fill_diagonal_(float("-inf"))
        # This encourages each sample to be close to itself relative to others
        # (will be near zero if embeddings collapse - acts as regularizer)
        loss = F.cross_entropy(sim, labels)
        return loss

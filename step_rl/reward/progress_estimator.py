"""
Progress Estimator v2.0
- Dense progress reward via MLP regression head on frozen LLM encoder
- Uncertainty estimation (evidential learning or MC Dropout)
- Contrastive ranking loss + monotonicity constraint
"""

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

from step_rl.utils.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class ProgressOutput:
    progress: float = 0.0
    uncertainty: float = 0.0
    encoded: Optional[torch.Tensor] = None


class EvidentialLayer(nn.Module):
    """
    Evidential learning: predict Dirichlet parameters for uncertainty.
    Output: gamma (mean), nu (precision), alpha (shape), beta (scale).
    Uncertainty ~ 1 / nu
    """

    def __init__(self, in_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        self.gamma = nn.Linear(hidden_dim, 1)  # mean
        self.nu = nn.Linear(hidden_dim, 1)  # precision (>0)
        self.alpha = nn.Linear(hidden_dim, 1)  # shape (>1)
        self.beta = nn.Linear(hidden_dim, 1)  # scale (>0)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.shared(x)
        gamma = self.gamma(h)
        nu = F.softplus(self.nu(h)) + 1.0
        alpha = F.softplus(self.alpha(h)) + 1.0
        beta = F.softplus(self.beta(h)) + 1e-6
        return gamma, nu, alpha, beta

    @staticmethod
    def nll_loss(y: torch.Tensor, gamma, nu, alpha, beta) -> torch.Tensor:
        """Evidential negative log-likelihood."""
        omega = 2.0 * beta * (1.0 + nu)
        error = torch.abs(y - gamma)
        nll = (
            0.5 * torch.log(math.pi / nu)
            - alpha * torch.log(omega)
            + (alpha + 0.5) * torch.log(error**2 * nu + omega)
            + torch.lgamma(alpha)
            - torch.lgamma(alpha + 0.5)
        )
        return nll.mean()

    @staticmethod
    def uncertainty(nu: torch.Tensor) -> torch.Tensor:
        return 1.0 / (nu + 1e-6)


class ProgressEstimator(nn.Module):
    """
    Progress Estimator v2.0: predicts task completion progress [0,1]
    with optional uncertainty estimation.
    """

    def __init__(
        self,
        encoder_name: str = "Qwen/Qwen3-8B-Instruct",
        hidden_dim: int = 512,
        num_layers: int = 3,
        dropout: float = 0.1,
        use_uncertainty: bool = True,
        uncertainty_method: str = "evidential",  # evidential, mc_dropout
        freeze_encoder: bool = True,
        device_map: str = "auto",
    ):
        super().__init__()
        self.use_uncertainty = use_uncertainty
        self.uncertainty_method = uncertainty_method
        self.freeze_encoder = freeze_encoder

        # Encoder
        self.encoder = AutoModel.from_pretrained(
            encoder_name,
            dtype=(
                torch.bfloat16
                if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
                else torch.float32
            ),
            device_map=device_map,
        )
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        encoder_dim = self.encoder.config.hidden_size

        # Progress regression head
        layers = []
        in_dim = encoder_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, 1))
        self.progress_head = nn.Sequential(*layers)

        # Uncertainty head
        if use_uncertainty:
            if uncertainty_method == "evidential":
                self.uncertainty_head = EvidentialLayer(encoder_dim, hidden_dim)
            elif uncertainty_method == "mc_dropout":
                self.mc_head = nn.Sequential(
                    nn.Linear(encoder_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout * 2),  # higher dropout for MC
                    nn.Linear(hidden_dim, 1),
                    nn.Sigmoid(),
                )
                self.mc_samples = 10
            else:
                self.uncertainty_head = nn.Sequential(
                    nn.Linear(encoder_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, 1),
                    nn.Sigmoid(),
                )

        self._step_count_embedding = nn.Embedding(100, 64)
        self._projector = nn.Linear(encoder_dim + 64, encoder_dim)

        # Sync custom heads to the encoder's actual device.
        # When device_map="auto" is used, the encoder may land on GPU while
        # nn.Module sub-layers default to CPU. We detect the encoder device
        # and explicitly move all custom heads there.
        self._sync_device()

    def _sync_device(self):
        """Detect the encoder's actual device and move custom heads there."""
        try:
            encoder_device = next(self.encoder.parameters()).device
        except StopIteration:
            return
        # Only move if encoder is on a non-CPU device; CPU is the default anyway.
        if encoder_device.type != "cpu":
            # Move all custom sub-modules to the same device as the encoder.
            # Note: we intentionally do NOT move self.encoder because
            # accelerate's device_map manages its own placement.
            self.progress_head = self.progress_head.to(encoder_device)
            self._step_count_embedding = self._step_count_embedding.to(encoder_device)
            self._projector = self._projector.to(encoder_device)
            if self.use_uncertainty:
                if self.uncertainty_method == "evidential":
                    self.uncertainty_head = self.uncertainty_head.to(encoder_device)
                elif self.uncertainty_method == "mc_dropout":
                    self.mc_head = self.mc_head.to(encoder_device)
                else:
                    self.uncertainty_head = self.uncertainty_head.to(encoder_device)

    def encode_observation(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        step_count: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Get encoder representation with optional step conditioning."""
        device_type = "cuda" if input_ids.is_cuda else "cpu"
        with torch.amp.autocast(device_type=device_type, enabled=True):
            outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            # Mean pool over valid tokens
            mask_expanded = attention_mask.unsqueeze(-1).float()
            pooled = (outputs.last_hidden_state * mask_expanded).sum(
                1
            ) / mask_expanded.sum(1).clamp_min(1e-6)

        if step_count is not None:
            step_emb = self._step_count_embedding(step_count.clamp(0, 99))
            pooled = torch.cat([pooled, step_emb], dim=-1)
            pooled = self._projector(pooled)
        return pooled

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        step_count: Optional[torch.Tensor] = None,
    ) -> ProgressOutput:
        pooled = self.encode_observation(input_ids, attention_mask, step_count)

        if self.use_uncertainty:
            if self.uncertainty_method == "evidential":
                gamma, nu, alpha, beta = self.uncertainty_head(pooled)
                # Use evidential gamma as progress (not progress_head) to avoid
                # training two heads for the same target. uncertainty from nu.
                progress = torch.sigmoid(gamma.squeeze(-1))
                uncertainty = EvidentialLayer.uncertainty(nu).squeeze(-1)
            elif self.uncertainty_method == "mc_dropout":
                # Training: use progress_head for progress, mc_head for uncertainty
                progress = torch.sigmoid(self.progress_head(pooled).squeeze(-1))
                uncertainty = torch.sigmoid(self.mc_head(pooled)).squeeze(-1)
            else:
                # Standard uncertainty head
                progress = torch.sigmoid(self.progress_head(pooled).squeeze(-1))
                uncertainty = self.uncertainty_head(pooled).squeeze(-1)
        else:
            progress = torch.sigmoid(self.progress_head(pooled).squeeze(-1))
            uncertainty = torch.zeros_like(progress)

        return ProgressOutput(
            progress=progress,
            uncertainty=uncertainty,
            encoded=pooled,
        )

    def mc_dropout_predict(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        step_count: Optional[torch.Tensor] = None,
    ) -> ProgressOutput:
        """Monte Carlo dropout inference."""
        if self.uncertainty_method != "mc_dropout":
            return self.forward(input_ids, attention_mask, step_count)

        was_training = self.training
        self.train()  # enable dropout
        preds = []
        try:
            with torch.no_grad():
                for _ in range(self.mc_samples):
                    out = self.forward(input_ids, attention_mask, step_count)
                    preds.append(out.progress.unsqueeze(-1))
        finally:
            if not was_training:
                self.eval()

        preds = torch.cat(preds, dim=-1)  # [B, mc_samples]
        mean = preds.mean(dim=-1)
        var = preds.var(dim=-1)
        return ProgressOutput(progress=mean, uncertainty=var)


# -----------------------------
# Loss functions
# -----------------------------


def ranking_loss(
    progress_i: torch.Tensor,
    progress_j: torch.Tensor,
    target: torch.Tensor,  # 1 if i > j, -1 if i < j
    margin: float = 0.1,
) -> torch.Tensor:
    """Margin ranking loss for contrastive pairs."""
    loss = F.margin_ranking_loss(progress_i, progress_j, target, margin=margin)
    return loss


def monotonicity_loss(
    progress_seq: torch.Tensor,
    weight: float = 1.0,
) -> torch.Tensor:
    """
    Soft monotonicity constraint: progress(t+1) >= progress(t).
    Hinge loss on negative differences.
    """
    diffs = progress_seq[:, 1:] - progress_seq[:, :-1]  # [B, T-1]
    loss = F.relu(-diffs).mean()
    return weight * loss


def progress_estimator_loss(
    model: ProgressEstimator,
    batch: Dict[str, torch.Tensor],
    weights: Dict[str, float],
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Combined loss: MSE + Ranking + Monotonicity + Evidential NLL.
    """
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    step_count = batch.get("step_count")

    # Forward
    out = model(input_ids, attention_mask, step_count)
    pooled = out.encoded

    total_loss = torch.tensor(0.0, device=input_ids.device)
    metrics = {}

    # MSE loss for labeled progress
    if "progress_label" in batch:
        label = batch["progress_label"]
        mse = F.mse_loss(out.progress, label)
        total_loss += weights.get("mse", 1.0) * mse
        metrics["mse"] = mse.item()

    # Ranking loss
    if "pair_indices" in batch:
        pairs = batch["pair_indices"]
        targets = batch.get("pair_targets", torch.ones(pairs.size(0)))
        if pairs.dim() == 2 and pairs.size(1) == 2:
            pi = out.progress[pairs[:, 0]]
            pj = out.progress[pairs[:, 1]]
            rank = ranking_loss(pi, pj, targets.to(pi.device))
            total_loss += weights.get("rank", 0.5) * rank
            metrics["rank"] = rank.item()

    # Monotonicity loss
    if "sequence_mask" in batch:
        # Assume batch grouped by trajectory; sequence_mask is integer trajectory ID
        seq_mask = batch["sequence_mask"]
        unique_ids = seq_mask.unique(sorted=True)
        mono_losses = []
        for tid in unique_ids:
            traj_mask = seq_mask == tid
            traj_progress = out.progress[traj_mask]
            if traj_progress.size(0) > 1:
                # Do NOT sort — keep temporal order for monotonicity constraint
                traj_progress = traj_progress.unsqueeze(0)
                mono_losses.append(monotonicity_loss(traj_progress, weight=1.0))
        if mono_losses:
            mono = torch.stack(mono_losses).mean()
            total_loss += weights.get("mono", 0.3) * mono
            metrics["mono"] = mono.item()

    # Evidential NLL if applicable
    if model.use_uncertainty and model.uncertainty_method == "evidential":
        if "progress_label" in batch and pooled is not None:
            label = batch["progress_label"]
            gamma, nu, alpha, beta = model.uncertainty_head(pooled)
            nll = EvidentialLayer.nll_loss(label.unsqueeze(-1), gamma, nu, alpha, beta)
            total_loss += weights.get("nll", 0.5) * nll
            metrics["nll"] = nll.item()

    metrics["total"] = total_loss.item()
    return total_loss, metrics

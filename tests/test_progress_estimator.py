"""Tests for ProgressEstimator."""

import pytest
import torch

pytest.importorskip("transformers")

from step_rl.reward.progress_estimator import (
    EvidentialLayer,
    ProgressEstimator,
    monotonicity_loss,
    progress_estimator_loss,
    ranking_loss,
)


class TestEvidentialLayer:
    def test_forward_shape(self):
        layer = EvidentialLayer(in_dim=128, hidden_dim=64)
        x = torch.randn(4, 128)
        gamma, nu, alpha, beta = layer(x)
        assert gamma.shape == (4, 1)
        assert nu.shape == (4, 1)
        assert alpha.shape == (4, 1)
        assert beta.shape == (4, 1)
        # Constraints
        assert (nu >= 1.0).all()
        assert (alpha >= 1.0).all()
        assert (beta >= 0.0).all()

    def test_uncertainty(self):
        nu = torch.tensor([[1.0], [10.0]])
        u = EvidentialLayer.uncertainty(nu)
        assert u[0] > u[1]  # lower precision => higher uncertainty

    def test_nll_loss_finite(self):
        _ = EvidentialLayer(in_dim=64, hidden_dim=32)
        y = torch.tensor([0.5, 0.8])
        gamma = torch.tensor([[0.4], [0.7]])
        nu = torch.tensor([[1.0], [2.0]])
        alpha = torch.tensor([[1.5], [1.5]])
        beta = torch.tensor([[0.1], [0.1]])
        loss = EvidentialLayer.nll_loss(y, gamma, nu, alpha, beta)
        assert loss.item() == loss.item()  # finite (not nan)
        assert loss.item() >= 0


class TestProgressEstimator:
    def test_forward_no_uncertainty(self):
        model = ProgressEstimator(
            encoder_name="gpt2",
            use_uncertainty=False,
            freeze_encoder=True,
        )
        device = next(model.encoder.parameters()).device
        input_ids = torch.randint(0, 100, (2, 10)).to(device)
        attention_mask = torch.ones(2, 10, dtype=torch.long).to(device)
        out = model(input_ids, attention_mask)
        assert out.progress.shape == (2,)
        assert out.uncertainty is not None  # returns zeros
        assert (out.uncertainty == 0).all()

    def test_forward_evidential(self):
        model = ProgressEstimator(
            encoder_name="gpt2",
            use_uncertainty=True,
            uncertainty_method="evidential",
            freeze_encoder=True,
        )
        device = next(model.encoder.parameters()).device
        input_ids = torch.randint(0, 100, (2, 10)).to(device)
        attention_mask = torch.ones(2, 10, dtype=torch.long).to(device)
        out = model(input_ids, attention_mask)
        assert out.progress.shape == (2,)
        assert out.uncertainty.shape == (2,)
        assert (out.progress >= 0).all() and (out.progress <= 1).all()
        assert (out.uncertainty >= 0).all()

    def test_mc_dropout_predict(self):
        model = ProgressEstimator(
            encoder_name="gpt2",
            use_uncertainty=True,
            uncertainty_method="mc_dropout",
            freeze_encoder=True,
        )
        device = next(model.encoder.parameters()).device
        input_ids = torch.randint(0, 100, (2, 10)).to(device)
        attention_mask = torch.ones(2, 10, dtype=torch.long).to(device)
        out = model.mc_dropout_predict(input_ids, attention_mask)
        assert out.progress.shape == (2,)
        assert out.uncertainty.shape == (2,)


class TestLossFunctions:
    def test_ranking_loss(self):
        pi = torch.tensor([0.8, 0.3])
        pj = torch.tensor([0.3, 0.8])
        target = torch.tensor([1.0, -1.0])
        loss = ranking_loss(pi, pj, target, margin=0.1)
        assert loss.item() >= 0

    def test_monotonicity_loss(self):
        progress_seq = torch.tensor([[0.2, 0.3, 0.25]])  # 0.25 < 0.3 => violation
        loss = monotonicity_loss(progress_seq, weight=1.0)
        assert loss.item() > 0

        progress_seq2 = torch.tensor([[0.2, 0.3, 0.4]])  # monotonic
        loss2 = monotonicity_loss(progress_seq2, weight=1.0)
        assert loss2.item() == 0.0

    def test_progress_estimator_loss_mse_only(self):
        model = ProgressEstimator(
            encoder_name="gpt2",
            use_uncertainty=False,
            freeze_encoder=True,
        )
        device = next(model.encoder.parameters()).device
        batch = {
            "input_ids": torch.randint(0, 100, (2, 10)).to(device),
            "attention_mask": torch.ones(2, 10, dtype=torch.long).to(device),
            "progress_label": torch.tensor([0.5, 0.8]).to(device),
        }
        weights = {"mse": 1.0}
        loss, metrics = progress_estimator_loss(model, batch, weights)
        assert isinstance(loss, torch.Tensor)
        assert "mse" in metrics
        assert "total" in metrics

    def test_progress_estimator_loss_evidential(self):
        model = ProgressEstimator(
            encoder_name="gpt2",
            use_uncertainty=True,
            uncertainty_method="evidential",
            freeze_encoder=True,
        )
        device = next(model.encoder.parameters()).device
        batch = {
            "input_ids": torch.randint(0, 100, (2, 10)).to(device),
            "attention_mask": torch.ones(2, 10, dtype=torch.long).to(device),
            "progress_label": torch.tensor([0.5, 0.8]).to(device),
        }
        weights = {"mse": 1.0, "nll": 0.5}
        loss, metrics = progress_estimator_loss(model, batch, weights)
        assert isinstance(loss, torch.Tensor)
        assert "nll" in metrics

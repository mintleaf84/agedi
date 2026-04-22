"""Extended tests for Diffusion: LBFGSStepSizer, forward_step, reverse_step,
configure_optimizers, regressor-related paths, and regressor_training property."""
import torch
import pytest
import numpy as np

from agedi.diffusion.diffusion import Diffusion, LBFGSStepSizer, BatchedLBFGSStepSizer
from agedi.data import AtomsGraph


# ── LBFGSStepSizer ───────────────────────────────────────────────────────────

class TestLBFGSStepSizer:
    def _make(self):
        return LBFGSStepSizer(memory_size=5, initial_step=0.1)

    def test_first_step_uses_scaling(self):
        sizer = self._make()
        pos = torch.zeros((4, 3))
        forces = torch.ones((4, 3))
        step = sizer.compute_step(pos, forces)
        assert step.shape == forces.shape

    def test_second_step_uses_lbfgs(self):
        sizer = self._make()
        pos = torch.zeros((4, 3))
        forces = torch.ones((4, 3))
        sizer.compute_step(pos, forces)
        pos2 = pos + 0.01 * forces
        forces2 = forces * 0.9
        step = sizer.compute_step(pos2, forces2)
        assert step.shape == forces2.shape

    def test_reset_clears_memory(self):
        sizer = self._make()
        pos = torch.zeros((4, 3))
        forces = torch.ones((4, 3))
        sizer.compute_step(pos, forces)
        sizer.reset()
        assert sizer.prev_pos is None
        assert len(sizer.s_list) == 0


class TestBatchedLBFGSStepSizer:
    def test_compute_step_correct_shape(self):
        sizer = BatchedLBFGSStepSizer(batch_size=2, memory_size=3, initial_step=0.05)
        pos = torch.zeros((6, 3))
        forces = torch.ones((6, 3))
        batch_idx = torch.tensor([0, 0, 0, 1, 1, 1])
        step = sizer.compute_step(pos, forces, batch_idx)
        assert step.shape == pos.shape

    def test_reset_delegates(self):
        sizer = BatchedLBFGSStepSizer(batch_size=2, memory_size=3, initial_step=0.05)
        pos = torch.zeros((4, 3))
        forces = torch.ones((4, 3))
        batch_idx = torch.tensor([0, 0, 1, 1])
        sizer.compute_step(pos, forces, batch_idx)
        sizer.reset()
        for inner in sizer.step_sizers:
            assert inner.prev_pos is None


# ── forward_step / reverse_step ──────────────────────────────────────────────

def test_forward_step_adds_noise(diffusion, batch):
    diffusion.sample_time(batch)
    pos_before = batch.pos.clone()
    out = diffusion.forward_step(batch)
    assert not torch.allclose(out.pos, pos_before)


def test_reverse_step_changes_positions(diffusion, batch):
    diffusion.score_model.sample_mode()
    diffusion.sample_time(batch)
    diffusion.forward_step(batch)
    pos_noised = batch.pos.clone()
    diffusion.reverse_step(batch, delta_t=torch.tensor(0.001), force_field_guidance=0.0)
    assert not torch.allclose(batch.pos, pos_noised)


# ── configure_optimizers ─────────────────────────────────────────────────────

def test_configure_optimizers_returns_optimizer_and_scheduler(diffusion):
    cfg = diffusion.configure_optimizers()
    assert "optimizer" in cfg
    assert "lr_scheduler" in cfg


# ── setup ────────────────────────────────────────────────────────────────────

def test_setup_puts_score_model_in_training_mode(diffusion):
    diffusion.setup()
    assert not diffusion.score_model.sample


# ── regressor_training property ──────────────────────────────────────────────

def test_regressor_training_returns_false_without_regressor(diffusion):
    assert diffusion.regressor_training is False


def test_regressor_training_setter_ignored_without_regressor(diffusion):
    diffusion.regressor_training = True
    assert diffusion.regressor_training is False


# ── regressor_loss raises without regressor ──────────────────────────────────

def test_regressor_loss_raises_without_regressor(diffusion, batch):
    with pytest.raises(ValueError, match="Regressor model is not defined"):
        diffusion.regressor_loss(batch, None)


# ── sample with > batch_size ─────────────────────────────────────────────────

def test_sample_split_batches(diffusion):
    out = diffusion.sample(
        5,
        batch_size=2,
        steps=3,
        atomic_numbers=[6, 8],
        cell=torch.diag(torch.tensor([8.0, 8.0, 8.0])),
        property={"property": 1.0},
    )
    assert len(out) == 5
    assert all(isinstance(g, AtomsGraph) for g in out)


# ── Alternating training (regressor present) ─────────────────────────────────

@pytest.fixture
def diffusion_with_regressor(diffusion):
    """Diffusion fixture that includes a Forces regressor."""
    from agedi.models.regressor import RegressorModel
    from agedi.models.schnetpack.regressor_heads import Forces

    feature_size = 64
    forces_head = Forces(input_dim_scalar=feature_size, input_dim_vector=feature_size)
    regressor = RegressorModel(
        translator=diffusion.score_model.translator,
        representation=diffusion.score_model.representation,
        heads=[forces_head],
    )
    diffusion.regressor_model = regressor
    return diffusion


def test_configure_optimizers_with_regressor_deduplicates_params(diffusion_with_regressor):
    """Optimizer should cover all unique parameters exactly once."""
    cfg = diffusion_with_regressor.configure_optimizers()
    assert "optimizer" in cfg
    opt_params = {id(p) for p in cfg["optimizer"].param_groups[0]["params"]}

    score_params = {id(p) for p in diffusion_with_regressor.score_model.parameters()}
    reg_params = {id(p) for p in diffusion_with_regressor.regressor_model.parameters()}
    all_unique = score_params | reg_params

    assert opt_params == all_unique


def test_training_step_even_batch_uses_diffusion_loss(diffusion_with_regressor, batch):
    """Even batch_idx should always go through diffusion_loss."""
    diffusion_with_regressor.score_model.training_mode()
    called = {}

    _orig = diffusion_with_regressor.diffusion_loss
    def _patched(b, bi):
        called["diffusion"] = True
        return _orig(b, bi)
    diffusion_with_regressor.diffusion_loss = _patched

    diffusion_with_regressor.training_step(batch, 0)  # even idx
    assert called.get("diffusion"), "diffusion_loss should be called on even batch"


def test_training_step_odd_batch_with_forces_uses_regressor_loss(diffusion_with_regressor, batch):
    """Odd batch_idx + forces present → regressor_loss."""
    diffusion_with_regressor.score_model.training_mode()
    batch.forces = torch.randn_like(batch.pos)

    called = {}
    _orig = diffusion_with_regressor.regressor_loss
    def _patched(b, bi):
        called["regressor"] = True
        return _orig(b, bi)
    diffusion_with_regressor.regressor_loss = _patched

    diffusion_with_regressor.training_step(batch, 1)  # odd idx
    assert called.get("regressor"), "regressor_loss should be called on odd batch when forces present"


def test_training_step_odd_batch_no_forces_falls_back_to_diffusion(diffusion_with_regressor, batch):
    """Odd batch_idx but no forces → fall back to diffusion_loss."""
    diffusion_with_regressor.score_model.training_mode()
    # Ensure batch has no 'forces' attribute
    if hasattr(batch, "forces"):
        del batch.forces

    called = {}
    _orig = diffusion_with_regressor.diffusion_loss
    def _patched(b, bi):
        called["diffusion"] = True
        return _orig(b, bi)
    diffusion_with_regressor.diffusion_loss = _patched

    diffusion_with_regressor.training_step(batch, 1)  # odd idx, no forces
    assert called.get("diffusion"), "diffusion_loss should be called on odd batch when forces absent"

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
        cell=np.diag([8.0, 8.0, 8.0]),
        property={"property": 1.0},
    )
    assert len(out) == 5
    assert all(isinstance(g, AtomsGraph) for g in out)

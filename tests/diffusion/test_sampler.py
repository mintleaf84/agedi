"""Tests for Diffusion (standalone sampler) and predictor-corrector sampling."""
import numpy as np
import pytest
import torch

from agedi.data import AtomsGraph
from agedi.diffusion import Agedi, Diffusion
from agedi.diffusion.noisers import CellPositions


# ---------------------------------------------------------------------------
# Diffusion standalone (no Lightning)
# ---------------------------------------------------------------------------

def test_diffusion_is_not_lightning(diffusion):
    """Diffusion must not inherit from LightningModule."""
    from lightning import LightningModule

    # Diffusion itself is not a LightningModule
    assert LightningModule not in Diffusion.__mro__
    # But the Agedi fixture (which IS a LightningModule) is still a Diffusion
    assert isinstance(diffusion, Diffusion)


def test_agedi_inherits_diffusion(diffusion):
    """Agedi should inherit from both LightningModule and Diffusion."""
    from lightning import LightningModule

    assert isinstance(diffusion, LightningModule)
    assert isinstance(diffusion, Diffusion)


def test_diffusion_standalone(package, conditionings, noisers):
    """Diffusion can be instantiated and sample without Lightning."""
    from agedi.models import ScoreModel

    translator, representation, heads = package
    score_model = ScoreModel(
        translator=translator,
        representation=representation,
        conditionings=conditionings,
        heads=heads,
    )

    sampler = Diffusion(score_model, noisers)
    assert sampler is not None
    assert sampler.device is not None

    out = sampler.sample(
        1,
        steps=3,
        atomic_numbers=[6, 8, 8],
        cell=np.diag([10.0, 10.0, 10.0]),
        property={"property": 1.0},
    )
    assert len(out) == 1
    assert isinstance(out[0], AtomsGraph)


# ---------------------------------------------------------------------------
# Predictor-corrector sampling
# ---------------------------------------------------------------------------

def test_sample_with_corrector_steps_returns_correct_count(diffusion):
    """sample() with corrector_steps>0 returns the right number of structures."""
    out = diffusion.sample(
        2,
        steps=4,
        atomic_numbers=[6, 8],
        cell=np.diag([10.0, 10.0, 10.0]),
        property={"property": 1.0},
        corrector_steps=1,
        corrector_step_size=1e-3,
    )
    assert len(out) == 2
    assert all(isinstance(g, AtomsGraph) for g in out)


def test_sample_corrector_zero_matches_no_corrector(diffusion):
    """corrector_steps=0 (default) should give the same result as not passing it."""
    torch.manual_seed(42)
    out_default = diffusion.sample(
        1,
        steps=3,
        atomic_numbers=[6, 8],
        cell=np.diag([10.0, 10.0, 10.0]),
        property={"property": 1.0},
    )
    torch.manual_seed(42)
    out_explicit = diffusion.sample(
        1,
        steps=3,
        atomic_numbers=[6, 8],
        cell=np.diag([10.0, 10.0, 10.0]),
        property={"property": 1.0},
        corrector_steps=0,
    )
    assert torch.allclose(out_default[0].pos, out_explicit[0].pos)


def test_sample_corrector_changes_positions(diffusion):
    """Applying corrector steps should change the final positions vs no corrector."""
    torch.manual_seed(0)
    out_no_corr = diffusion.sample(
        1,
        steps=5,
        atomic_numbers=[6, 8],
        cell=np.diag([10.0, 10.0, 10.0]),
        property={"property": 1.0},
        corrector_steps=0,
    )
    torch.manual_seed(0)
    out_with_corr = diffusion.sample(
        1,
        steps=5,
        atomic_numbers=[6, 8],
        cell=np.diag([10.0, 10.0, 10.0]),
        property={"property": 1.0},
        corrector_steps=2,
        corrector_step_size=1e-3,
    )
    # Corrector steps should produce different positions
    assert not torch.allclose(out_no_corr[0].pos, out_with_corr[0].pos)


def test_sample_corrector_split_batches(diffusion):
    """Corrector steps should work correctly when N > batch_size."""
    out = diffusion.sample(
        3,
        batch_size=2,
        steps=3,
        atomic_numbers=[6, 8],
        cell=np.diag([10.0, 10.0, 10.0]),
        property={"property": 1.0},
        corrector_steps=1,
        corrector_step_size=1e-3,
    )
    assert len(out) == 3
    assert all(isinstance(g, AtomsGraph) for g in out)


def test_sample_corrector_save_path(diffusion):
    """Corrector steps should work together with save_path=True."""
    steps = 4
    out = diffusion.sample(
        1,
        steps=steps,
        atomic_numbers=[6, 8],
        cell=np.diag([10.0, 10.0, 10.0]),
        property={"property": 1.0},
        corrector_steps=1,
        save_path=True,
    )
    # save_path=True returns a list of trajectories (one per graph)
    assert len(out) == 1
    # Each trajectory has one entry per step + 1 final snapshot
    assert len(out[0]) == steps + 1


# ---------------------------------------------------------------------------
# langevin_step on Noiser base class
# ---------------------------------------------------------------------------

def test_langevin_step_with_float(diffusion, batch):
    """langevin_step should accept a plain float step_size."""
    diffusion.score_model.sample_mode()
    diffusion.sample_time(batch)
    diffusion.forward_step(batch)
    # score must be populated before calling langevin_step
    batch = diffusion.score_model(batch)

    noiser = diffusion.noisers[0]
    pos_before = batch.pos.clone()
    batch = noiser.langevin_step(batch, step_size=1e-3)
    assert batch.pos.shape == pos_before.shape


def test_langevin_step_with_tensor(diffusion, batch):
    """langevin_step should accept a pre-created Tensor step_size."""
    diffusion.score_model.sample_mode()
    diffusion.sample_time(batch)
    diffusion.forward_step(batch)
    batch = diffusion.score_model(batch)

    noiser = diffusion.noisers[0]
    dt = torch.tensor(1e-3, dtype=batch.time.dtype, device=batch.time.device)
    pos_before = batch.pos.clone()
    batch = noiser.langevin_step(batch, step_size=dt)
    assert batch.pos.shape == pos_before.shape


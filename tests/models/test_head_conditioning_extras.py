"""Tests for Head base class (score_clip), Conditioning unsupported type,
and PositionsNoiser.periodic_distance."""
import torch
import pytest

from agedi.data import AtomsGraph, Representation
from agedi.models.head import Head
from agedi.models.conditionings.base import Conditioning
from agedi.diffusion.noisers.pos import PositionsNoiser


# ── Head: score_clip ─────────────────────────────────────────────────────────

class ClampingHead(Head):
    _key = "pos"

    def _score(self, translated_batch):
        return translated_batch


def test_head_clips_score():
    head = ClampingHead(score_clip=1.0)
    raw = torch.tensor([-3.0, 0.0, 3.0])
    out = head.forward(raw)
    assert torch.allclose(out, torch.tensor([-1.0, 0.0, 1.0]))


def test_head_no_clip_when_none():
    head = ClampingHead(score_clip=None)
    raw = torch.tensor([-5.0, 0.0, 5.0])
    out = head.forward(raw)
    assert torch.allclose(out, raw)


# ── Conditioning: unsupported concatenation_type ─────────────────────────────

class ConcreteConditioning(Conditioning):
    def get_conditioning(self, x):
        return x

    def get_empty_conditioning(self, n):
        return torch.zeros(n, self.output_dim)


def test_conditioning_unsupported_concat_type_raises():
    cond = ConcreteConditioning(
        property="time",
        input_dim=1,
        output_dim=2,
        concatenation_type="unsupported",
    )
    cond.sample_mode()
    batch = _make_small_batch()
    batch.time = torch.rand((batch.num_nodes, 1))
    batch.representation = Representation(scalar=torch.rand((batch.num_nodes, 4, 1)))
    with pytest.raises(ValueError, match="not supported"):
        cond(batch)


def _make_small_batch():
    from torch_geometric.data import Batch
    from ase.build import molecule
    atoms = molecule("H2O")
    atoms.set_cell([8.0, 8.0, 8.0])
    atoms.set_pbc(True)
    atoms.center()
    g = AtomsGraph.from_atoms(atoms)
    return Batch.from_data_list([g])


# ── PositionsNoiser.periodic_distance ────────────────────────────────────────

def test_periodic_distance_shape_and_type():
    noiser = PositionsNoiser()
    N = 5
    X = torch.rand((N, 3))
    noise = torch.rand((N, 3)) * 0.1
    cell = torch.eye(3).unsqueeze(0).expand(N, -1, -1).reshape(N * 3, 3)
    batch_idx = torch.zeros(N, dtype=torch.long)

    out = noiser.periodic_distance(X, noise, cell, batch_idx)
    assert out.shape == (N, 3)


def test_periodic_distance_small_noise_close_to_input_noise():
    noiser = PositionsNoiser()
    X = torch.zeros((3, 3))
    tiny_noise = torch.tensor([[0.01, 0.0, 0.0], [0.0, 0.01, 0.0], [0.0, 0.0, 0.01]])
    cell = (torch.eye(3) * 10.0).unsqueeze(0).expand(3, -1, -1).reshape(9, 3)
    batch_idx = torch.zeros(3, dtype=torch.long)

    out = noiser.periodic_distance(X, tiny_noise, cell, batch_idx)
    assert torch.allclose(out, tiny_noise, atol=1e-5)

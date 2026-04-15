"""Tests for Constant and Uniform (including UniformCellConfined) distributions."""
import torch
import pytest
from ase.build import molecule

from agedi.data import AtomsGraph
from agedi.diffusion.distributions.constant import Constant
from agedi.diffusion.distributions.uniform import Uniform, UniformCell, UniformCellConfined


def _build_single_graph():
    atoms = molecule("H2O")
    atoms.set_cell([8.0, 8.0, 8.0])
    atoms.set_pbc(True)
    atoms.center()
    return AtomsGraph.from_atoms(atoms)


# ── Constant ─────────────────────────────────────────────────────────────────

class TestConstant:
    def test_sample_with_explicit_shape(self):
        d = Constant(value=5, dtype=torch.int64)
        out = d._sample(shape=(4,))
        assert out.shape == (4,)
        assert (out == 5).all()

    def test_sample_with_batch_setup(self, batch):
        d = Constant(value=0, key="x")
        callable_fn = d.get_callable(batch)
        out = callable_fn()
        assert out.shape[0] == batch.num_nodes
        assert (out == 0).all()

    def test_default_value_is_zero(self):
        d = Constant()
        out = d._sample(shape=(10,))
        assert (out == 0).all()


# ── Uniform ──────────────────────────────────────────────────────────────────

class TestUniform:
    def test_sample_explicit_shape(self):
        d = Uniform(low=2.0, high=5.0)
        out = d._sample(shape=(8, 3))
        assert out.shape == (8, 3)
        assert (out >= 2.0).all()
        assert (out <= 5.0).all()

    def test_sample_uses_stored_shape(self, batch):
        d = Uniform(key="x")
        c = d.get_callable(batch)
        out = c()
        assert out.shape == batch.x.shape


# ── UniformCell non-batch path ────────────────────────────────────────────────

class TestUniformCellNonBatch:
    def test_sample_returns_correct_shape(self):
        graph = _build_single_graph()
        d = UniformCell()
        # non-batch: graph.batch is None
        graph_batch = graph.clone()
        object.__setattr__(graph_batch, "batch", None)
        d._setup(graph_batch)
        out = d._sample()
        n_atoms = graph.n_atoms.item()
        assert out.shape == (n_atoms, 3)


# ── UniformCellConfined ───────────────────────────────────────────────────────

class TestUniformCellConfined:
    def test_setup_adjusts_cell_and_corner(self):
        graph = _build_single_graph()
        graph.confinement = torch.tensor([[1.0, 7.0]])
        # Non-batch path
        object.__setattr__(graph, "batch", None)
        d = UniformCellConfined()
        d._setup(graph)
        assert d.corner[0, 2].item() == pytest.approx(1.0)
        assert d.cell[2, 2].item() == pytest.approx(6.0)

    def test_setup_raises_for_batch(self, batch):
        batch.confinement = torch.tensor([[1.0, 7.0]])
        d = UniformCellConfined()
        with pytest.raises(NotImplementedError):
            d._setup(batch)

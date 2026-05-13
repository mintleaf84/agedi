"""Benchmark / correctness test: nvidia chemiops neighbor list vs matscipy.

For each system type (molecule, surface, bulk) the test builds the neighbor
list with both backends and asserts that the resulting edge sets are
identical (same pairs + same periodic-image shift vectors, regardless of
ordering).

The nvidia tests are automatically skipped when the ``nvalchemiops`` package
is not installed.
"""

import numpy as np
import pytest
import torch
from ase.build import bulk, fcc111, molecule
from torch_geometric.data import Batch

from agedi.data import AtomsGraph
from agedi.data.atoms_graph import (
    NVIDIA_NEIGHBOR_IMPORT_ERROR,
    nvidia_neighbor_list,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

pytestmark = pytest.mark.skipif(
    nvidia_neighbor_list is None,
    reason=f"nvalchemiops not available: {NVIDIA_NEIGHBOR_IMPORT_ERROR}",
)


def _canonical_edges(
    edge_index: torch.Tensor,
    shift_vectors: torch.Tensor,
    atol: float = 1e-4,
) -> np.ndarray:
    """Return a sorted (N, 5) array of [src, dst, sx, sy, sz] rows.

    Shift vector components are rounded to *atol* before sorting so that
    floating-point noise does not produce duplicate or missing rows.
    """
    src = edge_index[0].cpu().numpy()
    dst = edge_index[1].cpu().numpy()
    sv = shift_vectors.cpu().numpy()
    sv_rounded = np.round(sv / atol) * atol
    rows = np.column_stack([src, dst, sv_rounded])
    # Sort lexicographically
    order = np.lexsort(rows[:, ::-1].T)
    return rows[order]


def _assert_edge_sets_equal(
    ei_a: torch.Tensor,
    sv_a: torch.Tensor,
    ei_b: torch.Tensor,
    sv_b: torch.Tensor,
    atol: float = 1e-3,
) -> None:
    """Assert that two (edge_index, shift_vectors) pairs encode the same graph."""
    assert ei_a.shape[1] == ei_b.shape[1], (
        f"Edge counts differ: nvidia={ei_a.shape[1]}, matscipy={ei_b.shape[1]}"
    )
    canon_a = _canonical_edges(ei_a, sv_a, atol=atol)
    canon_b = _canonical_edges(ei_b, sv_b, atol=atol)
    np.testing.assert_allclose(
        canon_a,
        canon_b,
        atol=atol,
        err_msg="Edge sets differ between nvidia and matscipy neighbor lists",
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

CUTOFF = 5.0


@pytest.fixture(
    params=["molecule", "surface", "bulk"],
    ids=["molecule", "surface", "bulk"],
)
def atoms(request):
    """Three representative ASE Atoms objects."""
    if request.param == "molecule":
        a = molecule("H2O")
        a.set_cell([10, 10, 10])
        a.set_pbc(True)
        a.center()
    elif request.param == "surface":
        a = fcc111("Au", (3, 3, 3), vacuum=10)
        a.set_pbc(True)
    elif request.param == "bulk":
        a = bulk("Cu", "fcc", a=3.6, cubic=True)
        a.set_pbc(True)
    return a


@pytest.fixture(
    params=["molecule_batch", "surface_batch", "bulk_batch"],
    ids=["molecule_batch", "surface_batch", "bulk_batch"],
)
def batch(request):
    """Batches of two or more structures of each type."""
    graphs = []
    if request.param == "molecule_batch":
        for name in ["H2O", "NH3", "CH4"]:
            a = molecule(name)
            a.set_cell([10, 10, 10])
            a.set_pbc(True)
            a.center()
            graphs.append(AtomsGraph.from_atoms(a, cutoff=CUTOFF))
    elif request.param == "surface_batch":
        a = fcc111("Au", (3, 3, 3), vacuum=10)
        a.set_pbc(True)
        graphs.append(AtomsGraph.from_atoms(a, cutoff=CUTOFF))
        a2 = fcc111("Cu", (2, 2, 4), vacuum=8)
        a2.set_pbc(True)
        graphs.append(AtomsGraph.from_atoms(a2, cutoff=CUTOFF))
    elif request.param == "bulk_batch":
        a = bulk("Cu", "fcc", a=3.6, cubic=True)
        a.set_pbc(True)
        graphs.append(AtomsGraph.from_atoms(a, cutoff=CUTOFF))
        a2 = bulk("Al", "bcc", a=3.2)
        a2.set_pbc(True)
        graphs.append(AtomsGraph.from_atoms(a2, cutoff=CUTOFF))
    return Batch.from_data_list(graphs)


# ---------------------------------------------------------------------------
# Single-system tests
# ---------------------------------------------------------------------------


def test_single_system_edge_count(atoms):
    """Nvidia and matscipy must find the same number of edges for a single system."""
    pos = torch.tensor(atoms.positions, dtype=torch.float32)
    cell = torch.tensor(np.array(atoms.cell), dtype=torch.float32)
    pbc = torch.tensor(atoms.pbc, dtype=torch.bool)

    ei_nvidia, sv_nvidia = AtomsGraph.make_graph(pos, cell, CUTOFF, pbc)
    ei_matscipy, sv_matscipy = AtomsGraph._make_graph_matscipy(
        pos, cell, CUTOFF, pbc
    )

    assert ei_nvidia.shape[1] == ei_matscipy.shape[1], (
        f"Edge count mismatch for {atoms.get_chemical_formula()}: "
        f"nvidia={ei_nvidia.shape[1]}, matscipy={ei_matscipy.shape[1]}"
    )


def test_single_system_edge_set(atoms):
    """Nvidia and matscipy must produce exactly the same edge + shift set for a single system."""
    pos = torch.tensor(atoms.positions, dtype=torch.float32)
    cell = torch.tensor(np.array(atoms.cell), dtype=torch.float32)
    pbc = torch.tensor(atoms.pbc, dtype=torch.bool)

    ei_nvidia, sv_nvidia = AtomsGraph.make_graph(pos, cell, CUTOFF, pbc)
    ei_matscipy, sv_matscipy = AtomsGraph._make_graph_matscipy(
        pos, cell, CUTOFF, pbc
    )

    _assert_edge_sets_equal(ei_nvidia, sv_nvidia, ei_matscipy, sv_matscipy)


def test_single_system_shift_vector_norms(atoms):
    """All shift vectors must have finite, non-negative norms for a single system."""
    pos = torch.tensor(atoms.positions, dtype=torch.float32)
    cell = torch.tensor(np.array(atoms.cell), dtype=torch.float32)
    pbc = torch.tensor(atoms.pbc, dtype=torch.bool)

    ei_nvidia, sv_nvidia = AtomsGraph.make_graph(pos, cell, CUTOFF, pbc)

    assert torch.isfinite(sv_nvidia).all(), "Non-finite shift vectors from nvidia"

    # Verify that adding the shift to the source-atom position gives the
    # neighbour position within cutoff.
    src = ei_nvidia[0]
    dst = ei_nvidia[1]
    delta = pos[dst] - pos[src] + sv_nvidia
    dists = delta.norm(dim=-1)
    assert (dists <= CUTOFF + 1e-3).all(), (
        "Some nvidia edges exceed the cutoff after applying shift vectors"
    )


# ---------------------------------------------------------------------------
# Batched tests
# ---------------------------------------------------------------------------


def test_batched_edge_count(batch):
    """Nvidia and matscipy must find the same total number of edges for a batch."""
    batch_idx = batch.batch.to(torch.int32)
    cell = batch.cell.view(-1, 3, 3).contiguous()
    pbc = batch.pbc.view(-1, 3).contiguous()

    ei_nvidia, sv_nvidia = AtomsGraph.make_graph(
        batch.pos, cell, CUTOFF, pbc, batch_idx=batch_idx
    )
    ei_matscipy, sv_matscipy = AtomsGraph._make_graph_matscipy(
        batch.pos, cell, CUTOFF, pbc, batch_idx=batch_idx
    )

    assert ei_nvidia.shape[1] == ei_matscipy.shape[1], (
        f"Batched edge count mismatch: "
        f"nvidia={ei_nvidia.shape[1]}, matscipy={ei_matscipy.shape[1]}"
    )


def test_batched_edge_set(batch):
    """Nvidia and matscipy must produce the same edge + shift set for a batch."""
    batch_idx = batch.batch.to(torch.int32)
    cell = batch.cell.view(-1, 3, 3).contiguous()
    pbc = batch.pbc.view(-1, 3).contiguous()

    ei_nvidia, sv_nvidia = AtomsGraph.make_graph(
        batch.pos, cell, CUTOFF, pbc, batch_idx=batch_idx
    )
    ei_matscipy, sv_matscipy = AtomsGraph._make_graph_matscipy(
        batch.pos, cell, CUTOFF, pbc, batch_idx=batch_idx
    )

    _assert_edge_sets_equal(ei_nvidia, sv_nvidia, ei_matscipy, sv_matscipy)


def test_batched_shift_vector_norms(batch):
    """All shift vectors must be finite and within cutoff for a batch."""
    batch_idx = batch.batch.to(torch.int32)
    cell = batch.cell.view(-1, 3, 3).contiguous()
    pbc = batch.pbc.view(-1, 3).contiguous()

    ei_nvidia, sv_nvidia = AtomsGraph.make_graph(
        batch.pos, cell, CUTOFF, pbc, batch_idx=batch_idx
    )

    assert torch.isfinite(sv_nvidia).all(), "Non-finite shift vectors from nvidia (batch)"

    src = ei_nvidia[0]
    dst = ei_nvidia[1]
    delta = batch.pos[dst] - batch.pos[src] + sv_nvidia
    dists = delta.norm(dim=-1)
    assert (dists <= CUTOFF + 1e-3).all(), (
        "Some batched nvidia edges exceed the cutoff after applying shift vectors"
    )

import numpy as np
import pytest
import torch
from ase import Atoms
from torch_geometric.data import Batch

import agedi.data.atoms_graph as atoms_graph_module
from agedi.data import AtomsGraph, Representation


def test_from_atoms(atoms: "Atoms") -> None:
    graph = AtomsGraph.from_atoms(atoms)
    assert isinstance(graph, AtomsGraph)

def test_to_atoms(atoms: "Atoms") -> None:
    graph = AtomsGraph.from_atoms(atoms)
    a = graph.to_atoms()

    # With canonical_cell=False (default) the cell is stored as-is so
    # fractional coordinates are trivially preserved.
    orig_frac = atoms.get_scaled_positions(wrap=False)
    new_frac = a.get_scaled_positions(wrap=False)
    assert np.allclose(
        (orig_frac + 0.5) % 1, (new_frac + 0.5) % 1, atol=1e-5
    )
    assert np.allclose(a.pbc, atoms.pbc)
    assert np.equal(a.numbers, atoms.numbers).all()

def test_make_graph(atoms: "Atoms") -> None:
    edge_index, shift_vectors = AtomsGraph.make_graph(
        torch.tensor(atoms.positions),
        torch.tensor(np.array(atoms.cell)),
        6.0,
        torch.tensor(atoms.pbc),
    )
    print(edge_index.shape, shift_vectors.shape)
    assert edge_index.shape[0] == 2
    assert edge_index.shape[1] == shift_vectors.shape[0]
    assert shift_vectors.shape[1] == 3

def test_clear_graph(graph: AtomsGraph) -> None:
    graph.clear_graph()

    assert "edge_index" not in graph.keys()
    assert "shift_vectors" not in graph.keys()
    
def test_update_graph(atoms: "Atoms") -> None:
    graph = AtomsGraph.from_atoms(atoms)
    graph.clear_graph()
    graph.update_graph()
    
    assert len(graph) == len(atoms)
    assert graph.edge_index.shape[0] == 2
    assert graph.edge_index.shape[1] == graph.shift_vectors.shape[0]
    assert graph.shift_vectors.shape[1] == 3


def test_pos_setter_preserves_graph_with_skin(atoms: "Atoms") -> None:
    graph = AtomsGraph.from_atoms(atoms, skin=0.2)
    edge_index = graph.edge_index.clone()
    shift_vectors = graph.shift_vectors.clone()

    graph.pos = graph.pos + 1e-3

    assert torch.equal(graph.edge_index, edge_index)
    assert torch.equal(graph.shift_vectors, shift_vectors)


def test_update_graph_method_skips_rebuild_with_skin(
    monkeypatch, atoms: "Atoms"
) -> None:
    graph = AtomsGraph.from_atoms(atoms, skin=0.2)
    edge_index = graph.edge_index.clone()
    shift_vectors = graph.shift_vectors.clone()
    graph.pos = graph.pos + 1e-3

    monkeypatch.setattr(
        atoms_graph_module,
        "neighbor_list_needs_rebuild",
        lambda **_: torch.tensor([False], dtype=torch.bool),
    )

    def fail_make_graph(*args, **kwargs):
        raise AssertionError("graph should not rebuild when all atoms stay within skin")

    monkeypatch.setattr(AtomsGraph, "make_graph", staticmethod(fail_make_graph))

    graph.update_graph()

    assert torch.equal(graph.edge_index, edge_index)
    assert torch.equal(graph.shift_vectors, shift_vectors)


def test_batch_update_graph_selectively_rebuilds_with_skin(monkeypatch) -> None:
    atoms1 = Atoms(
        "H2",
        positions=[[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]],
        cell=[10.0, 10.0, 10.0],
        pbc=[False, False, False],
    )
    atoms2 = Atoms(
        "H2",
        positions=[[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]],
        cell=[10.0, 10.0, 10.0],
        pbc=[False, False, False],
    )
    batch = Batch.from_data_list(
        [
            AtomsGraph.from_atoms(atoms1, cutoff=1.0, skin=0.2),
            AtomsGraph.from_atoms(atoms2, cutoff=1.0, skin=0.2),
        ]
    )

    total_atoms = batch.pos.shape[0]
    initial_neighbor_matrix = torch.tensor(
        [[1, total_atoms], [0, total_atoms], [3, total_atoms], [2, total_atoms]],
        dtype=torch.int32,
    )
    initial_num_neighbors = torch.tensor([1, 1, 1, 1], dtype=torch.int32)
    initial_neighbor_shifts = torch.zeros((4, 2, 3), dtype=torch.int32)
    batch.neighbor_matrix = initial_neighbor_matrix.clone()
    batch.num_neighbors = initial_num_neighbors.clone()
    batch.neighbor_matrix_shifts = initial_neighbor_shifts.clone()
    batch.reference_positions = batch.pos.clone()
    batch.reference_cell = batch.cell.clone()
    batch.reference_pbc = batch.pbc.clone()
    batch.edge_index, batch.shift_vectors = AtomsGraph._neighbor_matrix_to_graph(
        neighbor_matrix=batch.neighbor_matrix,
        num_neighbors=batch.num_neighbors,
        neighbor_matrix_shifts=batch.neighbor_matrix_shifts,
        cell=batch.cell.view(-1, 3, 3),
        dtype=batch.pos.dtype,
        fill_value=total_atoms,
        batch_idx=batch.batch.to(torch.int32),
    )

    initial_reference_positions = batch.reference_positions.clone()
    batch.pos[2:] = torch.tensor([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]], dtype=batch.pos.dtype)
    rebuild_flags = torch.tensor([False, True], dtype=torch.bool)

    def fake_batch_neighbor_list_needs_rebuild(
        reference_positions,
        current_positions,
        batch_idx,
        skin_distance_threshold,
        update_reference_positions,
        cell,
        cell_inv,
        pbc,
    ):
        assert skin_distance_threshold == pytest.approx(0.2)
        assert update_reference_positions is True
        reference_positions[batch_idx == 1] = current_positions[batch_idx == 1]
        return rebuild_flags

    def fake_batch_naive_neighbor_list(
        positions,
        cutoff,
        batch_idx,
        batch_ptr,
        cell,
        pbc,
        neighbor_matrix,
        neighbor_matrix_shifts,
        num_neighbors,
        shift_range_per_dimension,
        num_shifts_per_system,
        max_shifts_per_system,
        max_atoms_per_system,
        rebuild_flags,
    ):
        assert torch.equal(rebuild_flags, torch.tensor([False, True], dtype=torch.bool))
        neighbor_matrix[2:] = total_atoms
        neighbor_matrix_shifts[2:] = 0
        num_neighbors[2:] = 0
        return neighbor_matrix, num_neighbors, neighbor_matrix_shifts

    monkeypatch.setattr(
        atoms_graph_module,
        "batch_neighbor_list_needs_rebuild",
        fake_batch_neighbor_list_needs_rebuild,
    )
    monkeypatch.setattr(
        atoms_graph_module,
        "batch_naive_neighbor_list",
        fake_batch_naive_neighbor_list,
    )

    batch.update_graph()

    assert torch.equal(batch.neighbor_matrix[:2], initial_neighbor_matrix[:2])
    assert torch.equal(batch.num_neighbors[:2], initial_num_neighbors[:2])
    assert torch.equal(batch.reference_positions[:2], initial_reference_positions[:2])
    assert torch.equal(batch.reference_positions[2:], batch.pos[2:])
    assert batch.num_neighbors[2:].sum().item() == 0

def test_to_data_list_after_update_graph_with_skin(atoms: "Atoms") -> None:
    """Regression test: to_data_list() must work after update_graph() with skin.

    Previously, update_graph() stored reference_cell / reference_pbc in the
    locally-reshaped (N, 3, 3) / (N, 3) layout instead of the batched layout
    (N*3, 3) / (N*3,) that PyG's _slice_dict expects, causing a RuntimeError.
    """
    g1 = AtomsGraph.from_atoms(atoms, skin=0.2)
    g2 = AtomsGraph.from_atoms(atoms, skin=0.2)
    batch = Batch.from_data_list([g1, g2])
    batch.update_graph()
    data_list = batch.to_data_list()
    assert len(data_list) == 2
    for d in data_list:
        assert d.reference_cell.shape == (3, 3)
        assert d.reference_pbc.shape == (3,)


def test_len(atoms: "Atoms") -> None:
    graph = AtomsGraph.from_atoms(atoms)
    assert len(graph) == len(atoms)

@pytest.mark.parametrize("type", ["node", "graph"])
def test_add_batch_attr(type: str, batch: "Batch") -> None:
    if type == "node":
        attr = torch.randn(len(batch), 3)
        t = "x"
    elif type == "graph":
        attr = torch.randn((batch.num_graphs,))
        t = "n_atoms"

    batch.add_batch_attr("test", attr, type=type)

    assert (batch["test"] == attr).all()
    assert (batch._slice_dict["test"] == batch._slice_dict[t]).all()

def test_add_batch_attr_fail(batch: "Batch") -> None:
    attr = torch.randn(1)
    with pytest.raises(ValueError):
        batch.add_batch_attr("test", attr, type="other")

def test_positions_mask(graph: AtomsGraph) -> None:
    mask = graph.positions_mask
    assert mask.shape == (len(graph),3)

def test_pos_setter_clear(graph: AtomsGraph) -> None:
    graph.pos = torch.randn_like(graph.pos)
    
    assert "edge_index" not in graph.keys()
    assert "shift_vectors" not in graph.keys()
    
def test_pos_setter_shape(graph: AtomsGraph) -> None:
    new_pos = torch.randn_like(graph.pos).unsqueeze(0)
    with pytest.raises(IndexError):
        graph.pos = new_pos

def test_frac(graph: AtomsGraph) -> None:
    f = graph.frac
    f = graph.frac # test caching
    a = graph.to_atoms()
    close1 = np.isclose(f.detach().numpy(), a.get_scaled_positions())
    close2 = np.isclose((f.detach().numpy()+0.5)%1.0, (a.get_scaled_positions()+0.5)%1.0)
    allclose = np.logical_or(close1, close2)
    assert allclose.all()

def test_frac_setter(atoms: "Atoms") -> None:
    atoms.positions += 1e-4
    atoms.wrap()
    graph = AtomsGraph.from_atoms(atoms)
    f = torch.tensor(atoms.get_scaled_positions(wrap=True), dtype=torch.float32)
    positions = torch.tensor(atoms.positions, dtype=torch.float32)
    graph.frac = f

    assert torch.allclose(graph.pos, positions)

def test_frac_setter_clear(graph: AtomsGraph) -> None:
    graph.frac = torch.rand_like(graph.frac)

    assert "edge_index" not in graph.keys()
    assert "shift_vectors" not in graph.keys()

def test_pos_frac_batch(batch: "Batch") -> None:
    batch.frac = torch.rand_like(batch.frac)
    
    assert batch.pos.shape[0] == batch.batch.shape[0]

@pytest.mark.parametrize("mask", [True, False])
def test_x_setter_mask(graph: AtomsGraph, mask: bool) -> None:
    if mask:
        graph.mask = torch.rand(graph.mask.shape) > 0.5

    x_old = graph.x.clone()
    x = torch.randint(1, 92, graph.x.shape)
    graph.x = x.clone()

    if mask:
        assert torch.equal(graph.x[graph.mask], x_old[graph.mask])
        assert torch.equal(graph.x[~graph.mask], x[~graph.mask])
    else:
        assert torch.equal(graph.x, x)
                
def test_time_none(graph: AtomsGraph) -> None:
    assert graph.time is None

def test_time_setter(graph: AtomsGraph) -> None:
    t = torch.rand((graph.num_nodes,1), dtype=torch.float32)
    graph.time = t.clone()
    assert torch.equal(graph.time, t)

def test_time_mask(graph: AtomsGraph) -> None:
    t = torch.rand((graph.num_nodes,1), dtype=torch.float32)
    graph.mask = torch.rand(graph.mask.shape) > 0.5
    graph.time = t.clone()
    assert (graph.time[graph.mask] == 0.0).all()
    
def test_wrap(atoms: "Atoms") -> None:
    atoms.positions += 3
    graph = AtomsGraph.from_atoms(atoms)
    atoms.wrap()
    pos = torch.tensor(atoms.positions, dtype=torch.float32)
    
    graph.wrap_positions()
    assert np.allclose(graph.pos, pos)
    
def test_apply_mask(graph: AtomsGraph) -> None:
    mask = torch.rand(graph.mask.shape) > 0.5
    graph.mask = mask.clone()

    x = torch.randn((graph.num_nodes, ))
    masked_x = graph.apply_mask(x, val=-1)
    assert (masked_x[mask] == -1).all()

def test_apply_pos_mask(graph: AtomsGraph) -> None:
    mask = torch.rand(graph.mask.shape) > 0.5
    graph.mask = mask.clone()

    x = torch.randn((graph.num_nodes, 3))
    masked_x = graph.apply_mask(x, val=-1)
    assert (masked_x[mask, :] == -1).all()

def test_apply_mask_error(graph: AtomsGraph) -> None:
    mask = torch.rand(graph.mask.shape) > 0.5
    graph.mask = mask.clone()

    x = torch.randn((graph.num_nodes, 1))
    with pytest.raises(ValueError):
        graph.apply_mask(x, val=-1)

def test_empty() -> None:
    graph = AtomsGraph.empty()
    assert isinstance(graph, AtomsGraph)


def test_cell_is_canonical(atoms) -> None:
    """Cell stored in AtomsGraph must be canonical (cellpar round-trip) when canonical_cell=True."""
    graph = AtomsGraph.from_atoms(atoms, canonical_cell=True)
    cell = graph.cell
    cell_params = AtomsGraph.cell_to_vectors(cell)
    canonical = AtomsGraph.vector_to_cell(cell_params).view(3, 3)
    assert torch.allclose(cell, canonical, atol=1e-5)


def test_cell_setter_preserves_frac(graph: AtomsGraph) -> None:
    """Setting a new cell must not change fractional coordinates."""
    frac_before = graph.frac.clone()
    # Rotate the cell slightly by applying a small perturbation to the cellpar
    cell_params = AtomsGraph.cell_to_vectors(graph.cell).squeeze(0)
    cell_params[3] += 0.05  # shift alpha a little
    new_cell = AtomsGraph.vector_to_cell(cell_params).view(3, 3)
    graph.cell = new_cell
    frac_after = graph.frac
    assert torch.allclose(frac_before, frac_after, atol=1e-5)


def test_cell_setter_clears_graph(graph: AtomsGraph) -> None:
    """Setting the cell must invalidate the edge index."""
    cell_params = AtomsGraph.cell_to_vectors(graph.cell).squeeze(0)
    cell_params[0] += 0.1
    graph.cell = AtomsGraph.vector_to_cell(cell_params).view(3, 3)
    assert "edge_index" not in graph.keys()
    assert "shift_vectors" not in graph.keys()



def test_representation_to_tensor() -> None:
    N, d = 12, 64
    scalar = torch.randn((N, d, 1))
    vector = torch.randn((N, d, 3))
    tensor = torch.randn((N, d, 5))

    rep = Representation(scalar=scalar, vector=vector, tensor=tensor)

    t, _, _ = rep.to_tensor(n_graphs=1)
    assert t.shape == (N, d*9)

def test_representation_from_tensor() -> None:
    N, d = 12, 64
    scalar = torch.randn((N, d, 1))
    vector = torch.randn((N, d, 3))

    rep = Representation(scalar=scalar.clone(), vector=vector.clone())

    tu = rep.to_tensor(n_graphs=1)
    
    rep2 = Representation.from_tensor(*tu)

    assert torch.allclose(scalar, rep2.scalar)
    assert torch.allclose(vector, rep2.vector)

def test_representation_setters() -> None:
    N, d = 12, 64
    scalar = torch.randn((N, d, 1))
    vector = torch.randn((N, d, 3))

    rep = Representation(scalar=scalar.clone(), vector=vector.clone())

    scalar2 = torch.randn((N, d, 1))
    vector2 = torch.randn((N, d, 3))

    rep.scalar = scalar2.clone()
    rep.vector = vector2.clone()

    assert torch.allclose(rep.scalar, scalar2)
    assert torch.allclose(rep.vector, vector2)

def test_get_representation(graph: AtomsGraph) -> None:
    N, d = graph.num_nodes, 64
    scalar = torch.randn((N, d, 1))
    vector = torch.randn((N, d, 3))

    rep = Representation(scalar=scalar.clone(), vector=vector.clone())

    graph.representation = rep

    rep2 = graph.representation

    assert torch.allclose(scalar, rep2.scalar)
    assert torch.allclose(vector, rep2.vector)

    

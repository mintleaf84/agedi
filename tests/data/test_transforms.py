import torch
from ase.build import molecule

from agedi.data import AtomsGraph
from agedi.data.transforms import Repeat


def _build_graph():
    atoms = molecule("H2O")
    atoms.set_cell([8.0, 8.0, 8.0])
    atoms.set_pbc(True)
    atoms.center()
    return AtomsGraph.from_atoms(atoms)


def test_repeat_without_properties():
    graph = _build_graph()
    transform = Repeat(m=(2, 1, 1))

    repeated = transform.forward(graph)

    assert repeated.n_atoms.item() == graph.n_atoms.item() * 2


def test_repeat_with_node_and_graph_properties():
    graph = _build_graph()
    n = graph.num_nodes
    graph.add_batch_attr("node_prop", torch.arange(n), type="node")
    graph.add_batch_attr("graph_prop", torch.tensor([3.0]), type="graph")
    graph.add_batch_attr("keep_prop", torch.tensor([7.0]), type="graph")

    transform = Repeat(
        m=(2, 1, 1),
        property={"node_prop": "node", "graph_prop": "graph", "keep_prop": "none"},
    )
    repeated = transform.forward(graph)

    assert repeated.node_prop.shape[0] == n * 2
    assert torch.equal(repeated.node_prop, graph.node_prop.repeat(2))
    assert torch.equal(repeated.graph_prop, graph.graph_prop * 2)
    assert torch.equal(repeated.keep_prop, graph.keep_prop)


from typing import Tuple

from torch_geometric.transforms import BaseTransform

from agedi.data import AtomsGraph


class Repeat(BaseTransform):
    def __init__(self, m: Tuple[int, int, int] = (1, 1, 1)):
        self.m = m

    def forward(self, data: AtomsGraph) -> AtomsGraph:
        atoms = data.to_atoms()
        atoms = atoms.repeat(self.m)
        repeated_data = AtomsGraph.from_atoms(atoms)
        return repeated_data


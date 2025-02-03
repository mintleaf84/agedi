from typing import Tuple

from torch_geometric import BaseTransform

from agedi.data import AtomsGraph


class Repeat(BaseTransform):
    def __init__(self, m: Tuple[int, int, int] = (1, 1, 1)):
        self.m = m

    def forward(self, data: AtomsGraph) -> AtomsGraph:
        atoms = data.to_atoms()
        atoms.repeat(self.m)
        repeated_data = AtomsGraph.from_atoms(atoms)
        return repeated_data

    def __repr__(self):
        return "{}({})".format(self.__class__.__name__, self.N)

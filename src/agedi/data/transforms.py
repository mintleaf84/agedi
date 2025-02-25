from typing import Tuple, Dict

from torch_geometric.transforms import BaseTransform

from agedi.data import AtomsGraph


class Repeat(BaseTransform):
    def __init__(self, m: Tuple[int, int, int] = (1, 1, 1), property: Dict[str, str]={}):
        self.m = m

    def forward(self, data: AtomsGraph) -> AtomsGraph:
        new_properties = {}
        for key, val in property.items():
            repeats = self.m.prod()
            if val == 'node':
                new_properties[key] = data.getattr(key).repeat(repeats, 1)
            if val == 'graph':
                new_properties[key] = data.getattr(key) * repeats

        atoms = data.to_atoms()
        atoms = atoms.repeat(self.m)
        repeated_data = AtomsGraph.from_atoms(atoms)
        
        for key, val in new_properties.items():
            repeated_data.setattr(key, val)
            
        return repeated_data


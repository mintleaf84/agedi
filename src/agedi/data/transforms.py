from typing import Tuple, Dict, Optional
import torch

from torch_geometric.transforms import BaseTransform

from agedi.data import AtomsGraph


class Repeat(BaseTransform):
    def __init__(self, m: Tuple[int, int, int] = (1, 1, 1), property: Optional[Dict[str, str]]=None):
        self.m = m
        self.property = property

    def forward(self, data: AtomsGraph) -> AtomsGraph:
        if property is not None:
            new_properties = {}
            for key, val in self.property.items():
                repeats = torch.tensor(self.m).prod()
                if val == 'node':
                    prop = data.get_tensor(key)
                    repeat_shape = [1 for _ in range(len(prop.shape))]
                    repeat_shape[0] = repeats
                    new_properties[key] = prop.repeat(*repeat_shape)
                if val == 'graph':
                    new_properties[key] = data.get_tensor(key) * repeats
                if val == 'none':
                    new_properties[key] = data.get_tensor(key)

        atoms = data.to_atoms()
        atoms = atoms.repeat(self.m)
        repeated_data = AtomsGraph.from_atoms(atoms)

        if property is not None:
            for key, val in new_properties.items():
                repeated_data[key] = val

        return repeated_data


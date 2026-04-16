from typing import Tuple, Dict, Optional
import torch

from torch_geometric.transforms import BaseTransform

from agedi.data import AtomsGraph


class Repeat(BaseTransform):
    """Transform that tiles an :class:`~agedi.data.AtomsGraph` using supercell repeats.

    Wraps :meth:`~ase.Atoms.repeat` and optionally propagates per-node,
    per-graph, or invariant properties to the repeated structure.
    """

    def __init__(self, m: Tuple[int, int, int] = (1, 1, 1), property: Optional[Dict[str, str]]=None):
        """Initialize the repeat transform.

        Parameters
        ----------
        m : tuple[int, int, int], optional
            Number of repetitions along each lattice vector.
            Defaults to ``(1, 1, 1)`` (no repetition).
        property : dict[str, str], optional
            Mapping from property name to propagation mode.  Supported modes:

            * ``"node"``  – repeat the property along the node (atom) axis.
            * ``"graph"`` – multiply the scalar graph property by the total
              number of repeated cells.
            * ``"none"``  – copy the property unchanged.
        """
        self.m = m
        self.property = property

    def forward(self, data: AtomsGraph) -> AtomsGraph:
        """Apply the supercell repeat to *data* and return the new graph.

        Parameters
        ----------
        data : AtomsGraph
            The input atomistic graph to be repeated.

        Returns
        -------
        AtomsGraph
            A new :class:`~agedi.data.AtomsGraph` representing the repeated
            (tiled) structure, with properties propagated according to
            ``self.property``.
        """
        if self.property is not None:
            new_properties = {}
            for key, val in self.property.items():
                repeats = torch.tensor(self.m).prod()
                if val == 'node':
                    prop = data[key]
                    repeat_shape = [1 for _ in range(len(prop.shape))]
                    repeat_shape[0] = repeats
                    new_properties[key] = prop.repeat(*repeat_shape)
                if val == 'graph':
                    new_properties[key] = data[key] * repeats
                if val == 'none':
                    new_properties[key] = data[key]

        atoms = data.to_atoms()
        atoms = atoms.repeat(self.m)
        repeated_data = AtomsGraph.from_atoms(atoms)

        if self.property is not None:
            for key, val in new_properties.items():
                repeated_data[key] = val

        return repeated_data

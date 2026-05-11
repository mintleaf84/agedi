import functools
from typing import Callable, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from matscipy.neighbours import neighbour_list as matscipy_neighbour_list
from torch_geometric.data import Batch, Data

try:
    from nvalchemiops.torch.neighbors import neighbor_list as nvidia_neighbor_list
    from nvalchemiops.torch.neighbors.batch_naive import batch_naive_neighbor_list
    from nvalchemiops.torch.neighbors.rebuild_detection import (
        batch_neighbor_list_needs_rebuild,
        neighbor_list_needs_rebuild,
    )
    NVIDIA_NEIGHBOR_IMPORT_ERROR = None
except (ImportError, ModuleNotFoundError, TypeError) as exc:
    nvidia_neighbor_list = None
    batch_naive_neighbor_list = None
    batch_neighbor_list_needs_rebuild = None
    neighbor_list_needs_rebuild = None
    NVIDIA_NEIGHBOR_IMPORT_ERROR = exc


NEIGHBOR_CACHE_KEYS = (
    "edge_index",
    "shift_vectors",
    "neighbor_matrix",
    "neighbor_matrix_shifts",
    "num_neighbors",
    "reference_positions",
    "reference_cell",
    "reference_pbc",
    "shift_range_per_dimension",
    "num_shifts_per_system",
    "max_shifts_per_system",
    "max_atoms_per_system",
)


def batched(
    update_keys: Optional[Sequence[str]] = None, return_batch: bool = False
) -> Callable:
    """Batched decorator

    Decorator for functions that return Data objects, but can with this operator be
    called with batched inputs. The function will be called for each element in the
    batch, and the results will be concatenated into a single Data object.

    If called with a Data-object as input, the function will be called with as if it
    not decorated.


    Parameters
    ----------
    update_keys: Optional[Sequence[str]]
        The keys in the Batch object that should be updated. If None, no keys will be updated.
    return_batch: bool
        If True, the function will return a Batch object instead of None.

    Returns
    -------
    Callable
    """

    def decorator(func: Callable) -> Callable:
        """Wrap *func* so it can be called on both :class:`Data` and :class:`Batch` objects."""
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs) -> Union[Data, Batch]:
            """Dispatch the wrapped function to each element in a batch or call it directly."""
            if isinstance(self, Batch):
                data_list = self.to_data_list()
                for d in data_list:
                    func(d, *args, **kwargs)

                new_batch = Batch.from_data_list(data_list)
                if update_keys is not None:
                    for key in update_keys:
                        setattr(self, key, new_batch[key])
                if return_batch:
                    return new_batch
            elif isinstance(self, Data):
                return func(self, *args, **kwargs)
            else:
                raise TypeError("Object must be of type Data or Batch.")

        return wrapper

    return decorator


class Representation:
    """Representation class

    Class defining a general representation. The representation is a dictionary of tensors, where each tensor
    is a representation of a certain type of information. The tensors are stored in a dictionary, where the keys
    are degree of the representation l (with dim = 2l+1), and the values are the tensors themselves.

    The representation can be initialized with either a scalar or a vector representation, or both. The scalar
    representation is a tensor of shape (n_nodes, n_features, 1), and the vector representation is a tensor of shape
    (n_nodes, n_features, 3). The representation can be accessed with the properties scalar and vector, respectively.

    Parameters
    ----------
    scalar: Optional[torch.Tensor]
        The scalar representation of the atoms. Default is None.
    vector: Optional[torch.Tensor]
        The vector representation of the atoms. Default is None.
    kwargs: Dict[str, torch.Tensor]
        Additional representations of the atoms. The keys are the degrees of the representations, and the values are

    Returns
    -------
    Representation

    """

    def __init__(self, **kwargs):
        """Initialize the representation with the given tensors."""

        scalar = kwargs.pop("scalar", None)
        if scalar is not None:
            kwargs["l0"] = scalar

        vector = kwargs.pop("vector", None)
        if vector is not None:
            kwargs["l1"] = vector

        self._tensors = {}
        for key, value in kwargs.items():
            self._tensors[key] = value

    @property
    def scalar(self) -> torch.Tensor:
        """Return the scalar representation tensor.

        Returns
        -------
        torch.Tensor
        """
        return self._tensors["l0"]

    @scalar.setter
    def scalar(self, value: torch.Tensor) -> None:
        """Set the scalar representation tensor.

        Parameters
        ----------
        value: torch.Tensor
            The new scalar representation tensor.

        Returns
        -------
        None

        """
        self._tensors["l0"] = value

    @property
    def vector(self) -> torch.Tensor:
        """Return the vector representation tensor.

        Returns
        -------
        torch.Tensor

        """
        return self._tensors["l1"]

    @vector.setter
    def vector(self, value: torch.Tensor) -> None:
        """Set the vector representation tensor.

        Parameters
        ----------
        value: torch.Tensor
            The new vector representation tensor.

        Returns
        -------
        None
        """
        self._tensors["l1"] = value

    def to_tensor(self, n_graphs: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert to tensor

        Convert the representation to a tensor. The representations are concatenated along the
        1st dimension, and the resulting tensor is returned along with slices and names of the
        representations.

        Parameters
        ----------
        n_graphs: int
            The number of graphs in the batch.

        Returns
        -------
        tensor: torch.Tensor
            The tensor representation of the batch with shape (n_nodes, n_features).
        slices: torch.Tensor
            The slices of the tensor representation with shape (n_graphs, n_slices).
        ls: torch.Tensor
            The degrees of the tensor representation with shape (n_graphs, 1).

        """
        nodes = self.scalar.shape[0]

        tensor = []
        slices = [0]
        ls = []
        for name, value in self._tensors.items():
            ls.append((value.shape[2] - 1) / 2)
            tensor.append(value.reshape(nodes, -1))
            slices.append(tensor[-1].shape[1])

        slices = torch.cumsum(torch.tensor(slices, dtype=int), dim=0).repeat(
            n_graphs, 1
        )
        tensor = torch.cat(tensor, dim=1)
        ls = torch.tensor(ls, dtype=int).repeat(n_graphs, 1)

        return tensor, slices, ls

    @classmethod
    def from_tensor(
        cls, tensor: torch.Tensor, slices: torch.Tensor, ls: torch.Tensor
    ) -> "Representation":
        """Get representation from tensor

        Create a representation from a tensor. The tensor is split into the different representations
        according to the slices and the degrees of the representations are given by ls.

        Parameters
        ----------
        tensor: torch.Tensor
            The tensor representation of the batch with shape (n_nodes, n_features).
        slices: torch.Tensor
            The slices of the tensor representation with shape (n_graphs, n_slices).
        ls: torch.Tensor
            The degrees of the tensor representation with shape (n_graphs, 1).

        Returns
        -------
        representation: Representation
            The representation object.

        """
        n_nodes = tensor.shape[0]
        slices = slices[0]
        degrees = ls[0]
        names = [f"l{degree}" for degree in degrees]
        d = {}
        for i, (degree, name) in enumerate(zip(degrees, names)):
            d[name] = tensor[:, slices[i].item() : slices[i + 1].item()].reshape(
                n_nodes, -1, 2 * degree.item() + 1
            )

        return cls(**d)


class AtomsGraph(Data):
    """Atomistic Graph Class

    Class defining a graph with atoms as nodes and edges formed between all atoms
    within a finite curoff.formed betw

    Parameters
    ----------
    pos: torch.Tensor
        The positions of the atoms with shape (n_atoms, 3).
    x: torch.Tensor
        The node features i.e atomic types of the graph with shape (n_nodes, 1).
    edge_index: torch.Tensor
        The edge index tensor of the graph with shape (2, n_edges).
    edge_attr: torch.Tensor
        The edge attributes of the graph with shape (n_edges, n_edge_features).
    y: Optional[torch.Tensor]
        The target tensor of the graph with shape (n_targets,).
    representation: Optional[Representation]
        The representation of the atoms in the graph.
    confinement: Optional[torch.Tensor]
        z-directional confinement of the atoms with shape (1,2).
    kwargs: Dict[str, torch.Tensor]

    """

    @classmethod
    def from_atoms(
        cls,
        atoms: Atoms,
        cutoff: int = 6.0,
        skin: Optional[float] = None,
        dtype: torch.dtype = torch.float,
        initialize_mask: Optional[bool] = None,
        confinement: Optional[Tuple[float, float]] = None,
        canonical_cell: bool = False,
    ) -> "AtomsGraph":
        """Create a graph from an ASE Atoms object.

        Parameters
        ----------
        atoms: Atoms
            The ASE Atoms object.
        cutoff: float
            The cutoff radius for the edges.
        dtype: torch.dtype
            The data type of the tensors.
        initialize_mask: Optional[bool]
            Whether to initialize the mask tensor.  When ``None`` (the
            default), the mask is initialised only when ``confinement`` is
            not provided (i.e. ``initialize_mask`` defaults to ``False``
            for template / confinement graphs).
        confinement: Optional[Tuple[float, float]]
            Optional z-directional confinement bounds ``(z_min, z_max)`` to
            attach to the graph.  When provided, a ``confinement`` tensor of
            shape ``(1, 2)`` is stored on the graph.  When ``None`` (the
            default), no confinement attribute is added.
        canonical_cell: bool
            When ``True`` (the default), the cell is stored in canonical
            lower-triangular form.  If the input cell is not already
            canonical, Cartesian positions are recomputed to preserve
            fractional coordinates and a warning is printed.  Set to
            ``False`` to store the cell exactly as provided by ASE (no
            rotation or recomputation is performed).

        Returns
        -------
        graph: AtomsGraph
            The graph object.

        """
        if initialize_mask is None:
            initialize_mask = confinement is None

        # Nodes: The initial node features are just the atomic numbers.
        kwargs = {
            "cutoff": cutoff,
        }
        if skin is not None:
            kwargs["skin"] = skin

        kwargs["x"] = torch.tensor(
            atoms.get_atomic_numbers(), dtype=torch.long
        ).reshape(-1)
        if initialize_mask:
            kwargs["mask"] = torch.zeros_like(kwargs["x"], dtype=torch.bool)

        # Canonicalize the cell (cellpar -> cell round-trip) and update
        # Cartesian positions so that fractional coordinates are preserved.
        # Use float64 for the round-trip to avoid precision loss from log/exp.
        cell_np = np.array(atoms.get_cell())
        pos_np = np.array(atoms.get_positions())
        cell_f64 = torch.tensor(cell_np, dtype=torch.float64)

        if not canonical_cell:
            # Caller opted out: store the cell exactly as provided by ASE.
            final_cell_f64 = cell_f64
            final_pos = torch.tensor(pos_np, dtype=dtype)
        elif cls._is_lower_triangular(cell_f64):
            # Cell is already in canonical lower-triangular form; keep it as-is
            # to avoid introducing floating-point rounding artefacts.
            final_cell_f64 = cell_f64
            final_pos = torch.tensor(pos_np, dtype=dtype)
        else:
            print(
                "AtomsGraph.from_atoms: cell is not in canonical lower-triangular "
                "form; canonicalizing. Cartesian positions will be recomputed to "
                "preserve fractional coordinates."
            )
            final_cell_f64 = cls.vector_to_cell(cls.cell_to_vectors(cell_f64)).view(3, 3)
            pos_f64 = torch.tensor(pos_np, dtype=torch.float64)
            frac_f64 = torch.linalg.solve(cell_f64.T, pos_f64.T).T
            final_pos = (frac_f64 @ final_cell_f64).to(dtype)

        kwargs["pos"] = final_pos
        if skin is not None:
            kwargs["reference_positions"] = final_pos.clone()
        kwargs["cell"] = final_cell_f64.to(dtype)
        kwargs["pbc"] = torch.tensor(atoms.get_pbc())
        if skin is not None:
            kwargs["reference_cell"] = kwargs["cell"].clone()
            kwargs["reference_pbc"] = kwargs["pbc"].clone()

        edge_index, shift_vectors = cls.make_graph(
            kwargs["pos"], kwargs["cell"], cutoff, kwargs["pbc"]
        )
        kwargs["edge_index"] = edge_index
        kwargs["shift_vectors"] = shift_vectors

        kwargs["n_atoms"] = torch.tensor([len(atoms)]).reshape(1, 1)

        if confinement is not None:
            kwargs["confinement"] = torch.tensor(
                list(confinement), dtype=dtype
            ).reshape(1, 2)

        return cls(**kwargs)

    @classmethod
    def empty(cls, cutoff: int = 6.0, skin: Optional[float] = None) -> "AtomsGraph":
        """Create an empty graph.

        Parameters
        ----------
        cutoff: float
            The cutoff radius for the edges.

        Returns
        -------
        graph: AtomsGraph
            The graph object.

        """
        return cls(
            x=torch.empty(0, dtype=torch.long),
            pos=torch.empty(0, 3),
            n_atoms=torch.tensor([0]),
            cell=torch.empty(3, 3),
            pbc=torch.tensor([True, True, True], dtype=torch.bool),
            cutoff=cutoff,
            skin=skin,
            # mask=torch.empty(0, dtype=torch.bool),
        )

    def add_batch_attr(self, key: str, value: torch.Tensor, type: str = "node") -> None:
        """Add a batch attribute to the graph.

        Parameters
        ----------
        key: str
            The key of the attribute.
        value: torch.Tensor
            The value of the attribute.
        type: str
            The type of the attribute. Can be either "node" or "graph"

        Returns
        -------
        None

        """
        self._store[key] = value

        if hasattr(self, "_slice_dict"):
            if type == "node":
                k = "x"
            elif type == "graph":
                k = "n_atoms"
            else:
                raise ValueError("Invalid type")

            self._slice_dict[key] = self._slice_dict[k]
            self._inc_dict[key] = self._inc_dict[k]

    def to_atoms(self) -> Atoms:
        """Convert the graph to an ASE Atoms object.

        Only works on unbatched graphs.

        Returns
        -------
        atoms: ase.Atoms
            The atoms object.

        """
        numbers = self.x.detach().cpu().numpy()
        positions = self.pos.detach().cpu().numpy()
        atoms = Atoms(
            numbers=numbers,
            positions=positions,
            cell=self.cell.detach().cpu().numpy(),
            pbc=self.pbc.detach().cpu().numpy(),
        )

        if "energy_prediction" in self._store:
            energy = self.energy_prediction.item()
            atoms.calc = SinglePointCalculator(atoms, energy=energy)
            atoms.calc.name = "AGeDi"

        if "forces_prediction" in self._store:
            forces = self.forces_prediction.detach().cpu().numpy()
            if hasattr(atoms, "calc") and atoms.calc is not None:
                atoms.calc.results["forces"] = forces
            else:
                atoms.calc = SinglePointCalculator(atoms, forces=forces)
                atoms.calc.name = "AGeDi"
            
        return atoms

    def _get_scalar_attr(self, key: str) -> Optional[float]:
        value = self._store.get(key)
        if value is None:
            return None
        if isinstance(value, torch.Tensor):
            flat = value.reshape(-1)
            if flat.numel() == 0:
                return None
            if isinstance(self, Batch):
                unique = flat.unique()
                if unique.numel() > 1:
                    raise ValueError(
                        f"All graphs in the batch must have the same {key}, "
                        f"but found: {unique.tolist()}."
                    )
                return float(unique[0].item())
            return float(flat[0].item())
        return float(value)

    def _skin(self) -> Optional[float]:
        skin = self._get_scalar_attr("skin")
        if skin is None or skin <= 0:
            return None
        return skin

    def _has_neighbor_reference(self) -> bool:
        return "reference_positions" in self._store

    def _neighbor_geometry_is_current(self) -> bool:
        return (
            "reference_cell" in self._store
            and "reference_pbc" in self._store
            and torch.equal(self.reference_cell, self.cell)
            and torch.equal(self.reference_pbc, self.pbc)
        )

    def _can_preserve_neighbor_cache(self) -> bool:
        return (
            self._skin() is not None
            and self._has_neighbor_reference()
            and self._neighbor_geometry_is_current()
            and "edge_index" in self._store
            and "shift_vectors" in self._store
        )

    @staticmethod
    def _neighbor_matrix_to_graph(
        neighbor_matrix: torch.Tensor,
        num_neighbors: torch.Tensor,
        cell: torch.Tensor,
        dtype: torch.dtype,
        fill_value: int,
        batch_idx: Optional[torch.Tensor] = None,
        neighbor_matrix_shifts: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        mask = neighbor_matrix != fill_value
        edge_index = torch.stack(
            [
                torch.where(mask)[0].to(torch.long),
                neighbor_matrix[mask].to(torch.long),
            ],
            dim=0,
        )

        if neighbor_matrix_shifts is None:
            shift_vectors = torch.zeros(
                (edge_index.shape[1], 3), dtype=dtype, device=neighbor_matrix.device
            )
            return edge_index, shift_vectors

        unit_shifts = neighbor_matrix_shifts[mask].to(cell.dtype)
        if batch_idx is None:
            shift_vectors = torch.einsum("ij,jk->ik", unit_shifts, cell.view(3, 3))
        else:
            edge_cells = torch.index_select(cell.view(-1, 3, 3), 0, batch_idx[edge_index[0]])
            shift_vectors = torch.einsum("ni,nij->nj", unit_shifts, edge_cells)
        return edge_index, shift_vectors.to(dtype)

    @staticmethod
    def _make_graph_matscipy(
        positions: torch.Tensor,
        cell: torch.Tensor,
        cutoff: float,
        pbc: torch.Tensor,
        dtype: Optional[torch.dtype] = None,
        batch_idx: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        output_dtype = dtype or positions.dtype
        if batch_idx is None:
            i, j, shifts = matscipy_neighbour_list(
                "ijS",
                positions=positions.detach().cpu().numpy(),
                cell=cell.detach().cpu().numpy(),
                cutoff=cutoff,
                pbc=pbc.detach().cpu().numpy(),
            )
            edge_index = torch.tensor(
                np.stack([i, j]), dtype=torch.long, device=positions.device
            )
            unit_shifts = torch.tensor(
                shifts, dtype=cell.dtype, device=positions.device
            )
            shift_vectors = torch.einsum("ij,jk->ik", unit_shifts, cell.view(3, 3))
            return edge_index, shift_vectors.to(output_dtype)

        edge_index_parts = []
        shift_vector_parts = []
        batch_cells = cell.view(-1, 3, 3)
        batch_pbc = pbc.view(-1, 3)
        num_graphs = int(batch_idx.max().item()) + 1 if batch_idx.numel() > 0 else 0
        for graph_idx in range(num_graphs):
            atom_idx = torch.where(batch_idx == graph_idx)[0]
            if atom_idx.numel() == 0:
                continue
            local_edge_index, local_shift_vectors = AtomsGraph._make_graph_matscipy(
                positions[atom_idx],
                batch_cells[graph_idx],
                cutoff,
                batch_pbc[graph_idx],
                dtype=output_dtype,
            )
            edge_index_parts.append(atom_idx[local_edge_index])
            shift_vector_parts.append(local_shift_vectors)

        if not edge_index_parts:
            return (
                torch.empty((2, 0), dtype=torch.long, device=positions.device),
                torch.empty((0, 3), dtype=output_dtype, device=positions.device),
            )

        return torch.cat(edge_index_parts, dim=1), torch.cat(shift_vector_parts, dim=0)

    @staticmethod
    def make_graph(
        positions: torch.Tensor,
        cell: torch.Tensor,
        cutoff: int,
        pbc: torch.Tensor,
        dtype: torch.dtype = None,
        batch_idx: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create the graph-edges from the positions and cell.

        Parameters
        ----------
        positions: torch.Tensor
            The positions of the atoms.
        cell: torch.Tensor
            The cell of the system.
        cutoff: float
            The cutoff radius for the edges.
        pbc: torch.Tensor
            The periodic boundary conditions.
        dtype: torch.dtype
            The data type of the output.

        Returns
        -------
        edge_index: torch.Tensor
            The edge index tensor.
        shift_vectors: torch.Tensor
            The shift vectors tensor.

        """

        with torch.no_grad():
            if nvidia_neighbor_list is None:
                return AtomsGraph._make_graph_matscipy(
                    positions,
                    cell,
                    cutoff,
                    pbc,
                    dtype=dtype,
                    batch_idx=batch_idx,
                )

            kwargs = {
                "positions": positions,
                "cutoff": cutoff,
                "cell": cell.view(-1, 3, 3) if batch_idx is not None else cell,
                "pbc": pbc,
                "return_neighbor_list": True,
            }
            if batch_idx is not None:
                kwargs["batch_idx"] = batch_idx

            edge_index, _, shifts = nvidia_neighbor_list(**kwargs)
            edge_index = edge_index.to(torch.long)
            if batch_idx is None:
                shift_vectors = torch.einsum(
                    "ij,jk->ik", shifts.to(cell.dtype), cell.view(3, 3)
                )
            else:
                batch_cells = cell.view(-1, 3, 3)
                edge_cells = torch.index_select(batch_cells, 0, batch_idx[edge_index[0]])
                shift_vectors = torch.einsum(
                    "ni,nij->nj", shifts.to(edge_cells.dtype), edge_cells
                )

        return edge_index, shift_vectors

    # @batched(update_keys=["edge_index", "shift_vectors"])
    def update_graph(self) -> bool:
        """Update the graph with new edges

        This should be called after changing any of the positions or cell.

        Returns
        -------
        rebuilt: bool
            ``True`` when the neighbor list was fully recomputed, ``False``
            when the skin check determined that the existing list is still
            valid and no rebuild was performed.
        """

        cutoff = self._get_scalar_attr("cutoff")
        if cutoff is None:
            raise ValueError(
                "cutoff must be set on the graph before calling update_graph()."
            )

        skin = self._skin()
        if isinstance(self, Batch):
            batch_idx = self.batch.to(torch.int32)
            batch_ptr = self.ptr.to(torch.int32)
            cell = self.cell.view(-1, 3, 3).contiguous()
            pbc = self.pbc.view(-1, 3).contiguous()

            if (
                skin is not None
                and batch_naive_neighbor_list is not None
                and batch_neighbor_list_needs_rebuild is not None
                and self._has_neighbor_reference()
                and self._neighbor_geometry_is_current()
                and "neighbor_matrix" in self._store
                and "num_neighbors" in self._store
            ):
                # Reuse cached cell inverse when cell hasn't changed to avoid
                # recomputing torch.linalg.inv on every step.
                if "reference_cell_inv" in self._store:
                    cell_inv = self._store["reference_cell_inv"]
                else:
                    cell_inv = torch.linalg.inv(cell).contiguous()
                rebuild_flags = batch_neighbor_list_needs_rebuild(
                    reference_positions=self.reference_positions,
                    current_positions=self.pos,
                    batch_idx=batch_idx,
                    skin_distance_threshold=skin,
                    update_reference_positions=True,
                    cell=cell,
                    cell_inv=cell_inv,
                    pbc=pbc,
                )
                if not torch.any(rebuild_flags):
                    return False
                # If every system needs a rebuild, use the faster full-rebuild
                # kernel path instead of the selective-rebuild path.
                if torch.all(rebuild_flags):
                    rebuild_flags = None
            else:
                rebuild_flags = None

            if batch_naive_neighbor_list is None or skin is None:
                if (
                    skin is not None
                    and self._has_neighbor_reference()
                    and self._neighbor_geometry_is_current()
                    and "edge_index" in self._store
                    and "shift_vectors" in self._store
                ):
                    max_disp = (self.pos - self.reference_positions).norm(dim=-1).max()
                    if max_disp <= skin / 2:
                        return False
                self.edge_index, self.shift_vectors = self.make_graph(
                    self.pos, cell, cutoff, pbc, batch_idx=batch_idx
                )
            else:
                results = batch_naive_neighbor_list(
                    positions=self.pos,
                    cutoff=cutoff,
                    batch_idx=batch_idx,
                    batch_ptr=batch_ptr,
                    cell=cell,
                    pbc=pbc,
                    neighbor_matrix=self._store.get("neighbor_matrix"),
                    neighbor_matrix_shifts=self._store.get("neighbor_matrix_shifts"),
                    num_neighbors=self._store.get("num_neighbors"),
                    shift_range_per_dimension=self._store.get("shift_range_per_dimension"),
                    num_shifts_per_system=self._store.get("num_shifts_per_system"),
                    max_shifts_per_system=self._store.get("max_shifts_per_system"),
                    max_atoms_per_system=self._store.get("max_atoms_per_system"),
                    rebuild_flags=rebuild_flags,
                )
                neighbor_matrix, num_neighbors, neighbor_matrix_shifts = results
                fill_value = self.pos.shape[0]
                self._store["neighbor_matrix"] = neighbor_matrix
                self._store["neighbor_matrix_shifts"] = neighbor_matrix_shifts
                self._store["num_neighbors"] = num_neighbors
                self.edge_index, self.shift_vectors = self._neighbor_matrix_to_graph(
                    neighbor_matrix=neighbor_matrix,
                    num_neighbors=num_neighbors,
                    neighbor_matrix_shifts=neighbor_matrix_shifts,
                    cell=cell,
                    dtype=self.pos.dtype,
                    fill_value=fill_value,
                    batch_idx=batch_idx,
                )
            if skin is not None and rebuild_flags is None:
                self.add_batch_attr("reference_positions", self.pos.clone(), type="node")
            if skin is not None:
                # Store in the same batched layout as self.cell / self.pbc (e.g.
                # (N*3, 3) and (N*3,)) so that Batch.to_data_list() can use the
                # existing _slice_dict entries to split them back correctly.
                self._store["reference_cell"] = self.cell.clone()
                self._store["reference_pbc"] = self.pbc.clone()
                # Cache the cell inverse so it doesn't need to be recomputed on
                # every step when the cell hasn't changed.
                if batch_naive_neighbor_list is not None:
                    self._store["reference_cell_inv"] = torch.linalg.inv(cell).contiguous()
        else:
            if skin is not None and self._can_preserve_neighbor_cache():
                if neighbor_list_needs_rebuild is not None:
                    rebuild_needed = neighbor_list_needs_rebuild(
                        reference_positions=self.reference_positions,
                        current_positions=self.pos,
                        skin_distance_threshold=skin,
                        update_reference_positions=True,
                        cell=self.cell.view(1, 3, 3),
                        cell_inv=torch.linalg.inv(self.cell.view(1, 3, 3)),
                        pbc=self.pbc.view(1, 3),
                    )
                    if not torch.any(rebuild_needed):
                        return False
                else:
                    max_disp = (self.pos - self.reference_positions).norm(dim=-1).max()
                    if max_disp <= skin / 2:
                        return False

            self.edge_index, self.shift_vectors = self.make_graph(
                self.pos,
                self.cell,
                cutoff,
                self.pbc,
            )
            if skin is not None:
                self._store["reference_positions"] = self.pos.clone()
                self._store["reference_cell"] = self.cell.clone()
                self._store["reference_pbc"] = self.pbc.clone()
        return True

    def clear_graph(self) -> None:
        """Clear the graph removing all edges

        Returns
        -------
        None
        """
        for key in NEIGHBOR_CACHE_KEYS:
            if key in self._store:
                del self._store[key]

    def __len__(self) -> int:
        """Return the number of atoms in the graph.

        Returns
        -------
        n_atoms: int
            The number of atoms in the graph.

        """
        return self.pos.shape[0]

    @property
    def cell(self) -> torch.Tensor:
        """Return the canonical cell matrix of the graph.

        Returns
        -------
        cell: torch.Tensor
            The cell matrix of shape ``(3, 3)``.
        """
        return self._store["cell"]

    @cell.setter
    def cell(self, cell: torch.Tensor) -> None:
        """Set the cell matrix, preserving fractional coordinates.

        Cartesian positions are recomputed when both ``pos`` and an old
        ``cell`` are already present so that fractional coordinates remain
        unchanged.  No canonicalization is performed; use
        :meth:`from_atoms` with ``canonical_cell=True`` if you need the
        cell stored in canonical lower-triangular form.

        Parameters
        ----------
        cell: torch.Tensor
            The new cell matrix.

        Returns
        -------
        None
        """
        # Only invalidate the edge cache when the cell is being *changed*
        # (i.e. a cell already existed) on an individual graph.  Skip during
        # Batch construction (where edge_index is already correct) and skip
        # during initial construction (where no prior cell exists yet).
        cell_was_set = "cell" in self._store
        # Preserve fractional coordinates when both pos and old cell exist.
        if "pos" in self._store and cell_was_set:
            frac = self.pos_to_frac(self.pos)
            self._store["cell"] = cell
            pos = self.frac_to_pos(frac)
            Data.pos.fset(self, pos)
            if "frac" in self._store:
                del self._store["frac"]
        else:
            self._store["cell"] = cell

        if cell_was_set and not isinstance(self, Batch) and (
            "edge_index" in self._store or "shift_vectors" in self._store
        ):
            self.clear_graph()

    @Data.pos.setter
    def pos(self, pos: torch.Tensor) -> None:
        """Set the positions of the atoms.

        Parameters
        ----------
        pos: torch.Tensor
            The new positions of the atoms.

        Returns
        -------
        None

        """
        if "pos" in self._store and not self._can_preserve_neighbor_cache():
            self.clear_graph()
        if "frac" in self._store:
            del self["frac"]
        if "mask" in self._store:
            pos[self.positions_mask] = self.pos[self.positions_mask]
        Data.pos.fset(self, pos)

        # if "cell" in self._store:
        #     f = self.pos_to_frac(self.pos)
        #     self.add_batch_attr("frac", f, type="node")

    @property
    def frac(self) -> torch.Tensor:
        """Return the fractional coordinates of the positions

        Returns
        -------
        frac: torch.Tensor
            The fractional coordinates of the atoms.

        """
        if "frac" in self._store:
            return self["frac"]
        else:
            f = self.pos_to_frac(self.pos)
            self.add_batch_attr("frac", f, type="node")
            return f

    @frac.setter
    def frac(self, frac: torch.Tensor) -> None:
        """Set fractional coordinates.

        All positions are also updated.

        Parameters
        ----------
        frac: torch.Tensor
            The fractional coordinates of the atoms.

        Returns
        -------
        None

        """
        frac %= 1
        if "frac" in self._store and not self._can_preserve_neighbor_cache():
            self.clear_graph()
        if "mask" in self._store:
            frac[self.positions_mask] = self.frac[self.positions_mask]

        self.add_batch_attr("frac", frac, type="node")

        r = self.frac_to_pos(frac)
        Data.pos.fset(self, r)

    def frac_to_pos(self, f: torch.Tensor) -> torch.Tensor:
        """Fraction -> Cartesian coordinates.

        Convert fractional coordinates to cartesian coordinates.

        Parameters
        ----------
        f: torch.Tensor
            The fractional coordinates.

        Returns
        -------
        r: torch.Tensor
            The cartesian coordinates.

        """
        cells = self.cell
        if isinstance(self, Batch):
            cells = cells.view(-1, 3, 3)[self.batch]
        else:
            cells = cells.repeat(f.shape[0], 1, 1)

        r = (f.unsqueeze(1) @ cells).reshape(-1, 3)
        return r

    def pos_to_frac(self, r: torch.Tensor) -> torch.Tensor:
        """Cartesian -> Fractional coordinates.

        Convert cartesian coordinates to fractional coordinates.

        Parameters
        ----------
        r: torch.Tensor
            The cartesian coordinates.

        Returns
        -------
        f: torch.Tensor
            The fractional coordinates.

        """
        cells = self.cell
        if isinstance(self, Batch):
            cells = cells.view(-1, 3, 3)[self.batch]
        else:
            cells = cells.repeat(r.shape[0], 1, 1)

        f = torch.linalg.solve(torch.transpose(cells, 1, 2), r)
        return f

    @Data.x.setter
    def x(self, x: torch.Tensor) -> None:
        """Set the atomic types of the graph

        Parameters
        ----------
        x: torch.Tensor
            The atomic types of the graph

        Returns
        -------
        None

        """
        if "mask" in self._store:
            x[self.mask] = self.x[self.mask]
        Data.x.fset(self, x)

    @property
    def positions_mask(self) -> torch.Tensor:
        """Return the mask of the positions that are fixed.

        True for fixed atom-positions and else false.

        Returns
        -------
        mask: torch.Tensor
            The mask of the positions that are fixed.

        """
        pos_mask = torch.zeros_like(self.pos, dtype=torch.bool)
        pos_mask[self.mask, :] = True
        return pos_mask

    @property
    def time(self) -> torch.Tensor:
        """Return the time of the graph.

        Returns
        -------
        time: torch.Tensor
            The time of the graph.
        """
        return self["time"] if "time" in self._store else None

    @time.setter
    def time(self, t: torch.Tensor) -> None:
        """Set the time of the graph.

        Parameters
        ----------
        t: torch.Tensor
            The time of the graph.

        Returns
        -------
        None

        """
        if "mask" in self._store:
            t = self.apply_mask(t.squeeze()).unsqueeze(1)
        # self._store.t = t
        self.add_batch_attr("time", t, type="node")

    @property
    def representation(self) -> Representation:
        """Return the representation of the graph.

        Returns
        -------
        representation: Representation
            The representation of the graph.

        """
        return (
            Representation.from_tensor(self.repr, self.repr_slices, self.repr_ls)
            if "repr" in self._store
            else None
        )

    @representation.setter
    def representation(self, representation: Representation) -> None:
        """Set the representation of the graph.

        Parameters
        ----------
        representation: Representation
            The representation of the graph.

        Returns
        -------
        None

        """
        n_graphs = self.num_graphs if "num_graphs" in self._store else 1
        tensor, slices, ls = representation.to_tensor(n_graphs)

        self.add_batch_attr("repr", tensor, type="node")
        self.add_batch_attr(
            "repr_slices", slices.repeat(self.n_atoms.shape[0], 1), type="graph"
        )
        self.add_batch_attr(
            "repr_ls", ls.repeat(self.n_atoms.shape[0], 1), type="graph"
        )

    def wrap_positions(self) -> None:
        """Wrap the positions of the atoms to the unit cell.

        Returns
        -------
        None

        """
        pbc = torch.repeat_interleave(self.pbc.view(-1, 3), self.n_atoms.view(-1), dim=0)
        f = self.frac
        f[pbc] = f[pbc] % 1
        self.pos = self.frac_to_pos(f)


    def apply_mask(self, x: torch.Tensor, val: float = 0.0) -> torch.Tensor:
        """Apply the mask to the tensor x.

        Parameters
        ----------
        x: torch.Tensor
            The tensor to apply the mask to.
        val: float
            The value to set the masked values to.

        Returns
        -------
        x: torch.Tensor
            The tensor with the mask applied.

        """

        if x.shape == self.mask.shape:
            x[self.mask] = val
        elif x.shape == self.positions_mask.shape:
            x[self.positions_mask] = val
        else:
            raise ValueError("Invalid shape for mask.")
        return x

    @property
    def confinement(self) -> torch.Tensor:
        """Return the confinement of the graph.

        Returns
        -------
        confinement: torch.Tensor
            The confinement of the graph.

        """
        return self["confinement"] if "confinement" in self._store else None

    @confinement.setter
    def confinement(self, confinement: torch.Tensor) -> None:
        """Set the confinement bounds for the graph.

        Parameters
        ----------
        confinement : torch.Tensor
            Tensor of shape ``(1, 2)`` containing the lower and upper
            Z-confinement bounds.
        """
        self.add_batch_attr("confinement", confinement, type="graph")


    @property
    def cellpar(self) -> torch.Tensor:
        """Return the cell parameters of the graph."""
        return self.cell_to_vectors(self.cell)

    @cellpar.setter
    def cellpar(self, cellpar: torch.Tensor) -> None:
        """Set the cell parameters of the graph.

        Parameters
        ----------
        cellpar: torch.Tensor
            The cell parameters of the graph.

        Returns
        -------
        None

        """
        cell = self.vector_to_cell(cellpar)
        if cell.ndim == 3 and cell.shape[0] == 1:
            cell = cell.reshape(3, 3)
        self.cell = cell
        
    @staticmethod
    def _is_lower_triangular(cell: torch.Tensor) -> bool:
        """Return True if *cell* is in canonical lower-triangular form.

        A cell matrix is considered canonical when the three strictly
        upper-triangular entries (cell[0,1], cell[0,2], cell[1,2]) are
        all zero (within a tight floating-point tolerance of 1e-10).

        Parameters
        ----------
        cell : torch.Tensor
            The cell matrix.

        Returns
        -------
        bool
            True if the cell is already lower-triangular.
        """
        c = cell.reshape(3, 3)
        return bool(
            c[0, 1].abs() < 1e-10
            and c[0, 2].abs() < 1e-10
            and c[1, 2].abs() < 1e-10
        )

    @staticmethod
    def cell_to_vectors(cell: torch.Tensor) -> torch.Tensor:
        """Convert cell matrix to cell parameters.

        Parameters
        ----------
        cell : torch.Tensor
            The cell matrix of shape ``(N, 3)`` or ``(N, 3, 3)``.

        Returns
        -------
        torch.Tensor
            The cell parameters of shape ``(N, 6)``.

        """
        cell = cell.view(-1, 3, 3)
        
        a = torch.norm(cell[..., 0, :], dim=-1)
        b = torch.norm(cell[..., 1, :], dim=-1)
        c = torch.norm(cell[..., 2, :], dim=-1)

        alpha = torch.acos(
            torch.sum(cell[..., 1, :] * cell[..., 2, :], dim=-1) / (b * c)
        )
        beta = torch.acos(
            torch.sum(cell[..., 0, :] * cell[..., 2, :], dim=-1) / (a * c)
        )
        gamma = torch.acos(
            torch.sum(cell[..., 0, :] * cell[..., 1, :], dim=-1) / (a * b)
        )

        # alpha = alpha * 180 / torch.pi
        # beta = beta * 180 / torch.pi
        # gamma = gamma * 180 / torch.pi

        a,b,c = torch.log(a), torch.log(b), torch.log(c)
        alpha, beta, gamma = alpha - torch.pi / 2, beta - torch.pi / 2, gamma - torch.pi / 2

        return torch.stack([a, b, c, alpha, beta, gamma], dim=-1)

    @staticmethod
    def vector_to_cell(cellpar: torch.Tensor) -> torch.Tensor:
        """Convert cell parameters to cell matrix.

        Parameters
        ----------
        cellpar : torch.Tensor
            The cell parameters of shape ``(N, 6)``.

        Returns
        -------
        torch.Tensor
            The cell matrix of shape ``(N, 3, 3)`` where each row is a lattice vector.

        """
        a, b, c, alpha, beta, gamma = cellpar.unbind(-1)

        a, b, c = torch.exp(a), torch.exp(b), torch.exp(c)
        alpha, beta, gamma = alpha + torch.pi / 2, beta + torch.pi / 2, gamma + torch.pi / 2
        

        cos_alpha = torch.cos(alpha)
        cos_beta = torch.cos(beta)
        cos_gamma = torch.cos(gamma)
        sin_gamma = torch.sin(gamma)

        cell = torch.zeros(cellpar.shape[:-1] + (3, 3), device=cellpar.device, dtype=cellpar.dtype)
        cell[..., 0, 0] = a
        cell[..., 1, 0] = b * cos_gamma
        cell[..., 2, 0] = c * cos_beta

        cell[..., 1, 1] = b * sin_gamma
        cell[..., 2, 1] = c * (cos_alpha - cos_beta * cos_gamma) / sin_gamma

        cell[..., 2, 2] = c * torch.sqrt(torch.clamp(1 - cos_beta ** 2 - cell[..., 2, 1] ** 2 / c ** 2, min=0))

        return cell

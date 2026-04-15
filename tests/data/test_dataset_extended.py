"""Tests for the Dataset class."""
import torch
import pytest
from ase.build import molecule, bulk
from ase.constraints import FixAtoms

from agedi.data.dataset import Dataset
from agedi.data import AtomsGraph


def _make_molecules(n=6):
    atoms_list = []
    for name in ["H2O", "NH3", "CH4", "H2", "N2", "CO"]:
        a = molecule(name)
        a.set_cell([10.0, 10.0, 10.0])
        a.set_pbc(True)
        a.center()
        atoms_list.append(a)
    return atoms_list[:n]


def _make_graphs(n=4):
    return [AtomsGraph.from_atoms(a) for a in _make_molecules(n)]


class TestDatasetAddAndSetup:
    def test_add_atoms_data_populates_dataset(self):
        ds = Dataset(batch_size=2, n_train=0.8, n_val=0.2, n_test=0.0)
        ds.add_atoms_data(_make_molecules(4))
        assert len(ds.dataset) == 4

    def test_add_atoms_data_twice_extends(self):
        ds = Dataset(batch_size=2, n_train=1.0, n_val=0.0, n_test=0.0)
        ds.add_atoms_data(_make_molecules(2))
        ds.add_atoms_data(_make_molecules(2))
        assert len(ds.dataset) == 4

    def test_add_graph_data_populates_dataset(self):
        ds = Dataset(batch_size=2, n_train=1.0, n_val=0.0, n_test=0.0)
        ds.add_graph_data(_make_graphs(3))
        assert len(ds.dataset) == 3

    def test_add_atoms_data_with_properties(self):
        ds = Dataset(batch_size=2, n_train=1.0, n_val=0.0, n_test=0.0)
        mols = _make_molecules(2)
        props = [{"energy": -1.0}, {"energy": -2.0}]
        ds.add_atoms_data(mols, properties=props)
        assert hasattr(ds.dataset[0], "energy")

    def test_add_atoms_data_mask_fixed(self):
        from ase.build import fcc111
        surf = fcc111("Au", (2, 2, 2), vacuum=5)
        surf.set_pbc(True)
        surf.set_constraint(FixAtoms(indices=[0, 1]))
        ds = Dataset(batch_size=2, n_train=1.0, n_val=0.0, n_test=0.0)
        ds.add_atoms_data([surf], mask_method="MaskFixed")
        assert ds.dataset[0].mask[0] == True
        assert ds.dataset[0].mask[1] == True

    def test_add_atoms_data_invalid_mask_raises(self):
        ds = Dataset(batch_size=2, n_train=1.0, n_val=0.0, n_test=0.0)
        with pytest.raises(ValueError):
            ds.add_atoms_data(_make_molecules(1), mask_method="invalid_mask")

    def test_add_atoms_data_confinement(self):
        ds = Dataset(batch_size=2, n_train=1.0, n_val=0.0, n_test=0.0)
        ds.add_atoms_data(_make_molecules(1), confinement=[2.0, 8.0])
        assert ds.dataset[0].confinement is not None

    def test_setup_splits_data(self):
        ds = Dataset(batch_size=2, n_train=0.8, n_val=0.2, n_test=0.0)
        ds.add_atoms_data(_make_molecules(5))
        ds.setup()
        assert len(ds.train_idx) + len(ds.val_idx) + len(ds.test_idx) == 5

    def test_dataloaders_are_accessible(self):
        ds = Dataset(batch_size=2, n_train=1.0, n_val=0.0, n_test=0.0)
        ds.add_atoms_data(_make_molecules(4))
        ds.setup()
        assert ds.train_dataloader() is not None
        assert ds.val_dataloader() is not None
        assert ds.test_dataloader() is not None

    def test_setup_is_idempotent(self):
        ds = Dataset(batch_size=2, n_train=0.8, n_val=0.2, n_test=0.0)
        ds.add_atoms_data(_make_molecules(5))
        ds.setup()
        first_train = list(ds.train_idx)
        ds.setup()
        assert list(ds.train_idx) == first_train

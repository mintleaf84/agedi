import numpy as np
import torch
from ase.build import molecule

from agedi import (
    create_dataset,
    create_diffusion,
    load_diffusion,
    sample,
    train,
    train_from_atoms,
)
from agedi.data import AtomsGraph, Dataset
from agedi.diffusion import Diffusion


def _test_atoms():
    atoms = molecule("H2O")
    atoms.set_cell([10.0, 10.0, 10.0])
    atoms.set_pbc(True)
    atoms.center()
    return atoms


def test_create_diffusion():
    diffusion = create_diffusion(noisers=("positions",))
    assert isinstance(diffusion, Diffusion)


def test_create_dataset():
    dataset = create_dataset([_test_atoms(), _test_atoms()], batch_size=2)
    assert isinstance(dataset, Dataset)
    assert len(dataset.dataset) == 2


def test_train_uses_provided_trainer():
    class DummyTrainer:
        def __init__(self):
            self.called = False

        def fit(self, diffusion_model, data):
            self.called = True
            self.diffusion_model = diffusion_model
            self.data = data

    diffusion = create_diffusion(noisers=("positions",))
    dataset = create_dataset([_test_atoms(), _test_atoms()], batch_size=2)
    trainer = DummyTrainer()

    returned = train(diffusion, dataset, trainer=trainer)
    assert returned is trainer
    assert trainer.called
    assert trainer.diffusion_model is diffusion
    assert trainer.data is dataset


def test_sample_returns_atoms(diffusion):
    structures = sample(
        diffusion,
        n_samples=1,
        steps=2,
        atomic_numbers=[6, 8, 8],
        cell=np.diag([10.0, 10.0, 10.0]),
        property={"property": 1.0},
    )
    assert len(structures) == 1
    assert structures[0].positions.shape == (3, 3)


def test_load_diffusion(tmp_path):
    diffusion = create_diffusion(noisers=("positions",), lr=2e-4)
    log_dir = tmp_path / "logs" / "version_0"
    checkpoint_dir = log_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True)

    hparams = {
        "model": "PaiNN",
        "cutoff": 6.0,
        "feature_size": 64,
        "n_blocks": 4,
        "noisers": ["positions"],
        "lr": 2e-4,
        "lr_factor": 0.95,
        "lr_patience": 100,
        "conditioning": "none",
        "conditioning_type": "scalar",
        "style": "Default",
    }

    (log_dir / "hparams.yaml").write_text(
        "\n".join(f"{k}: {v}" for k, v in hparams.items())
    )
    torch.save({"state_dict": diffusion.state_dict()}, checkpoint_dir / "last_model.ckpt")

    loaded = load_diffusion(log_dir)
    assert isinstance(loaded, Diffusion)


def test_train_from_atoms_with_custom_trainer():
    class DummyTrainer:
        def __init__(self):
            self.fit_calls = 0

        def fit(self, diffusion_model, data):
            self.fit_calls += 1
            self.diffusion_model = diffusion_model
            self.data = data

    trainer = DummyTrainer()
    diffusion, dataset, used_trainer = train_from_atoms(
        [_test_atoms(), _test_atoms()],
        noisers=("positions",),
        trainer=trainer,
    )
    assert isinstance(diffusion, Diffusion)
    assert isinstance(dataset, Dataset)
    assert used_trainer is trainer
    assert trainer.fit_calls == 1
    assert isinstance(trainer.data.dataset[0], AtomsGraph)

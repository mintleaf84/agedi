import numpy as np
import torch
import yaml
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
    """load_diffusion should reconstruct the model from the Hydra hparams format."""
    diffusion = create_diffusion(noisers=("positions",), lr=2e-4)
    log_dir = tmp_path / "logs" / "version_0"
    checkpoint_dir = log_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True)

    hparams = {"diffusion": diffusion.get_hparams()}

    with open(log_dir / "hparams.yaml", "w") as f:
        yaml.dump(hparams, f, default_flow_style=False)

    torch.save({"state_dict": diffusion.state_dict()}, checkpoint_dir / "last_model.ckpt")

    loaded = load_diffusion(log_dir)
    assert isinstance(loaded, Diffusion)


def test_load_diffusion_missing_diffusion_key(tmp_path):
    """load_diffusion should raise ValueError when hparams.yaml lacks 'diffusion' key."""
    log_dir = tmp_path / "logs" / "version_0"
    checkpoint_dir = log_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True)

    (log_dir / "hparams.yaml").write_text("model: PaiNN\ncutoff: 6.0\n")
    torch.save({}, checkpoint_dir / "last_model.ckpt")

    try:
        load_diffusion(log_dir)
        assert False, "Expected ValueError"
    except ValueError as e:
        assert "diffusion" in str(e)


def test_diffusion_get_hparams():
    """diffusion.get_hparams() should return a nested dict with all required keys."""
    diffusion = create_diffusion(noisers=("positions",))
    hparams = diffusion.get_hparams()

    assert "_target_" in hparams
    assert "score_model" in hparams
    assert "noisers" in hparams
    assert "_target_" in hparams["score_model"]
    assert "representation" in hparams["score_model"]
    assert "conditionings" in hparams["score_model"]
    assert "heads" in hparams["score_model"]
    assert len(hparams["noisers"]) == 1
    noiser_hparams = hparams["noisers"][0]
    assert "_target_" in noiser_hparams
    assert "sde" in noiser_hparams
    assert "_target_" in noiser_hparams["sde"]


def test_diffusion_on_fit_start_writes_hparams(tmp_path):
    """Diffusion.on_fit_start should write hparams.yaml to the log directory."""
    from unittest.mock import MagicMock

    diffusion = create_diffusion(noisers=("positions",))
    log_dir = tmp_path / "version_0"
    log_dir.mkdir(parents=True)

    # Simulate what Lightning does: set self.trainer on the module
    mock_trainer = MagicMock()
    mock_trainer.logger.log_dir = str(log_dir)
    diffusion._trainer = mock_trainer  # Lightning uses _trainer internally

    # Manually invoke the hook (bypasses Lightning's internal wiring)
    diffusion.on_fit_start()

    hparams_file = log_dir / "hparams.yaml"
    assert hparams_file.exists(), "hparams.yaml was not written"
    with open(hparams_file) as f:
        saved = yaml.safe_load(f)
    assert "diffusion" in saved
    assert "_target_" in saved["diffusion"]


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

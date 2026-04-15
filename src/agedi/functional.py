from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union
import warnings

import numpy as np
import torch
import yaml
from ase import Atoms
from lightning import Trainer
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger

from agedi import Diffusion
from agedi.cli.train import data_info, get_conditioning, get_noisers, get_package
from agedi.data import AtomsGraph, Dataset
from agedi.data.callbacks import TrainingPhase
from agedi.data.transforms import Repeat
from agedi.models import ScoreModel


def create_diffusion(
    model: str = "PaiNN",
    cutoff: float = 6.0,
    feature_size: int = 64,
    n_blocks: int = 4,
    noisers: Sequence[str] = ("positions",),
    style: str = "Default",
    conditioning: str = "none",
    conditioning_type: str = "scalar",
    confinement: Optional[Tuple[float, float]] = None,
    lr: float = 1e-4,
    lr_factor: float = 0.95,
    lr_patience: int = 100,
    guidance_weight: float = -1.0,
    device: Optional[Union[str, torch.device]] = None,
) -> Diffusion:
    """Create a diffusion model for script-based training and sampling."""
    torch_device = torch.device(device) if device is not None else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    conditioning_modules = get_conditioning(conditioning, type=conditioning_type)
    head_dim = feature_size + sum(module.output_dim for module in conditioning_modules)

    translator, representation, heads = get_package(
        model,
        cutoff,
        noisers,
        feature_size,
        n_blocks,
        head_dim=head_dim,
    )

    confined = confinement is not None and "positions" in noisers
    noiser_modules = get_noisers(noisers, style, confined=confined)

    score_model = ScoreModel(
        translator=translator,
        representation=representation,
        conditionings=conditioning_modules,
        heads=list(heads),
        w=guidance_weight,
    )

    return Diffusion(
        score_model=score_model,
        noisers=noiser_modules,
        optim_config={"lr": lr},
        scheduler_config={"factor": lr_factor, "patience": lr_patience},
    ).to(torch_device)


def create_dataset(
    data: Sequence[Atoms],
    cutoff: float = 6.0,
    batch_size: int = 64,
    mask: str = "none",
    confinement: Optional[Tuple[float, float]] = None,
    conditioning: str = "none",
    conditioning_type: str = "scalar",
    repeat: Optional[int] = None,
) -> Dataset:
    """Create and setup an AGeDi Dataset from ASE Atoms objects."""
    phase_transforms = None
    if repeat is not None:
        if repeat < 2:
            raise ValueError(f"repeat must be at least 2, got {repeat}")

        property_kinds = {"mask": "node", "confinement": "none"}
        if conditioning != "none":
            property_kinds[conditioning] = (
                "node" if conditioning_type == "node" else "none"
            )
        phase_transforms = [[]]
        for i in range(2, repeat + 1):
            phase_transforms.append([Repeat((i, i, 1), property=property_kinds)])

    dataset = Dataset(
        cutoff=cutoff,
        batch_size=batch_size,
        phase_transforms=phase_transforms,
    )

    properties = None
    if conditioning != "none":
        properties = []
        for sample in data:
            value = None
            try:
                value = getattr(sample, f"get_{conditioning}")()
            except AttributeError:
                pass

            try:
                value = sample.info[conditioning]
            except KeyError:
                pass

            if value is None:
                value = 0
                warnings.warn(
                    f"Conditioning '{conditioning}' not found for one sample; using 0.",
                    stacklevel=2,
                )

            properties.append({conditioning: value})

    dataset.add_atoms_data(
        list(data),
        mask_method=mask,
        confinement=confinement,
        properties=properties,
    )
    dataset.setup()
    return dataset


def create_trainer(
    *,
    epochs: int = -1,
    time_hours: Optional[int] = 24,
    logger: str = "tensorboard",
    log_dir: str = "logs",
    project: str = "agedi",
    name: str = "agedi",
    log_interval: int = 10,
    gradient_clip_val: float = 10.0,
    progress_bar: bool = False,
    repeat: Optional[int] = None,
    repeat_epoch: Optional[int] = None,
    hparams: Optional[Dict] = None,
) -> Trainer:
    """Create a Lightning trainer configured for AGeDi."""
    if logger == "tensorboard":
        run_logger = TensorBoardLogger(save_dir=log_dir, name="")
    elif logger == "wandb":
        run_logger = WandbLogger(
            save_dir=log_dir,
            project=project,
            name=name,
        )
    else:
        raise ValueError(
            f'Unknown logger "{logger}". Valid options are: tensorboard, wandb'
        )

    callbacks = [
        LearningRateMonitor(logging_interval="epoch"),
        ModelCheckpoint(
            filename="best_model",
            monitor="val_loss",
            mode="min",
            save_top_k=1,
        ),
        ModelCheckpoint(
            filename="last_model",
            monitor=None,
            save_top_k=1,
            every_n_epochs=1,
        ),
    ]

    if repeat is not None:
        if repeat_epoch is None:
            raise ValueError("repeat_epoch must be set when repeat is not None")
        callbacks.append(TrainingPhase(repeat, [repeat_epoch for _ in range(repeat - 1)]))

    if hparams is not None:
        run_logger.log_hyperparams(hparams)

    return Trainer(
        accelerator="auto",
        devices=1,
        max_epochs=epochs,
        max_time={"hours": time_hours} if time_hours is not None else None,
        logger=run_logger,
        callbacks=callbacks,
        gradient_clip_val=gradient_clip_val,
        enable_progress_bar=progress_bar,
        log_every_n_steps=log_interval,
        reload_dataloaders_every_n_epochs=1 if repeat is not None else 0,
        inference_mode=False,
    )


def train(
    diffusion: Diffusion,
    dataset: Dataset,
    trainer: Optional[Trainer] = None,
    **trainer_kwargs,
) -> Trainer:
    """Train a diffusion model and return the trainer used."""
    current_trainer = trainer or create_trainer(**trainer_kwargs)
    current_trainer.fit(diffusion, dataset)
    return current_trainer


def sample(
    diffusion: Diffusion,
    *,
    n_samples: int,
    n_atoms: Optional[int] = None,
    atomic_numbers: Optional[List[int]] = None,
    cell: Optional[np.ndarray] = None,
    template: Optional[AtomsGraph] = None,
    confinement: Optional[Tuple[float, float]] = None,
    steps: int = 500,
    eps: float = 1e-3,
    batch_size: int = 64,
    force_field_guidance: float = 0.0,
    zeta: float = 3.0,
    force_threshold: float = 0.05,
    max_extra_steps: int = 100,
    property: Optional[Dict[str, float]] = None,
    progress_bar: bool = False,
    save_trajectory: bool = False,
    save_path: Optional[bool] = None,
    as_atoms: bool = True,
) -> Union[List[AtomsGraph], List[Atoms], List[List[AtomsGraph]], List[List[Atoms]]]:
    """Sample structures from a trained diffusion model."""
    if save_path is not None:
        warnings.warn(
            "'save_path' is deprecated; use 'save_trajectory' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        save_trajectory = save_path

    diffusion.eval()
    with torch.no_grad():
        sampled = diffusion.sample(
            N=n_samples,
            template=template,
            batch_size=batch_size,
            steps=steps,
            eps=eps,
            n_atoms=n_atoms,
            atomic_numbers=atomic_numbers,
            cell=cell,
            confinement=confinement,
            force_field_guidance=force_field_guidance,
            zeta=zeta,
            force_threshold=force_threshold,
            max_extra_steps=max_extra_steps,
            property=property,
            progress_bar=progress_bar,
            save_path=save_trajectory,
        )

    if not as_atoms:
        return sampled

    if save_trajectory:
        return [[graph.to_atoms() for graph in trajectory] for trajectory in sampled]
    return [graph.to_atoms() for graph in sampled]


def load_diffusion(
    path: Union[str, Path],
    checkpoint: Optional[Union[str, Path]] = None,
    device: Optional[Union[str, torch.device]] = None,
) -> Diffusion:
    """Load a trained diffusion model from an AGeDi log directory."""
    root_path = Path(path)
    if root_path.is_file():
        root_path = root_path.parent.parent

    params_path = root_path / "hparams.yaml"
    if not params_path.exists():
        raise FileNotFoundError(f"Could not find hparams file: {params_path}")

    with open(params_path, "r") as file:
        params = yaml.safe_load(file)

    current_device = torch.device(device) if device is not None else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    diffusion = create_diffusion(
        model=params["model"],
        cutoff=params["cutoff"],
        feature_size=params["feature_size"],
        n_blocks=params["n_blocks"],
        noisers=params["noisers"],
        style=params.get("style", "Default"),
        conditioning=params.get("conditioning", "none"),
        conditioning_type=params.get("conditioning_type", "scalar"),
        lr=params["lr"],
        lr_factor=params["lr_factor"],
        lr_patience=params["lr_patience"],
        device=current_device,
    )

    checkpoint_path = (
        Path(checkpoint)
        if checkpoint is not None
        else root_path / "checkpoints" / "last_model.ckpt"
    )
    checkpoint_data = torch.load(
        checkpoint_path,
        weights_only=True,
        map_location=current_device,
    )
    state_dict = checkpoint_data.get("state_dict", checkpoint_data)
    diffusion.load_state_dict(state_dict)
    diffusion.eval()
    return diffusion


def train_from_atoms(
    data: Sequence[Atoms],
    *,
    model: str = "PaiNN",
    cutoff: float = 6.0,
    feature_size: int = 64,
    n_blocks: int = 4,
    noisers: Sequence[str] = ("positions",),
    style: str = "Default",
    conditioning: str = "none",
    conditioning_type: str = "scalar",
    mask: str = "none",
    confinement: Optional[Tuple[float, float]] = None,
    batch_size: int = 64,
    repeat: Optional[int] = None,
    lr: float = 1e-4,
    lr_factor: float = 0.95,
    lr_patience: int = 100,
    guidance_weight: float = -1.0,
    trainer: Optional[Trainer] = None,
    **trainer_kwargs,
) -> Tuple[Diffusion, Dataset, Trainer]:
    """Build, train, and return an AGeDi model from ASE Atoms data."""
    diffusion = create_diffusion(
        model=model,
        cutoff=cutoff,
        feature_size=feature_size,
        n_blocks=n_blocks,
        noisers=noisers,
        style=style,
        conditioning=conditioning,
        conditioning_type=conditioning_type,
        confinement=confinement,
        lr=lr,
        lr_factor=lr_factor,
        lr_patience=lr_patience,
        guidance_weight=guidance_weight,
    )
    dataset = create_dataset(
        data,
        cutoff=cutoff,
        batch_size=batch_size,
        mask=mask,
        confinement=confinement,
        conditioning=conditioning,
        conditioning_type=conditioning_type,
        repeat=repeat,
    )

    hparams = {
        "model": model,
        "cutoff": cutoff,
        "feature_size": feature_size,
        "n_blocks": n_blocks,
        "noisers": list(noisers),
        "style": style,
        "conditioning": conditioning,
        "conditioning_type": conditioning_type,
        "mask": mask,
        "batch_size": batch_size,
        "lr": lr,
        "lr_factor": lr_factor,
        "lr_patience": lr_patience,
    } | data_info(list(data))
    if trainer is None:
        trainer_kwargs.setdefault("repeat", repeat)
        trainer_kwargs.setdefault("hparams", hparams)
    elif hasattr(trainer, "logger") and getattr(trainer, "logger") is not None:
        logger = getattr(trainer, "logger")
        if hasattr(logger, "log_hyperparams"):
            logger.log_hyperparams(hparams)

    fit_trainer = train(
        diffusion=diffusion,
        dataset=dataset,
        trainer=trainer,
        **trainer_kwargs,
    )
    return diffusion, dataset, fit_trainer

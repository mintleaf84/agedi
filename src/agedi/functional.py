from datetime import timedelta
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union
import time
import warnings

import numpy as np
import torch
import yaml
from ase import Atoms
from lightning import Trainer
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from agedi import Diffusion
from agedi.data import AtomsGraph, Dataset
from agedi.data.callbacks import (
    EpochProgressPrinter,
    GradNormLogger,
    HParamsMetricLogger,
    TrainingPhase,
)
from agedi.data.transforms import Repeat
from agedi.models import ScoreModel


# ---------------------------------------------------------------------------
# Private helpers (model/data construction utilities)
# ---------------------------------------------------------------------------

def _build_noisers(noisers: Sequence[Union[str, "Noiser"]], style: str, confined: bool = False) -> List["Noiser"]:
    """Build a list of Noiser objects from a sequence of noiser names or objects.

    Parameters
    ----------
    noisers : Sequence[Union[str, Noiser]]
        A sequence of noiser identifiers (``"positions"``, ``"types"``) or
        already-instantiated :class:`~agedi.diffusion.noisers.Noiser` objects.
    style : str
        The sampling style (e.g. ``"surface"``, ``"cluster"``).  Controls which
        distribution and prior are used for position noisers.
    confined : bool, optional
        If ``True``, use a confined (truncated-normal / confined-cell) prior for
        position noisers.  Ignored for non-position noisers.

    Returns
    -------
    List[Noiser]
        Instantiated noisers in the same order as *noisers*.
    """
    from agedi.diffusion.noisers import Noiser, PositionsNoiser, TypesNoiser
    from agedi.diffusion.distributions import (
        Normal,
        TruncatedNormal,
        UniformCell,
        UniformCellConfined,
        StandardNormal,
    )

    noiser_list = []
    for noiser in noisers:
        if isinstance(noiser, Noiser):
            noiser_list.append(noiser)
            continue
        match noiser:
            case "positions":
                if style == "surface":
                    if confined:
                        distribution = TruncatedNormal()
                        prior = UniformCellConfined()
                    else:
                        distribution = Normal()
                        prior = UniformCell()
                    noiser_list.append(PositionsNoiser(distribution=distribution, prior=prior))
                elif style == "cluster":
                    noiser_list.append(PositionsNoiser(prior=StandardNormal()))
                else:
                    noiser_list.append(PositionsNoiser())
            case "types":
                noiser_list.append(TypesNoiser())
            case _:
                raise ValueError(f"Unknown noiser '{noiser}'")

    return noiser_list


def _build_conditioning(condition: str, type: Optional[str] = None) -> List["Conditioning"]:
    """Build a list of conditioning modules.

    Always includes a :class:`~agedi.models.conditionings.TimeConditioning`.
    When *condition* is not ``"none"``, an additional property-conditioning
    module is appended.

    Parameters
    ----------
    condition : str
        Name of the property to condition on, or ``"none"`` for
        time-only conditioning.
    type : str, optional
        Type of the conditioning module: ``"scalar"`` or ``"integer"``.
        Required when *condition* is not ``"none"``.

    Returns
    -------
    List[Conditioning]
        The list of conditioning modules.
    """
    from agedi.models.conditionings import TimeConditioning

    conditioning = [TimeConditioning()]

    if condition != "none":
        from agedi.models.conditionings import ScalarConditioning, IntegerConditioning

        if type == "scalar":
            conditioning.append(ScalarConditioning(property=condition))
        elif type == "integer":
            conditioning.append(IntegerConditioning(property=condition))
        else:
            raise ValueError(f"Unknown conditioning type '{type}'")

    return conditioning


def _build_score_components(model: str, cutoff: float, heads: Sequence[str], feature_size: int, n_blocks: int, head_dim: int, n_rbf: int = 30) -> Tuple["Translator", "torch.nn.Module", List["Head"]]:
    """Instantiate the translator, representation, and score heads for a model.

    Parameters
    ----------
    model : str
        Name of the GNN backbone (currently only ``"PaiNN"`` is supported).
    cutoff : float
        Cutoff radius (Å) for the neighbour list.
    heads : Sequence[str]
        Names of score heads to build (``"positions"``, ``"types"``).
    feature_size : int
        Embedding/feature dimension for the backbone.
    n_blocks : int
        Number of interaction blocks in the backbone.
    head_dim : int
        Input dimension for each score head (typically
        ``feature_size + conditioning output dims``).
    n_rbf : int, optional
        Number of radial basis functions.  Default is 30.

    Returns
    -------
    Tuple[Translator, nn.Module, List[Head]]
        A 3-tuple of the translator, the representation backbone, and the list
        of score-head modules.
    """
    match model:
        case "PaiNN":
            import schnetpack as spk
            from agedi.models.schnetpack import (
                PositionsScore,
                TypesScore,
                SchNetPackTranslator,
            )

            translator = SchNetPackTranslator(
                input_modules=[spk.atomistic.PairwiseDistances()]
            )
            representation = spk.representation.PaiNN(
                n_atom_basis=feature_size,
                n_interactions=n_blocks,
                radial_basis=spk.nn.GaussianRBF(n_rbf=n_rbf, cutoff=cutoff),
                cutoff_fn=spk.nn.CosineCutoff(cutoff),
            )

            h = []
            for head in heads:
                match head:
                    case "positions":
                        h.append(PositionsScore(input_dim_scalar=head_dim))
                    case "types":
                        h.append(TypesScore(input_dim_scalar=head_dim))
                    case _:
                        raise ValueError(f"Unknown head '{head}'")

        case _:
            raise ValueError(f"Unknown model '{model}'")

    return translator, representation, h


def _extract_data_info(data: Sequence[Atoms]) -> Dict:
    """Extract summary information from a list of ASE Atoms objects.

    Parameters
    ----------
    data : Sequence[Atoms]
        List of ASE :class:`~ase.Atoms` objects to inspect.

    Returns
    -------
    dict
        A dictionary with the following keys:

        * ``"cell"`` – flattened 9-element list of the (shared) unit-cell
          matrix, or ``None`` if cells differ between structures.
        * ``"symbols"`` – list of unique chemical symbols present in the data.
        * ``"n_training_data"`` – total number of structures.
    """
    elements = set()
    out = {"cell": None}
    check_cell = True
    for d in data:
        elements.update(d.get_chemical_symbols())
        if check_cell:
            if d.cell is not None:
                out["cell"] = np.array(d.cell)
            else:
                if not np.all(out["cell"] == d.cell):
                    check_cell = False
    out["cell"] = out["cell"].flatten().tolist() if out["cell"] is not None else None
    out |= {
        "symbols": list(elements),
        "n_training_data": len(data),
    }
    return out


def _print_training_config(hparams: dict) -> None:
    """Print a Rich-formatted training configuration panel."""
    console = Console()
    table = Table(show_header=False, box=box.SIMPLE, padding=(0, 1))
    table.add_column("Key", style="bold cyan", min_width=22, no_wrap=True)
    table.add_column("Value", style="white")

    # Score Model
    table.add_row("[bold]Score Model[/bold]", "")
    table.add_row("  model", str(hparams.get("model", "")))
    table.add_row("  feature_size", str(hparams.get("feature_size", "")))
    table.add_row("  n_blocks", str(hparams.get("n_blocks", "")))
    table.add_row("  cutoff", f"{hparams.get('cutoff', '')} Å")

    # Diffusion
    table.add_row("", "")
    table.add_row("[bold]Diffusion[/bold]", "")
    noisers = hparams.get("noisers", [])
    table.add_row("  noisers", ", ".join(noisers) if noisers else "")
    table.add_row("  style", str(hparams.get("style", "")))
    confinement = hparams.get("confinement")
    if confinement:
        lo, hi = confinement
        table.add_row("  confinement", f"{lo} – {hi} Å")
    conditioning = hparams.get("conditioning", "none")
    if conditioning != "none":
        table.add_row(
            "  conditioning",
            f"{conditioning} ({hparams.get('conditioning_type', '')})",
        )

    # Dataset
    table.add_row("", "")
    table.add_row("[bold]Dataset[/bold]", "")
    if hparams.get("data_path"):
        table.add_row("  data", str(hparams["data_path"]))
    table.add_row("  n_train", str(hparams.get("n_train", "")))
    table.add_row("  n_val", str(hparams.get("n_val", "")))
    table.add_row("  batch_size", str(hparams.get("batch_size", "")))
    mask = hparams.get("mask", "none")
    if mask and mask != "none":
        table.add_row("  mask", str(mask))
    repeat = hparams.get("repeat")
    if repeat is not None:
        table.add_row("  repeat", str(repeat))
        table.add_row("  repeat_epoch", str(hparams.get("repeat_epoch", "")))

    # Optimizer
    table.add_row("", "")
    table.add_row("[bold]Optimizer[/bold]", "")
    table.add_row("  lr", str(hparams.get("lr", "")))
    table.add_row("  lr_patience", str(hparams.get("lr_patience", "")))
    table.add_row("  lr_factor", str(hparams.get("lr_factor", "")))
    table.add_row("  weight_decay", str(hparams.get("weight_decay", 0.0)))
    table.add_row("  gradient_clip_val", str(hparams.get("gradient_clip_val", "")))

    # Parameters
    n_parameters = hparams.get("n_parameters")
    if n_parameters is not None:
        table.add_row("", "")
        table.add_row("[bold]Model[/bold]", "")
        table.add_row("  parameters", f"{n_parameters:,}")

    console.print(
        Panel(table, title="[bold]AGeDi Training Configuration[/bold]", border_style="blue")
    )


def _print_sampling_config(
    n_samples: int,
    steps: int,
    eps: float,
    batch_size: int,
    formula=None,
    n_atoms=None,
    template=None,
    cell=None,
    confinement=None,
    property=None,
    force_field_guidance: float = 0.0,
) -> None:
    """Print a Rich-formatted sampling configuration panel."""
    console = Console()
    table = Table(show_header=False, box=box.SIMPLE, padding=(0, 1))
    table.add_column("Key", style="bold cyan", min_width=14, no_wrap=True)
    table.add_column("Value", style="white")

    table.add_row("  n_samples", str(n_samples))
    table.add_row("  steps", str(steps))
    table.add_row("  eps", str(eps))
    table.add_row("  batch_size", str(batch_size))
    if formula is not None:
        table.add_row("  formula", str(formula))
    elif n_atoms is not None:
        table.add_row("  n_atoms", str(n_atoms))
    if template is not None:
        table.add_row("  template", "provided")
    if cell is not None:
        table.add_row("  cell", "provided")
    if confinement is not None:
        table.add_row("  confinement", f"{confinement[0]} – {confinement[1]} Å")
    if property is not None:
        for k, v in property.items():
            table.add_row(f"  {k}", str(v))
    if force_field_guidance > 0.0:
        table.add_row("  ff_guidance", str(force_field_guidance))

    console.print(
        Panel(table, title="[bold]AGeDi Sampling Configuration[/bold]", border_style="blue")
    )


def create_diffusion(
    model: str = "PaiNN",
    cutoff: float = 6.0,
    feature_size: int = 64,
    n_blocks: int = 4,
    n_rbf: int = 30,
    noisers: Sequence[Union[str, "Noiser"]] = ("positions",),
    style: str = "Default",
    conditioning: str = "none",
    conditioning_type: str = "scalar",
    confinement: Optional[Tuple[float, float]] = None,
    lr: float = 1e-4,
    lr_factor: float = 0.95,
    lr_patience: int = 100,
    weight_decay: float = 0.0,
    eps: float = 1e-5,
    guidance_weight: float = -1.0,
    device: Optional[Union[str, torch.device]] = None,
) -> Diffusion:
    """Create a diffusion model for script-based training and sampling."""
    torch_device = torch.device(device) if device is not None else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    conditioning_modules = _build_conditioning(conditioning, type=conditioning_type)
    head_dim = feature_size + sum(module.output_dim for module in conditioning_modules)

    translator, representation, heads = _build_score_components(
        model,
        cutoff,
        noisers,
        feature_size,
        n_blocks,
        head_dim=head_dim,
        n_rbf=n_rbf,
    )

    confined = confinement is not None and "positions" in noisers
    noiser_modules = _build_noisers(noisers, style, confined=confined)

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
        optim_config={"lr": lr, "weight_decay": weight_decay},
        scheduler_config={"factor": lr_factor, "patience": lr_patience},
        eps=eps,
    ).to(torch_device)


def create_dataset(
    data: Sequence[Atoms],
    cutoff: float = 6.0,
    batch_size: int = 64,
    train_split: Union[float, int] = 0.9,
    val_split: Union[float, int] = 0.1,
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

        property_kinds = {"mask": "node"}
        if confinement is not None:
            property_kinds["confinement"] = "none"
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
        n_train=train_split,
        n_val=val_split,
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
    max_time: Optional[Union[int, Dict, timedelta]] = 24,
    accelerator: str = "auto",
    devices: int = 1,
    logger: str = "tensorboard",
    log_dir: str = "logs",
    project: str = "agedi",
    name: str = "agedi",
    log_interval: int = 10,
    gradient_clip_val: float = 10.0,
    progress_bar: bool = False,
    print_epoch_interval: int = 10,
    log_grad_norm: bool = True,
    repeat: Optional[int] = None,
    repeat_epoch: Optional[int] = None,
    hparams: Optional[Dict] = None,
) -> Trainer:
    """Create a Lightning trainer configured for AGeDi.

    Parameters
    ----------
    epochs:
        Maximum number of training epochs (``-1`` = unlimited).
    max_time:
        Wall-clock time limit for training.  Accepts:

        * ``int``   – number of *hours* (e.g. ``24`` ≡ 24 hours).
        * ``dict``  – Lightning-style mapping, e.g.
          ``{"days": 0, "hours": 12, "minutes": 30, "seconds": 0}``.
        * :class:`datetime.timedelta` – a Python timedelta object.
        * ``None``  – no time limit.
    accelerator:
        Hardware accelerator to use (e.g. ``"auto"``, ``"gpu"``, ``"cpu"``).
    devices:
        Number of devices to train on.
    print_epoch_interval:
        Print a one-line training summary to stdout every this many epochs.
        Set to ``0`` to disable (default: 10).
    log_grad_norm:
        Whether to log the total gradient norm during training (default: ``True``).
        Disable for large models where the per-step overhead is undesirable.
    """
    if max_time is None:
        _max_time = None
    elif isinstance(max_time, int):
        _max_time = {"hours": max_time}
    elif isinstance(max_time, (dict, timedelta)):
        _max_time = max_time
    else:
        raise TypeError(
            f"max_time must be None, int (hours), dict, or timedelta; "
            f"got {type(max_time).__name__}"
        )
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

    if log_grad_norm:
        callbacks.append(GradNormLogger(log_every_n_steps=log_interval))

    if print_epoch_interval > 0:
        callbacks.append(EpochProgressPrinter(print_epoch_interval))

    if repeat is not None:
        if repeat_epoch is None:
            raise ValueError("repeat_epoch must be set when repeat is not None")
        callbacks.append(TrainingPhase(repeat, [repeat_epoch for _ in range(repeat - 1)]))

    if hparams is not None:
        callbacks.append(HParamsMetricLogger(hparams))

    return Trainer(
        accelerator=accelerator,
        devices=devices,
        max_epochs=epochs,
        max_time=_max_time,
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
    formula: Optional[str] = None,
    positions: Optional[np.ndarray] = None,
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
    """Sample structures from a trained diffusion model.

    Parameters
    ----------
    diffusion:
        A trained :class:`~agedi.Diffusion` model.
    n_samples:
        Number of structures to generate.
    n_atoms:
        Number of atoms per structure. Automatically determined from
        ``formula`` if provided, or from the length of ``atomic_numbers``
        when ``n_atoms`` is not explicitly given.
    atomic_numbers:
        Atomic numbers of the generated atoms.  Not required when the model
        has a types-noiser or when ``formula`` is provided.
    formula:
        Chemical formula (e.g. ``"H2O"``).  Used to derive ``n_atoms`` and
        ``atomic_numbers`` when they are not provided explicitly.
    positions:
        Fixed positions of the atoms (shape ``(n_atoms, 3)``).  Required
        when no positions-noiser is configured (type-only diffusion).
        Positions will not be modified during sampling.
    cell:
        Unit-cell matrix (3×3 array or flat length-9 array).  Not required
        when ``template`` is provided (the template's cell is used instead).
    template:
        Template :class:`~agedi.AtomsGraph`.  When given, ``cell`` and
        ``pbc`` are taken from the template unless explicitly provided.
    """
    if save_path is not None:
        warnings.warn(
            "'save_path' is deprecated; use 'save_trajectory' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        save_trajectory = save_path

    _print_sampling_config(
        n_samples=n_samples,
        steps=steps,
        eps=eps,
        batch_size=batch_size,
        formula=formula,
        n_atoms=n_atoms,
        template=template,
        cell=cell,
        confinement=confinement,
        property=property,
        force_field_guidance=force_field_guidance,
    )

    _start = time.monotonic()

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
            formula=formula,
            positions=positions,
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

    elapsed = time.monotonic() - _start
    n_generated = len(sampled)
    Console().print(f"[green]✓[/green] Generated {n_generated} structure(s) in {elapsed:.1f}s")

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
        n_rbf=params.get("n_rbf", 30),
        noisers=params["noisers"],
        style=params.get("style", "Default"),
        conditioning=params.get("conditioning", "none"),
        conditioning_type=params.get("conditioning_type", "scalar"),
        lr=params["lr"],
        lr_factor=params["lr_factor"],
        lr_patience=params["lr_patience"],
        weight_decay=params.get("weight_decay", 0.0),
        eps=params.get("eps", 1e-5),
        guidance_weight=params.get("guidance_weight", -1.0),
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
    n_rbf: int = 30,
    noisers: Sequence[str] = ("positions",),
    style: str = "Default",
    conditioning: str = "none",
    conditioning_type: str = "scalar",
    mask: str = "none",
    confinement: Optional[Tuple[float, float]] = None,
    batch_size: int = 64,
    train_split: Union[float, int] = 0.9,
    val_split: Union[float, int] = 0.1,
    repeat: Optional[int] = None,
    lr: float = 1e-4,
    lr_factor: float = 0.95,
    lr_patience: int = 100,
    weight_decay: float = 0.0,
    eps: float = 1e-5,
    guidance_weight: float = -1.0,
    data_path: Optional[str] = None,
    trainer: Optional[Trainer] = None,
    **trainer_kwargs,
) -> Tuple[Diffusion, Dataset, Trainer]:
    """Build, train, and return an AGeDi model from ASE Atoms data."""
    diffusion = create_diffusion(
        model=model,
        cutoff=cutoff,
        feature_size=feature_size,
        n_blocks=n_blocks,
        n_rbf=n_rbf,
        noisers=noisers,
        style=style,
        conditioning=conditioning,
        conditioning_type=conditioning_type,
        confinement=confinement,
        lr=lr,
        lr_factor=lr_factor,
        lr_patience=lr_patience,
        weight_decay=weight_decay,
        eps=eps,
        guidance_weight=guidance_weight,
    )
    dataset = create_dataset(
        data,
        cutoff=cutoff,
        batch_size=batch_size,
        train_split=train_split,
        val_split=val_split,
        mask=mask,
        confinement=confinement,
        conditioning=conditioning,
        conditioning_type=conditioning_type,
        repeat=repeat,
    )

    n_parameters = sum(
        p.numel() for p in diffusion.score_model.parameters() if p.requires_grad
    )

    hparams = {
        "model": model,
        "cutoff": cutoff,
        "feature_size": feature_size,
        "n_blocks": n_blocks,
        "n_rbf": n_rbf,
        "noisers": list(noisers),
        "style": style,
        "conditioning": conditioning,
        "conditioning_type": conditioning_type,
        "mask": mask,
        "confinement": list(confinement) if confinement is not None else None,
        "batch_size": batch_size,
        "train_split": train_split,
        "val_split": val_split,
        "n_train": len(dataset.train_idx),
        "n_val": len(dataset.val_idx),
        "lr": lr,
        "lr_factor": lr_factor,
        "lr_patience": lr_patience,
        "weight_decay": weight_decay,
        "eps": eps,
        "guidance_weight": guidance_weight,
        "gradient_clip_val": (
            trainer.gradient_clip_val
            if trainer is not None and hasattr(trainer, "gradient_clip_val")
            else trainer_kwargs.get("gradient_clip_val", 10.0)
        ),
        "n_parameters": n_parameters,
        "repeat": repeat,
        "repeat_epoch": trainer_kwargs.get("repeat_epoch"),
        "data_path": data_path,
    } | _extract_data_info(list(data))

    _print_training_config(hparams)

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

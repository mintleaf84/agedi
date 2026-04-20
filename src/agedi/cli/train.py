import rich_click as click
from rich.console import Console
from pathlib import Path

from ase.io import read

from agedi.functional import train_from_atoms

click.rich_click.OPTION_GROUPS.update(
    {
        "agedi train": [
            {
                "name": "Score Model Options",
                "options": ["--model", "--cutoff", "--feature_size", "--n_blocks"],
            },
            {
                "name": "Diffusion Model Options",
                "options": ["--noisers", "--sde", "--conditioning", "--conditioning_type"],
            },
            {
                "name": "Training Options",
                "options": [
                    "--epochs",
                    "--max_time",
                    "--lr",
                    "--batch_size",
                    "--lr_patience",
                    "--lr_factor",
                    "--progress_bar",
                    "--gradient_clip_val",
                ],
            },
            {
                "name": "Data Options",
                "options": [
                    "--mask",
                    "--confinement",
                    "--repeat",
                    "--repeat_epoch",
                ],
            },
            {
                "name": "Logging Options",
                "options": [
                    "--logger",
                    "--log_dir",
                    "--project",
                    "--name",
                    "--log_interval",
                ],
            },
        ]
    }
)


@click.command()
@click.argument("data", type=click.Path(exists=True))
@click.option(
    "--sde",
    type=click.Choice(["ve", "vp"]),
    default="ve",
    show_default=True,
    help="SDE to use for position noisers",
)
@click.option(
    "--model",
    "-m",
    type=click.Choice(["PaiNN"]),
    default="PaiNN",
    show_default=True,
    help="Representation to use for the model",
)
@click.option(
    "--cutoff",
    "-r",
    type=float,
    default=6.0,
    show_default=True,
    help="Cutoff for the representation in Å",
)
@click.option(
    "--feature_size",
    "-f",
    type=int,
    default=64,
    show_default=True,
    help="Feature size for the representation",
)
@click.option(
    "--n_blocks",
    type=int,
    default=4,
    show_default=True,
    help="Number of blocks for the representation",
)
@click.option(
    "--noisers",
    "-n",
    type=click.Choice(
        [
            "Positions",
            "CellPositions",
            "ConfinedCellPositions",
            "Types",
            # snake_case aliases kept for backwards compatibility
            "positions",
            "cell_positions",
            "confined_cell_positions",
            "types",
        ]
    ),
    default=["CellPositions"],
    multiple=True,
    show_default=True,
    help="Type of noisers to use",
)
@click.option(
    "--conditioning",
    "-c",
    type=str,
    default="none",
    help="Property to condition on",
    hidden=True,
)
@click.option(
    "--conditioning_type",
    type=click.Choice(["scalar", "integer", "node"]),
    default="scalar",
    show_default=True,
    help="Type of conditioning to use",
)
@click.option(
    "--epochs",
    "-e",
    type=int,
    default=-1,
    show_default=True,
    help="Number of epochs to train for",
)
@click.option(
    "--max_time",
    "-t",
    type=int,
    default=24,
    show_default=True,
    help="Maximum training time in hours",
)
@click.option("--lr", type=float, default=1e-4, show_default=True, help="Learning rate")
@click.option(
    "--batch_size", "-b", type=int, default=64, show_default=True, help="Batch size"
)
@click.option(
    "--lr_patience",
    type=int,
    default=100,
    show_default=True,
    help="Number of epochs to wait before reducing the learning rate",
)
@click.option(
    "--lr_factor",
    type=float,
    default=0.95,
    show_default=True,
    help="Factor to reduce the learning rate by",
)
@click.option(
    "--gradient_clip_val",
    type=float,
    default=10.0,
    show_default=True,
    help="Gradient clipping value",
)
@click.option(
    "--mask",
    type=click.Choice(["MaskFixed", "none"]),
    default="none",
    help="Masking to use for the data",
)
@click.option(
    "--confinement",
    nargs=2,
    type=float,
    default=None,
    help="Z-confinement to use for the data. Give min and max value",
)
@click.option(
    "--repeat",
    type=int,
    default=None,
    help="Maximal number of times to repeat the data for training",
)
@click.option(
    "--repeat_epoch",
    type=int,
    default=None,
    help="How many epochs between repeats",
)
@click.option(
    "--logger",
    type=click.Choice(["tensorboard", "wandb"]),
    default="tensorboard",
    help="Logger to use",
)
@click.option(
    "--log_dir",
    type=click.Path(),
    default="logs",
    show_default=True,
    help="Directory to save logs to",
)
@click.option(
    "--project",
    type=str,
    default="agedi",
    show_default=True,
    help="Project name for wandb",
)
@click.option(
    "--name",
    type=str,
    default="agedi",
    show_default=True,
    help="Display name for wandb",
)
@click.option(
    "--log_interval", type=int, default=10, show_default=True, help="Interval to log at"
)
@click.option("--progress_bar", is_flag=True, help="Show progress bar")
def train(**params) -> None:
    """Train an AGeDi diffusion model from the command line.

    Reads the dataset from the file specified by DATA, constructs a
    diffusion model and trainer from the remaining CLI options, and starts
    training via :func:`~agedi.functional.train_from_atoms`.
    """
    data_path = str(Path(params["data"]).resolve())
    data = read(data_path, ":")

    train_from_atoms(
        data,
        model=params["model"],
        cutoff=params["cutoff"],
        feature_size=params["feature_size"],
        n_blocks=params["n_blocks"],
        noisers=params["noisers"],
        sde=params["sde"],
        conditioning=params["conditioning"],
        conditioning_type=params["conditioning_type"],
        mask=params["mask"],
        confinement=params["confinement"],
        batch_size=params["batch_size"],
        repeat=params["repeat"],
        lr=params["lr"],
        lr_factor=params["lr_factor"],
        lr_patience=params["lr_patience"],
        data_path=data_path,
        # trainer kwargs forwarded via **trainer_kwargs in train_from_atoms
        epochs=params["epochs"],
        max_time=params["max_time"],
        logger=params["logger"],
        log_dir=params["log_dir"],
        project=params["project"],
        name=params["name"],
        log_interval=params["log_interval"],
        gradient_clip_val=params["gradient_clip_val"],
        progress_bar=params["progress_bar"],
        repeat_epoch=params["repeat_epoch"],
    )

    console = Console()
    console.print(f"\n[green]✓ Training complete.[/green]")
    console.print(f"To sample from the model run:")
    console.print(f"  [bold]agedi sample {params['log_dir']} -f ...[/bold]")

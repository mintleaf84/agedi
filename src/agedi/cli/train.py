import rich_click as click
from rich import box
from rich.console import Console
from rich.table import Table
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
                "options": ["--noisers", "--conditioning"],
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
                    "--style",
                    "--mask",
                    "--confinement",
                    "--repeat",
                    "--repeat_epoch",
                    "--conditioning_type",
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


def _print_training_config(params: dict, n_data: int) -> None:
    """Print a structured summary of training options using Rich."""
    console = Console()
    table = Table(show_header=False, box=box.SIMPLE, padding=(0, 1))
    table.add_column("Key", style="bold cyan", min_width=22, no_wrap=True)
    table.add_column("Value", style="white")

    # Score Model
    table.add_row("[bold]Score Model[/bold]", "")
    table.add_row("  model", str(params["model"]))
    table.add_row("  feature_size", str(params["feature_size"]))
    table.add_row("  n_blocks", str(params["n_blocks"]))
    table.add_row("  cutoff", f"{params['cutoff']} Å")

    # Diffusion
    table.add_row("", "")
    table.add_row("[bold]Diffusion[/bold]", "")
    table.add_row("  noisers", ", ".join(params["noisers"]))
    table.add_row("  style", str(params["style"]))
    if params["confinement"]:
        lo, hi = params["confinement"]
        table.add_row("  confinement", f"{lo} – {hi} Å")
    if params["conditioning"] != "none":
        table.add_row(
            "  conditioning",
            f"{params['conditioning']} ({params['conditioning_type']})",
        )

    # Dataset
    table.add_row("", "")
    table.add_row("[bold]Dataset[/bold]", "")
    table.add_row("  data", str(params["data"]))
    table.add_row("  samples", str(n_data))
    table.add_row("  batch_size", str(params["batch_size"]))
    if params["mask"] != "none":
        table.add_row("  mask", str(params["mask"]))
    if params["repeat"] is not None:
        table.add_row("  repeat", str(params["repeat"]))
        table.add_row("  repeat_epoch", str(params["repeat_epoch"]))

    # Optimizer
    table.add_row("", "")
    table.add_row("[bold]Optimizer[/bold]", "")
    table.add_row("  lr", str(params["lr"]))
    table.add_row("  lr_patience", str(params["lr_patience"]))
    table.add_row("  lr_factor", str(params["lr_factor"]))
    table.add_row("  weight_decay", str(params.get("weight_decay", 0.0)))
    table.add_row("  gradient_clip_val", str(params["gradient_clip_val"]))

    # Training schedule
    table.add_row("", "")
    table.add_row("[bold]Training[/bold]", "")
    epochs_str = str(params["epochs"]) if params["epochs"] > 0 else "unlimited"
    table.add_row("  epochs", epochs_str)
    table.add_row("  max_time", f"{params['max_time']}h")

    # Logging
    table.add_row("", "")
    table.add_row("[bold]Logging[/bold]", "")
    table.add_row("  logger", str(params["logger"]))
    table.add_row("  log_dir", str(params["log_dir"]))
    if params["logger"] == "wandb":
        table.add_row("  project", str(params["project"]))
        table.add_row("  name", str(params["name"]))

    from rich.panel import Panel

    console.print(
        Panel(table, title="[bold]AGeDi Training Configuration[/bold]", border_style="blue")
    )


@click.command()
@click.argument("data", type=click.Path(exists=True))
@click.option(
    "--style",
    "-s",
    type=click.Choice(["Default", "surface", "cluster"]),
    default="Default",
    show_default=True,
    help="Style of diffusion model depending on data type",
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
    type=click.Choice(["positions", "types", "cell"]),
    default=["positions"],
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

    Reads the dataset from the file specified by ``--data``, constructs a
    diffusion model and trainer from the remaining CLI options, and starts
    training via :func:`~agedi.functional.train_from_atoms`.

    Parameters
    ----------
    **params
        CLI options forwarded from Click (``data``, ``model``, ``cutoff``,
        ``feature_size``, ``n_blocks``, ``noisers``, ``style``,
        ``conditioning``, ``conditioning_type``, ``mask``, ``confinement``,
        ``batch_size``, ``repeat``, ``lr``, ``lr_factor``, ``lr_patience``,
        ``epochs``, ``max_time``, ``logger``, ``log_dir``, ``project``,
        ``name``, ``log_interval``, ``gradient_clip_val``, ``progress_bar``,
        ``repeat_epoch``).

    Returns
    -------
    None
    """
    data_path = str(Path(params["data"]).resolve())
    data = read(data_path, ":")

    _print_training_config(params, len(data))

    train_from_atoms(
        data,
        model=params["model"],
        cutoff=params["cutoff"],
        feature_size=params["feature_size"],
        n_blocks=params["n_blocks"],
        noisers=params["noisers"],
        style=params["style"],
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

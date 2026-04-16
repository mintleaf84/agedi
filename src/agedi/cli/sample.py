import yaml
from rich.console import Console
import rich_click as click
from pathlib import Path

import numpy as np
import torch
from ase.io import read, write

from agedi.functional import load_diffusion, sample as functional_sample
from agedi.data import AtomsGraph


click.rich_click.OPTION_GROUPS.update(
    {
        "agedi sample": [
            {"name": "Model Options", "options": ["path", "--style"]},
            {
                "name": "Structure Options",
                "options": [
                    "--n_samples",
                    "--n_atoms",
                    "--formula",
                    "--cell",
                    "--template_path",
                    "--confinement",
                ],
            },
            {
                "name": "Sampling Hyperparameters",
                "options": [
                    "--output",
                    "--name",
                    "--steps",
                    "--seed",
                    "--eps",
                    "--batch_size",
                ],
            },
        ]
    }
)


@click.command()
@click.argument("path", type=click.Path(exists=True))
@click.option("--n_samples", "-n", type=int, show_default=True, default=12)
@click.option("--seed", "-s", type=int, show_default=True, default=42)
@click.option("--steps", type=int, show_default=True, default=500)
@click.option("--eps", type=float, show_default=True, default=0.005)
@click.option("--batch_size", "-b", show_default=True, type=int, default=64)
@click.option("--output", "-o", type=click.Path(), show_default=True, default=".")
@click.option("--name", type=str, show_default=True, default="sampled")
@click.option("--n_atoms", "-a", type=int)
@click.option("--formula", "-f", type=str)
@click.option("--cell", "-c", nargs=9, type=float)
@click.option("--template_path", "-t", type=click.Path(exists=True))
@click.option(
    "--confinement",
    nargs=2,
    type=float,
    default=None,
    help="Z-confinement to use for the data. Give min and max value",
)
@click.option("--progress_bar", is_flag=True, help="Show progress bar")
@click.option(
    "--save_trajectory", is_flag=True, help="Save entire diffusion trajectory"
)
@click.option(
    "--style",
    type=click.Choice(["Default", "surface", "cluster"]),
    default=None,
    show_default=False,
    help="Override the diffusion style (default: read from model hparams)",
)
def sample(path: str, **kwargs) -> None:
    """Sample structures from a trained AGeDi diffusion model.

    Loads the model from *path*, generates structures according to the provided
    options, and writes the output to the specified directory.

    Parameters
    ----------
    path : str
        Path to the AGeDi log / model directory containing the checkpoint.
    **kwargs
        CLI options forwarded from Click (``n_samples``, ``steps``, ``eps``,
        ``batch_size``, ``output``, ``name``, ``n_atoms``, ``formula``,
        ``cell``, ``template_path``, ``confinement``, ``progress_bar``,
        ``save_trajectory``, ``seed``, ``style``).

    Returns
    -------
    None
    """
    console = Console()
    console.print(f"Loading model from: [cyan]{path}[/cyan]")

    diffusion = load_diffusion(path, style=kwargs.get("style"))

    sample_kwargs = dict(
        n_samples=kwargs["n_samples"],
        n_atoms=kwargs["n_atoms"],
        steps=kwargs["steps"],
        eps=kwargs["eps"],
        batch_size=kwargs["batch_size"],
        progress_bar=kwargs["progress_bar"],
        save_trajectory=kwargs["save_trajectory"],
        confinement=kwargs["confinement"],
        as_atoms=True,
    )

    cell = None
    if kwargs["template_path"]:
        t = read(kwargs["template_path"])
        template = AtomsGraph.from_atoms(t, initialize_mask=False)
        if kwargs["confinement"]:
            template.confinement = torch.tensor(kwargs["confinement"]).reshape(1, 2)
        sample_kwargs["template"] = template

    if kwargs["formula"]:
        sample_kwargs["formula"] = kwargs["formula"]

    if cell is None and "template" not in sample_kwargs:
        # Fall back to cell stored in hparams
        root_path = Path(path)
        if root_path.is_file():
            root_path = root_path.parent.parent
        with open(root_path / "hparams.yaml", "r") as f:
            params = yaml.safe_load(f)
        cell = np.array(params["cell"]).reshape(3, 3)

    if cell is not None:
        sample_kwargs["cell"] = cell

    structures = functional_sample(diffusion, **sample_kwargs)

    output_dir = Path(kwargs["output"])
    output_dir.mkdir(parents=True, exist_ok=True)
    name = kwargs["name"]

    if kwargs["save_trajectory"]:
        for i, trajectory in enumerate(structures):
            write(output_dir / f"{name}_{i}.traj", trajectory)
        out_desc = f"{len(structures)} trajectory file(s) in {output_dir}/"
    else:
        out_path = output_dir / f"{name}.traj"
        write(out_path, structures)
        out_desc = str(out_path)

    console.print(f"Saved to: [cyan]{out_desc}[/cyan]")

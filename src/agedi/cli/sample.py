import yaml
from rich import print
import rich_click as click
from pathlib import Path

import torch
import numpy as np
from ase import Atoms
from ase.io import read, write

from agedi.cli.train import get_package, get_conditioning, get_noisers

from agedi import Diffusion
from agedi.models import ScoreModel
from agedi.data import Dataset, AtomsGraph


click.rich_click.OPTION_GROUPS.update(
    {
        "agedi sample": [
            {"name": "Model Options", "options": ["path"]},
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
    "--save_path", is_flag=True, help="Save entire diffusion trajectory pathway"
)
def sample(path, **kwargs):
    click.echo(f"Loading model from: {path}")
    # read yaml file
    with open(Path(path) / "hparams.yaml", "r") as file:
        params = yaml.safe_load(file)

    sample_kwargs = {
        "N": kwargs["n_samples"],
        "n_atoms": kwargs["n_atoms"],
        "steps": kwargs["steps"],
        "eps": kwargs["eps"],
        "batch_size": kwargs["batch_size"],
        "progress_bar": kwargs["progress_bar"],
        "save_path": kwargs["save_path"],
        "confinement": kwargs["confinement"],
    }

    if kwargs["template_path"]:
        t = read(kwargs["template_path"])
        sample_kwargs["cell"] = np.array(t.cell)
        template = AtomsGraph.from_atoms(t, initialize_mask=False)
        if kwargs["confinement"]:
            template.confinement = torch.tensor(kwargs["confinement"]).reshape(1, 2)
        sample_kwargs["template"] = template

    if kwargs["n_atoms"]:
        sample_kwargs["n_atoms"] = kwargs["n_atoms"]

    if kwargs["formula"]:
        a = Atoms(kwargs["formula"])
        sample_kwargs["n_atoms"] = len(a)
        sample_kwargs["atomic_numbers"] = a.get_atomic_numbers()

    if sample_kwargs.get("cell") is None:
        sample_kwargs["cell"] = np.array(params["cell"]).reshape(3, 3)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Model
    conditionings = get_conditioning(params["conditioning"], type=params["conditioning_type"])
    head_dim = params["feature_size"] + sum([c.output_dim for c in conditionings])

    translator, representation, heads = get_package(
        params["model"],
        params["cutoff"],
        params["noisers"],
        params["feature_size"],
        params["n_blocks"],
        head_dim=head_dim,
    )
    if kwargs["confinement"] is not None and "positions" in params["noisers"]:
        confined = True
    else:
        confined = False

    style = params.get("style", "Default")
    noisers = get_noisers(params["noisers"], style=style, confined=confined)

    score_model = ScoreModel(
        translator=translator,
        representation=representation,
        conditionings=conditionings,
        heads=[h.to(device) for h in heads],
    )

    diffusion = Diffusion(
        score_model,
        noisers,
        optim_config={"lr": params["lr"]},
        scheduler_config={
            "factor": params["lr_factor"],
            "patience": params["lr_patience"],
        },
    ).to(device)

    diffusion.load_state_dict(
        torch.load(
            Path(path) / "checkpoints/last_model.ckpt",
            weights_only=True,
            map_location=device,
        )["state_dict"]
    )

    diffusion.eval()

    with torch.no_grad():
        graph_list = diffusion.sample(**sample_kwargs)

    Path(kwargs["output"]).mkdir(parents=True, exist_ok=True)
    name = kwargs["name"]

    if sample_kwargs.get("save_path", False):
        for i, graph_list_i in enumerate(graph_list):
            atoms_list = [g.to_atoms() for g in graph_list_i]
            write(Path(kwargs["output"]) / f"{name}_{i}.traj", atoms_list)

    else:
        atoms_list = [g.to_atoms() for g in graph_list]
        write(Path(kwargs["output"]) / f"{name}.traj", atoms_list)

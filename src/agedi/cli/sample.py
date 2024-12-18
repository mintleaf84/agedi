import yaml
from rich import print
import rich_click as click
from pathlib import Path

import torch
import numpy as np
from ase import Atoms
from ase.io import read, write

from agedi.cli.train import get_package, get_conditioning, get_noisers
from agedi.diffusion import Diffusion
from agedi.models import ScoreModel
from agedi.data import Dataset


click.rich_click.OPTION_GROUPS.update({
    "agedi sample": [
        {"name": "Model Options", "options": ['path']},
        {"name": "Structure Options", "options": ['--n_samples', '--n_atoms', '--formula', '--cell', '--template_path']},
        {"name": "Sampling Hyperparameters", "options": ['--output', '--name', '--steps', '--seed', '--eps', '--batch_size']},
    ]
})

@click.command()
@click.argument("path", type=click.Path(exists=True))
@click.option("--n_samples", '-n', type=int, show_default=True, default=16)
@click.option("--seed", '-s', type=int, show_default=True, default=42)
@click.option("--steps", type=int, show_default=True, default=500)
@click.option("--eps", type=float, show_default=True, default=0.005)
@click.option("--batch_size", '-b', show_default=True, type=int, default=64)
@click.option("--output", '-o', type=click.Path(), show_default=True, default=".")
@click.option("--name", type=str, show_default=True, default="sampled")
@click.option("--n_atoms", '-a', type=int)
@click.option("--formula", '-f', type=str)
@click.option("--cell", '-c', nargs=9, type=float)
@click.option("--template_path", '-t', type=click.Path(exists=True))
@click.option('--progress_bar', is_flag=True, help='Show progress bar')
@click.option('--save_path', is_flag=True, help='Show progress bar')
def sample(path, **kwargs):
    click.echo(f"Loading model from: {path}")
    # read yaml file
    with open(Path(path) / 'hparams.yaml', "r") as file:
        params = yaml.safe_load(file)

    sample_kwargs = {
        "N": kwargs["n_samples"],
        "n_atoms": kwargs["n_atoms"],
        "steps": kwargs["steps"],
        "eps": kwargs["eps"],
        "batch_size": kwargs["batch_size"],
        "progress_bar": kwargs["progress_bar"],
        "save_path": kwargs["save_path"],
    }
        
    if kwargs["template_path"]:
        t = read(kwargs["template_path"])
        sample_kwargs["template"] = t
        sample_kwargs["n_atoms"] = len(t)
        sample_kwargs["atomic_numbers"] = t.get_atomic_numbers()
        sample_kwargs["cell"] = np.array(t.cell)
                      
    if kwargs['n_atoms']:
        sample_kwargs["n_atoms"] = kwargs["n_atoms"]

    if kwargs['formula']:
        a = Atoms(kwargs['formula'])
        sample_kwargs["n_atoms"] = len(a)
        sample_kwargs["atomic_numbers"] = a.get_atomic_numbers()

    if sample_kwargs.get('cell') is None:
        sample_kwargs["cell"] = np.array(params["cell"]).reshape(3, 3)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Model
    translator, representation, heads = get_package(params['model'], params['cutoff'], params['noisers'], params['feature_size'], params['n_blocks'])
    conditionings = get_conditioning(params['conditioning'])
    noisers = get_noisers(params['noisers'], params['noiser_sdes'])  

    score_model = ScoreModel(
        translator=translator,
        representation=representation,
        conditionings=conditionings,
        heads=[h.to(device) for h in heads],
    )
    
    diffusion = Diffusion(
        score_model,
        noisers,
        optim_config={"lr": params['lr']},
        scheduler_config={"factor": params['lr_factor'], "patience": params['lr_patience']},
    ).to(device)


    diffusion.load_state_dict(torch.load(Path(path) / 'checkpoints/best_model.ckpt', weights_only=True, map_location=device)['state_dict'])

    diffusion.eval()

    with torch.no_grad():
        graph_list = diffusion.sample(**sample_kwargs)


    Path(kwargs["output"]).mkdir(parents=True, exist_ok=True)
    name = kwargs["name"]
        
    if sample_kwargs.get('save_path', False):
        for i, graph_list_i in enumerate(graph_list):
            atoms_list = [g.to_atoms() for g in graph_list_i]
            write(Path(kwargs["output"]) / f"{name}_{i}.traj", atoms_list)

    else:
        atoms_list = [g.to_atoms() for g in graph_list]
        write(Path(kwargs["output"]) / f"{name}.traj", atoms_list)


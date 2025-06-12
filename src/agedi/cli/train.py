import rich_click as click
from rich import print

import torch
import numpy as np
from pathlib import Path

from lightning import Trainer
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint

from ase.io import read

from agedi import Diffusion
from agedi.models import ScoreModel
from agedi.data import Dataset

from agedi.data.callbacks import TrainingPhase
from agedi.data.transforms import Repeat

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
                    "--time",
                    "--lr",
                    "--batch_size",
                    "--lr_patience",
                    "--lr_factor",
                    "--progress-bar",
                    "--gradient_clip_val",
                ],
            },
            {"name": "Data Options", "options": ["--style", "--mask", "--confinement", "--repeat", "--repeat_epoch", "--conditioning_type"]},
            {"name": "Logging Options", "options": ["--logger", "--log_dir", "--project", "--name", "--log_interval"]},
        ]
    }
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
    "-b",
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
    help="type of heads to use",
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
    help="What type of conditionning to use (only relevant for data-augmentation!)",
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
    "--time", "-t", type=int, default=24, show_default=True, help="Time to train for in hours"
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
def train(**params):
    
    print("AGeDi Training Diffusion Model")
    print("-" * 30)
    print("Options:")
    for key, value in params.items():
        print(f"{key}: {value}")

    params["data"] = str(Path(params["data"]).resolve())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    conditionings = get_conditioning(params["conditioning"], type=params["conditioning_type"])
    head_dim = params["feature_size"] + sum([c.output_dim for c in conditionings])

    # Model
    translator, representation, heads = get_package(
        params["model"],
        params["cutoff"],
        params["noisers"],
        params["feature_size"],
        params["n_blocks"],
        head_dim=head_dim,
    )

    if params["confinement"] is not None and "positions" in params["noisers"]:
        confined = True
    else:
        confined = False

    noisers = get_noisers(params["noisers"], params["style"], confined=confined)

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
    )

    # Data
    data = read(params["data"], ":")

    if params["repeat"] is not None:
        if params["repeat"] < 2:
            raise ValueError("Repeat must be greater than 1")
        
        property={"mask": "node", "confinement": "none"}
        if params["conditioning"] != "none":
            if params["conditioning_type"] == "node":
                property[params["conditioning"]] = "node"
            else:
                property[params["conditioning"]] = "none"

        phase_transforms = [[],]
        for i in range(2, params["repeat"]+1):
            phase_transforms.append([Repeat((i, i, 1), property=property)])

    else:
        phase_transforms = None
    
    dataset = Dataset(
        cutoff=params["cutoff"],
        batch_size=params["batch_size"],
        phase_transforms=phase_transforms,
    )
    click.echo(f"Loaded dataset with {len(data)} samples")
    
    if params["conditioning"] != "none":
        properties = []
        for d in data:
            p = None
            try:
                p = getattr(d, f"get_{params['conditioning']}")()
            except AttributeError:
                pass
                
            try:
                p = d.info[params["conditioning"]]
            except KeyError:
                pass

            if p is None:
                p = 0
                print(f"Warning: {params['conditioning']} not found in data. Setting to 0!")

            properties.append({params["conditioning"]: p})
                
    else:
        properties = None
        
    dataset.add_atoms_data(
        data, mask_method=params["mask"], confinement=params["confinement"], properties=properties
    )
    dataset.setup()

    # Training
    if params["logger"] == "tensorboard":
        logger = TensorBoardLogger(save_dir=params["log_dir"], name="")
    elif params["logger"] == "wandb":
        logger = WandbLogger(
            save_dir=params["log_dir"],
            project=params["project"],
            name=params["name"],
        )
    # params["log_dir"] = str(Path(logger.log_dir).resolve())
    log_hparams = params | data_info(data)
    logger.log_hyperparams(log_hparams)

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
        # ema callback!
    ]
    if params["repeat"] is not None:
        callbacks.append(TrainingPhase(params["repeat"], [params["repeat_epoch"] for _ in range(params["repeat"]-1)]))

    trainer = Trainer(
        accelerator="auto",
        devices=1,
        max_epochs=params["epochs"],
        max_time={"hours": params["time"]} if params["time"] is not None else None,
        logger=logger,
        callbacks=callbacks,
        gradient_clip_val=params["gradient_clip_val"],
        enable_progress_bar=params["progress_bar"],
        log_every_n_steps=params["log_interval"],
        reload_dataloaders_every_n_epochs=1 if params["repeat"] is not None else 0,
        inference_mode=False,
    )

    trainer.fit(diffusion, dataset)

    print("To sample from model use: ")
    print(f"agedi sample {params['log_dir']} -f ...")


def get_noisers(noisers, style, confined=False):
    from agedi.diffusion.noisers import PositionsNoiser, TypesNoiser
    from agedi.diffusion.distributions import Normal, TruncatedNormal, UniformCell, UniformCellConfined, StandardNormal

    noiser_list = []
    for noiser in noisers:
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
                    prior=StandardNormal()
                    noiser_list.append(PositionsNoiser(prior=prior))
                else:
                    noiser_list.append(PositionsNoiser())


            case "types":
                # if "positions" in noisers:
                #     loss_scaling = 0.01
                # else:
                #     loss_scaling = 1.0
                noiser_list.append(TypesNoiser()) # loss_scaling=loss_scaling

            case _:
                raise ValueError(f"Unknown noiser {noiser}")

    return noiser_list


def get_conditioning(condition, type):
    from agedi.models.conditionings import TimeConditioning

    conditioning = [
        TimeConditioning(),
    ]

    if condition != "none":
        from agedi.models.conditionings import ScalarConditioning, IntegerConditioning
        if type == "scalar":
            conditioning.append(ScalarConditioning(property=condition))
        elif type == "integer":
            conditioning.append(IntegerConditioning(property=condition))
        else:
            raise ValueError(f"Unknown conditioning type {type}")

    return conditioning


def get_package(model, cutoff, heads, feature_size, n_blocks, head_dim):
    match model:
        case "PaiNN":
            import schnetpack as spk

            from agedi.models.schnetpack import (
                PositionsScore,
                TypesScore,
                SchNetPackTranslator,
            )

            input_modules = [
                spk.atomistic.PairwiseDistances(),
            ]

            translator = SchNetPackTranslator(input_modules=input_modules)

            representation = spk.representation.PaiNN(
                n_atom_basis=feature_size,
                n_interactions=n_blocks,
                radial_basis=spk.nn.GaussianRBF(n_rbf=30, cutoff=cutoff),
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
                        raise ValueError(f"Unknown head {head}")

        case _:
            raise ValueError(f"Unknown model {model}")

    return translator, representation, h


def data_info(data):
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

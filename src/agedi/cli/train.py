import rich_click as click
from rich import print

import torch
import numpy as np
from pathlib import Path

from lightning import Trainer
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint

from ase.io import read

from agedi.diffusion import Diffusion
from agedi.models import ScoreModel
from agedi.data import Dataset

click.rich_click.OPTION_GROUPS.update({
    "agedi train": [
        {"name": "Score Model Options", "options": ['--model', '--cutoff', '--feature_size', '--n_blocks']},
        {"name": "Diffusion Model Options", "options": ['--noisers', '--noiser_sdes', '--noiser_distributions', '--prior_distributions', '--condition']}, # , '--noiser_sde_kwargs'
        {"name": "Training Options", "options": ['--epochs', '--time', '--lr', '--batch_size', '--lr_patience', '--lr_factor', '--progress-bar']},
        {"name": "Data Options", "options": ['--mask', '--confinement']},
        {"name": "Logging Options", "options": ['--log_dir', '--log_interval']},
    ]
})

# @click.option('--noiser_sde_kwargs' , type=click.Choice(["VE", "VP"]), default=["VP"], multiple=True, show_default=True, help='type of SDE for each noiser')

@click.command()
@click.argument('data', type=click.Path(exists=True))
@click.option('--model', '-m', type=click.Choice(['PaiNN']), default="PaiNN", show_default=True, help='Representation to use for the model')
@click.option('--cutoff', '-r', type=float, default=6.0, show_default=True, help='Cutoff for the representation in Å')
@click.option('--feature_size', '-f', type=int, default=64, show_default=True, help='Feature size for the representation')
@click.option('--n_blocks', '-b', type=int, default=4, show_default=True, help='Number of blocks for the representation')
@click.option('--noisers', "-n" , type=click.Choice(["positions", "types", 'cell']), default=["positions"], multiple=True, show_default=True, help='type of heads to use')
@click.option('--noiser_sdes' , type=click.Choice(["VE", "VP"]), default=["VE"], multiple=True, show_default=True, help='type of SDE for each noiser')
@click.option('--noiser_distributions' , type=click.Choice(["Normal", "TruncatedNormal", "WrappedNormal"]),
              default=["Normal"], multiple=True, show_default=True, help='Noise distribution for each noiser')
@click.option('--prior_distributions' , type=click.Choice(["UniformCell", "UniformCellConfined"]),
              default=["UniformCell"], multiple=True, show_default=True, help='Prior distribution for each noiser')
@click.option('--conditioning', '-c', type=click.Choice(['none']), default='none', help='type of conditionings to use', hidden=True)
@click.option('--epochs', '-e', type=int, default=1e6, show_default=True, help='Number of epochs to train for')
@click.option('--time', '-t', type=int, default=1440, help='Time to train for in minutes')
@click.option('--lr', type=float, default=1e-4, show_default=True, help='Learning rate')
@click.option('--batch_size', '-b', type=int, default=32, show_default=True, help='Batch size')
@click.option('--lr_patience', type=int, default=100, show_default=True, help='Number of epochs to wait before reducing the learning rate')
@click.option('--lr_factor', type=float, default=0.98, show_default=True, help='Factor to reduce the learning rate by')
@click.option('--mask', type=click.Choice(['MaskFixed', 'none']), default='none', help='Masking to use for the data')
@click.option('--confinement', nargs=2, type=float, default=None, help='Z-confinement to use for the data. Give min and max value')
@click.option('--log_dir', type=click.Path(), default='logs', show_default=True, help='Directory to save logs to')
@click.option('--log_interval', type=int, default=10, show_default=True, help='Interval to log at')
@click.option('--progress_bar', is_flag=True, help='Show progress bar')
def train(
        **params
):
    print("AGeDi Training Diffusion Model")
    print('-'*30)
    print('Options:')
    for key, value in params.items():
        print(f'{key}: {value}')

    params['data'] = str(Path(params['data']).resolve())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Model
    translator, representation, heads = get_package(params['model'], params['cutoff'], params['noisers'], params['feature_size'], params['n_blocks'])
    conditionings = get_conditioning(params['conditioning'])
    noisers = get_noisers(params['noisers'], params['noiser_sdes'], params['noiser_distributions'], params['prior_distributions'])

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
    )

    # Data
    data = read(params['data'], ":")
    dataset = Dataset(cutoff=params['cutoff'], batch_size=params['batch_size'])
    click.echo(f"Loaded dataset with {len(data)} samples")
    dataset.add_atoms_data(data, mask_method=params['mask'], confinement=params['confinement'])
    dataset.setup()

    # Training
    logger = TensorBoardLogger(save_dir=params['log_dir'], name='')
    params['log_dir'] = str(Path(logger.log_dir).resolve())
    log_hparams = params | data_info(data)
    logger.log_hyperparams(log_hparams, {"val_loss": 0})

    callbacks = [
        LearningRateMonitor(logging_interval="epoch"),
        ModelCheckpoint(
            monitor="val_loss",
            filename="best_model",
            save_top_k=1,
            mode="min",
        ),
        ModelCheckpoint(
            filename="last_model",
            monitor=None,
            save_top_k=1,
            every_n_epochs=100,
        ),
        # ema callback!
    ]

    trainer = Trainer(
        accelerator="auto",
        devices=1,
        max_epochs=params['epochs'],
        max_time={"minutes": params['time']} if params['time'] is not None else None,
        logger=logger,
        callbacks=callbacks,
        gradient_clip_val=1.0,
        enable_progress_bar=params['progress_bar'],
        log_every_n_steps=params['log_interval'],
        inference_mode=False,
    )

    trainer.fit(diffusion, dataset)

    print("To sample from model use: ")
    print(f"agedi sample {params['log_dir']} -f ...")
    
def get_noisers(noisers, sdes, distributions, priors):
    from agedi.diffusion.noisers import PositionsNoiser
    from agedi.diffusion.noisers import VP, VE
    from agedi.diffusion.noisers import Normal, TruncatedNormal, WrappedNormal, UniformCell, UniformCellConfined
    noiser_list = []
    for noiser, sde, dist, prior in zip(noisers, sdes, distributions, priors):
        match sde:
            case "VE":
                sde = VE
            case "VP":
                sde = VP
            case _:
                raise ValueError(f"Unknown SDE {sde}")
        match dist:
            case "Normal":
                dist = Normal()
            case "TruncatedNormal":
                dist = TruncatedNormal()
            case "WrappedNormal":
                dist = WrappedNormal()
            case _:
                raise ValueError(f"Unknown Distribution {dist}")
        match prior:
            case "UniformCell":
                prior = UniformCell()
            case "UniformCellConfined":
                prior = UniformCellConfined()
            case _:
                raise ValueError(f"Unknown Prior {prior}")
        match noiser:
            case 'positions':
                noiser_list.append(PositionsNoiser(sde_class=sde, distribution=dist, prior=prior))
            case _:
                raise ValueError(f'Unknown noiser {noiser}')

    return noiser_list
    
def get_conditioning(condition):
    from agedi.models.conditionings import TimeConditioning
    conditioning = [TimeConditioning(),]
    match condition:
        case 'none':
            pass

    return conditioning
    
def get_package(model, cutoff, heads, feature_size, n_blocks):
    match model:
        case 'PaiNN':
            import schnetpack as spk

            from agedi.models.schnetpack import (PositionsScore,
                                                 SchNetPackTranslator)

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
                    case 'positions':
                        h.append(PositionsScore())
                    case _:
                        raise ValueError(f'Unknown head {head}')


        case _:
            raise ValueError(f'Unknown model {model}')


    return translator, representation, h
    
def data_info(data):
    elements = set()

    out = {'cell': None}
    check_cell = True
    for d in data:
        elements.update(d.get_chemical_symbols())
        if check_cell:
            if d.cell is not None:
                out['cell'] = np.array(d.cell)
            else:
                if not np.all(out['cell'] == d.cell):
                    check_cell = False
    out['cell'] = out['cell'].flatten().tolist() if out['cell'] is not None else None
                    
    out |= {
        "symbols": list(elements),
        "n_training_data": len(data),
    }
        
    return out



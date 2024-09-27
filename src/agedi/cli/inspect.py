import yaml
from rich import print
import rich_click as click
from pathlib import Path

@click.command()
@click.argument("path", type=click.Path(exists=True))
def inspect(path):
    click.echo(f"Inspecting {path}")
    # read yaml file
    with open(Path(path) / 'hparams.yaml', "r") as file:
        params = yaml.safe_load(file)
        
    print(params)
    # print(click.rich_click.OPTION_GROUPS)

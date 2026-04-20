import rich_click as click
from rich.console import Console


@click.command("train-hydra")
@click.argument("config", type=click.Path(exists=True))
@click.argument("overrides", nargs=-1, metavar="[KEY=VALUE ...]")
def train_hydra(config: str, overrides: tuple) -> None:
    """Train an AGeDi model from a YAML configuration file.

    CONFIG is the path to a YAML file containing all training parameters.
    A ready-to-edit template is available at ``agedi/conf/train.yaml``.

    Optional KEY=VALUE pairs can be appended to override individual config
    entries without editing the file:

    \b
        agedi train-hydra conf/train.yaml feature_size=128 epochs=200

    Supported override types: int, float, bool (``true``/``false``), and str.
    Nested keys are not currently supported via overrides (edit the YAML
    directly for nested values).
    """
    import yaml

    from agedi.functional import train_from_config

    # Load base config from file.
    with open(config) as fh:
        cfg: dict = yaml.safe_load(fh) or {}

    # Apply KEY=VALUE overrides.
    for override in overrides:
        if "=" not in override:
            raise click.UsageError(
                f"Override '{override}' is not in KEY=VALUE format."
            )
        key, _, raw_value = override.partition("=")
        cfg[key] = _parse_override_value(raw_value)

    train_from_config(cfg)

    console = Console()
    console.print("\n[green]✓ Training complete.[/green]")
    log_dir = cfg.get("log_dir", "logs")
    console.print("To sample from the model run:")
    console.print(f"  [bold]agedi sample {log_dir} -f ...[/bold]")


def _parse_override_value(raw: str):
    """Coerce a CLI override string to int, float, bool, or str."""
    if raw.lower() == "true":
        return True
    if raw.lower() == "false":
        return False
    if raw.lower() in ("null", "none", "~"):
        return None
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        pass
    return raw

import yaml
import rich_click as click
from pathlib import Path
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table


@click.command()
@click.argument("path", type=click.Path(exists=True))
def inspect(path: str) -> None:
    """Inspect a trained AGeDi model directory.

    Reads and prints the hyperparameters stored in ``hparams.yaml`` inside the
    given model directory, and reports available training metrics (epochs,
    loss, etc.) from the latest checkpoint.
    """
    from agedi.functional import _print_training_config

    console = Console()
    root_path = Path(path)
    # If the user points at a checkpoint file, go up two levels to the run dir.
    if root_path.is_file():
        root_path = root_path.parent.parent

    params_path = root_path / "hparams.yaml"
    if not params_path.exists():
        console.print(f"[red]Error:[/red] No hparams.yaml found in {root_path}")
        raise SystemExit(1)

    with open(params_path, "r") as fh:
        params = yaml.safe_load(fh) or {}

    # --- Hyperparameter panel ---
    _print_training_config(params)

    # --- Training metrics from checkpoint ---
    ckpt_path = root_path / "checkpoints" / "last_model.ckpt"
    if not ckpt_path.exists():
        # Try any .ckpt file in the checkpoints directory
        ckpt_dir = root_path / "checkpoints"
        ckpts = sorted(ckpt_dir.glob("*.ckpt")) if ckpt_dir.is_dir() else []
        ckpt_path = ckpts[-1] if ckpts else None

    if ckpt_path is not None and ckpt_path.exists():
        try:
            import torch

            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            table = Table(show_header=False, box=box.SIMPLE, padding=(0, 1))
            table.add_column("Key", style="bold cyan", min_width=22, no_wrap=True)
            table.add_column("Value", style="white")

            epoch = ckpt.get("epoch")
            global_step = ckpt.get("global_step")
            if epoch is not None:
                table.add_row("  epoch", str(epoch))
            if global_step is not None:
                table.add_row("  global_step", str(global_step))

            # Best val loss from ModelCheckpoint callback state
            best_val_loss = None
            best_ckpt_path_str = None
            callbacks_state = ckpt.get("callbacks", {})
            for cb_key, cb_state in callbacks_state.items():
                # cb_key is a string like "ModelCheckpoint{...}"
                if "ModelCheckpoint" in str(cb_key):
                    score = cb_state.get("best_model_score")
                    if score is not None:
                        try:
                            best_val_loss = float(score)
                        except (TypeError, ValueError):
                            pass
                    best_ckpt_path_str = cb_state.get("best_model_path") or None
                    break

            if best_val_loss is not None:
                table.add_row("  best_val_loss", f"{best_val_loss:.6f}")
            if best_ckpt_path_str:
                table.add_row("  best_checkpoint", best_ckpt_path_str)

            table.add_row("  checkpoint", str(ckpt_path))

            console.print(
                Panel(
                    table,
                    title="[bold]Training Metrics[/bold]",
                    border_style="magenta",
                )
            )
        except Exception as exc:
            console.print(f"[yellow]Could not read checkpoint metrics:[/yellow] {exc}")

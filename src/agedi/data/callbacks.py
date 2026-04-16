import time
from pathlib import Path
from typing import Dict, List, Optional

import torch
import yaml
from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from torch_geometric.transforms import BaseTransform


class GradNormLogger(Callback):
    """Logs the total gradient norm of the score-model parameters before each optimizer step.

    Parameters
    ----------
    log_every_n_steps:
        Log the gradient norm every this many optimizer steps (default: ``50``).
        Set to ``1`` to log every step.
    """

    def __init__(self, log_every_n_steps: int = 50):
        if log_every_n_steps < 1:
            raise ValueError(f"log_every_n_steps must be >= 1, got {log_every_n_steps}")
        self.log_every_n_steps = log_every_n_steps

    def on_before_optimizer_step(self, trainer, pl_module, optimizer):
        if trainer.global_step % self.log_every_n_steps != 0:
            return
        norms = [
            p.grad.detach().norm()
            for p in pl_module.score_model.parameters()
            if p.grad is not None
        ]
        if norms:
            total_norm = torch.stack(norms).norm()
            pl_module.log("grad_norm", total_norm, on_step=True, on_epoch=False)


class EpochProgressPrinter(Callback):
    """Prints epoch-level training progress to stdout at a configurable interval.

    Parameters
    ----------
    print_epoch_interval:
        Print a summary line every this many epochs (default: 10).
    """

    def __init__(self, print_epoch_interval: int = 10):
        self.print_epoch_interval = print_epoch_interval
        self._fit_start_time: float = 0.0

    def on_fit_start(self, trainer, pl_module):
        self._fit_start_time = time.monotonic()

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return

        epoch = trainer.current_epoch + 1
        if epoch % self.print_epoch_interval != 0:
            return

        metrics = trainer.callback_metrics
        train_loss = metrics.get("train_loss_epoch", metrics.get("train_loss"))
        val_loss = metrics.get("val_loss")

        # LearningRateMonitor logs LR as 'lr-<OptimizerClass>' or 'lr-<name>'
        lr = None
        for k, v in metrics.items():
            if k.lower().startswith("lr"):
                lr = float(v)
                break

        parts = [f"Epoch {epoch:>6d}"]
        if train_loss is not None:
            parts.append(f"train_loss: {float(train_loss):.4f}")
        if val_loss is not None:
            parts.append(f"val_loss: {float(val_loss):.4f}")
        if lr is not None:
            parts.append(f"lr: {lr:.2e}")

        print(" | ".join(parts))

    def on_fit_end(self, trainer, pl_module):
        elapsed = time.monotonic() - self._fit_start_time
        hours, rem = divmod(int(elapsed), 3600)
        minutes, seconds = divmod(rem, 60)

        best_val_loss: Optional[float] = None
        best_ckpt_path: Optional[str] = None
        for cb in trainer.callbacks:
            if isinstance(cb, ModelCheckpoint) and cb.monitor == "val_loss":
                if cb.best_model_score is not None:
                    best_val_loss = float(cb.best_model_score)
                best_ckpt_path = cb.best_model_path or None
                break

        print(f"\nTraining complete  |  elapsed: {hours:02d}h {minutes:02d}m {seconds:02d}s")
        if best_val_loss is not None:
            print(f"Best val_loss      : {best_val_loss:.6f}")
        if best_ckpt_path:
            print(f"Best checkpoint    : {best_ckpt_path}")


class HParamsMetricLogger(Callback):
    """Manages hyperparameter logging and populates the TensorBoard hp_metric panel.

    For TensorBoard:
      * Writes ``hparams.yaml`` at training start (before any epoch) so the file
        is available for crash recovery via :func:`~agedi.functional.load_diffusion`.
      * Logs a single HPARAMS entry with ``hp_metric = best_val_loss`` at fit end,
        replacing the empty "dot" that TensorBoard normally shows.

    For other loggers (e.g. WandB):
      * Calls ``log_hyperparams`` normally at training start.

    Parameters
    ----------
    hparams:
        Dictionary of hyperparameters to log.
    """

    def __init__(self, hparams: Dict):
        self._hparams = hparams

    def on_train_start(self, trainer, pl_module):
        from lightning.pytorch.loggers import TensorBoardLogger

        if isinstance(trainer.logger, TensorBoardLogger):
            # Write hparams.yaml directly so it is available immediately without
            # creating a "dot" entry in the TensorBoard HPARAMS plugin.
            log_dir = Path(trainer.logger.log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)
            with open(log_dir / "hparams.yaml", "w") as fh:
                yaml.safe_dump(self._hparams, fh, default_flow_style=False)
        elif trainer.logger is not None:
            trainer.logger.log_hyperparams(self._hparams)

    def on_fit_end(self, trainer, pl_module):
        from lightning.pytorch.loggers import TensorBoardLogger

        if not isinstance(trainer.logger, TensorBoardLogger):
            return

        best_val_loss: Optional[float] = None
        for cb in trainer.callbacks:
            if isinstance(cb, ModelCheckpoint) and cb.monitor == "val_loss":
                if cb.best_model_score is not None:
                    best_val_loss = float(cb.best_model_score)
                break

        metrics = {"hp_metric": best_val_loss} if best_val_loss is not None else {}
        trainer.logger.log_hyperparams(self._hparams, metrics)


class TrainingPhase(Callback):
    """Lightning callback that advances the dataset through training phases.

    Each phase can use a different set of data augmentation transforms (e.g.
    supercell repeats).  The callback monitors epoch count and calls
    :meth:`~agedi.data.Dataset.set_phase` on the datamodule when it is time
    to move to the next phase.
    """

    def __init__(
        self,
        n_phases: int,
        epochs_per_phase: List[int],
        **kwargs
    ) -> None:
        """Initialize the training phase callback.

        Parameters
        ----------
        n_phases : int
            Total number of training phases.
        epochs_per_phase : list[int]
            Number of epochs to spend in each phase (length ``n_phases - 1``).
        **kwargs
            Additional keyword arguments forwarded to :class:`~lightning.pytorch.callbacks.Callback`.
        """
        super().__init__(**kwargs)
        self.n_phases = n_phases

        self.epochs_per_phase = epochs_per_phase
        self.epoch_counter = 0

        self.current_phase = 0


    def _prepare_epoch(self, trainer: Trainer, model: LightningModule) -> None:
        """Advance to the next training phase if enough epochs have elapsed.

        Called at the end of each validation epoch.  When the epoch counter
        reaches the threshold for the current phase, the datamodule is
        instructed to switch to the next phase via
        :meth:`~agedi.data.Dataset.set_phase`.

        Parameters
        ----------
        trainer : Trainer
            The active Lightning trainer.
        model : LightningModule
            The model being trained (unused, required by Lightning callback API).
        """

        if self.current_phase == self.n_phases - 1:
            return

        epoch = trainer.current_epoch
        if self.epoch_counter >= self.epochs_per_phase[self.current_phase]:
            self.current_phase += 1
            trainer.datamodule.set_phase(self.current_phase)
            
            self.epoch_counter = 0
        else:
            self.epoch_counter += 1
            

    def on_validation_end(self, trainer: Trainer, model: LightningModule) -> None:
        """Hook called by Lightning at the end of each validation epoch.

        Delegates to :meth:`_prepare_epoch` to check whether the current
        training phase should advance.

        Parameters
        ----------
        trainer : Trainer
            The active Lightning trainer.
        model : LightningModule
            The model being trained.
        """
        self._prepare_epoch(trainer, model)

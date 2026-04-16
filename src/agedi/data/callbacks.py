from typing import List
from lightning.pytorch.callbacks import Callback
from torch_geometric.transforms import BaseTransform

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
    ):
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


    def _prepare_epoch(self, trainer, model):
        """Advance to the next training phase if enough epochs have elapsed.

        Called at the end of each validation epoch.  When the epoch counter
        reaches the threshold for the current phase, the datamodule is
        instructed to switch to the next phase via
        :meth:`~agedi.data.Dataset.set_phase`.

        Parameters
        ----------
        trainer : lightning.Trainer
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
            

    def on_validation_end(self, trainer, model):
        """Hook called by Lightning at the end of each validation epoch.

        Delegates to :meth:`_prepare_epoch` to check whether the current
        training phase should advance.

        Parameters
        ----------
        trainer : lightning.Trainer
            The active Lightning trainer.
        model : LightningModule
            The model being trained.
        """
        self._prepare_epoch(trainer, model)

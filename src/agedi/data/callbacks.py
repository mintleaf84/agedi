from typing import List
from lightning.pytorch.callbacks import Callback
from torch_geometric import Transform


class TrainingPhaseCallback(Callback):
    def __init__(
        self,
        n_phases: int,
        monitor: str = "val_loss",
        mode: str = "min",
        patience: int = 1000,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.n_phases = n_phases

        self.monitor = monitor
        self.mode = mode
        self.patience = patience

        self.change_phase = False
        self.current_phase = 0
        self.current_patience = patience
        self.best_score = None

    def _prepare_epoch(self, trainer, model):
        # phase = ...
        # trainer.datamodule.set_phase(phase)

        if self.current_phase == self.n_phases - 1:
            return

        value = trainer.callback_metrics.get(self.monitor)
        if self.best_score is None:
            self.best_score = value
        elif self.mode == "min":
            better = value < self.best_score
        elif self.mode == "max":
            better = value > self.best_score

        if better:
            self.best_score = value
            self.current_patience = self.patience
        else:
            self.current_patience -= 1

        if self.current_patience == 0:
            self.current_phase += 1
            self.current_patience = self.patience
            self.change_phase = True

        trainer.datamodule.set_phase(self.current_phase)

    def on_epoch_end(self, trainer, model):
        self._prepare_epoch(trainer, model)

    def on_epoch_start(self, trainer, model):
        if self.change_phase:
            trainer.train_dataloader = trainer.datamodule.train_dataloader()
            trainer.val_dataloader = trainer.datamodule.val_dataloader()
            trainer.test_dataloader = trainer.datamodule.test_dataloader()

            self.change_phase = False

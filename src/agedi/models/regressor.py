import torch
from torch.nn import functional as F
from lightning import LightningModule

from typing import Dict, List

from torch_geometric.data import Batch
from agedi.models.translator import Translator
from agedi.data import Representation
from agedi.models.head import Head


class RegressorModel(LightningModule):
    """Class that defines a regressor model.

    It is a combination of a translator, a representation
    and a list of heads.

    Parameters
    ----------
    translator: Translator
        The translator that will be used to translate the input batch.
    representation: Representation
        The representation that will be used to represent the translated batch.
    heads: List[Head]
        The list of heads that will be used to compute scores.

    """

    def __init__(
        self,
        translator: Translator,
        representation: Representation,
        heads: List[Head] = [],
        head_weights = {},
        use_weighting: bool = False,
        **kwargs
    ):
        """Constructor for the ScoreModel class."""
        super().__init__(**kwargs)
        self.translator = translator
        self.representation = representation
        self.head_weights = head_weights
        self.use_weighting = use_weighting
        
        self.head_keys = [head.key for head in heads]
        for key in self.head_keys:
            if key not in ["energy", "forces"]:
                raise ValueError(f"Head key {key} not recognized.")
        
        self.heads = torch.nn.ModuleList(heads)

    def get_hparams(self) -> Dict:
        """Return hyperparameters sufficient to reconstruct this regressor model.

        Returns
        -------
        dict
            Hyperparameter dictionary with a ``_target_`` key and nested
            ``translator``, ``representation``, and ``heads`` entries.
        """
        return {
            "_target_": f"{type(self).__module__}.{type(self).__qualname__}",
            "translator": self.translator.get_hparams(),
            "representation": self.translator.get_representation_hparams(self.representation),
            "heads": [h.get_hparams() for h in self.heads],
        }

    def forward(self, batch: Batch) -> Batch:
        """Forward pass of the model.

        Parameters
        ----------
        batch: Batch
            The input batch that will be used to compute the scores.

        Returns
        -------
        Batch
            The output batch containing the scores.

        """
        translated_batch = self.translator(batch)
        
        # if batch.representation is None:
        rep = self.representation(translated_batch)
        batch = self.translator.add_representation(batch, rep)
        translated_batch = self.translator(batch)

        predictions = {}
        for head in self.heads:
            predictions[head.key] = head(translated_batch)
                
        batch = self.translator.add_prediction(batch, predictions)
        return batch

    def loss(self, batch: Batch) -> torch.Tensor:
        """Compute the loss of the model.

        Parameters
        ----------
        batch: Batch
            The input batch that will be used to compute the loss.

        Returns
        -------
        torch.Tensor
            The computed loss.

        """
        batch = self(batch)

        loss = 0.0
        for key in self.head_keys:
            f = batch[key]
            f_pred = batch[f"{key}_prediction"]
            if self.use_weighting and 'weight' in batch:
                weights = batch.weight[batch.batch].unsqueeze(-1)
                loss += self.head_weights.get(key, 1.0) * (F.mse_loss(f, f_pred, reduction='none') * weights).mean()
            else:
                loss += self.head_weights.get(key, 1.0) * F.mse_loss(f, f_pred)
        return loss

    




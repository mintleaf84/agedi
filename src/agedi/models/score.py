import torch

from typing import List

from torch_geometric.data import Batch
from agedi.models.conditionings import Conditioning, TimeConditioning
from agedi.models.translator import Translator
from agedi.data import Representation
from agedi.models.head import Head


class ScoreModel(torch.nn.Module):
    """Class that defines a the score model.

    It is a combination of a translator, a representation, a list of conditionings
    and a list of heads.

    Parameters
    ----------
    translator: Translator
        The translator that will be used to translate the input batch.
    representation: Representation
        The representation that will be used to represent the translated batch.
    conditionings: List[Conditioning]
        The list of conditionings that will be applied to the representation.
    heads: List[Head]
        The list of heads that will be used to compute scores.

    """

    def __init__(
        self,
        translator: Translator,
        representation: Representation,
        conditionings: List[Conditioning] = [
            TimeConditioning(),
        ],
        heads: List[Head] = [],
        w: float = -1.0,
        **kwargs
    ):
        """Constructor for the ScoreModel class."""
        super().__init__(**kwargs)
        self.translator = translator
        self.representation = representation
        self.conditionings = torch.nn.ModuleList(conditionings),
        self.heads = torch.nn.ModuleList(heads)

        # self.register_buffer("w", torch.tensor(w))
        self.w = torch.tensor(w)
        self.guidance = True if w > -1.0 else False

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
        rep = self.representation(translated_batch)

        batch = self.translator.add_representation(batch, rep)

        if self.guidance:
            batch_cond = batch.clone()

        for conditioning in self.conditionings:
            batch = conditioning(batch, empty=True)
            if self.guidance:
                batch_cond = conditioning(batch_cond, empty=False)

        translated_batch = self.translator(batch)
        if self.guidance:
            translated_batch_cond = self.translator(batch_cond)
        scores = {}
        for head in self.heads:
            if self.guidance:
                scores[head.key] = (1 + self.w) * head(
                    translated_batch_cond
                ) - self.w * head(translated_batch)
            else:
                scores[head.key] = head(translated_batch)

        batch = self.translator.add_scores(batch, scores)

        return batch

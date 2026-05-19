import torch
from lightning import LightningModule

from typing import Dict, List, Optional

from torch_geometric.data import Batch
from agedi.models.conditionings import Conditioning, TimeConditioning
from agedi.models.translator import Translator
from agedi.data import Representation
from agedi.models.head import Head


class ScoreModel(LightningModule):
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
        conditionings: Optional[List[Conditioning]] = None,
        heads: Optional[List[Head]] = None,
        w: float = -1.0,
        **kwargs
    ):
        """Constructor for the ScoreModel class."""
        super().__init__(**kwargs)
        self.translator = translator
        self.representation = representation
        self.conditionings = torch.nn.ModuleList(
            conditionings if conditionings is not None else [TimeConditioning()]
        )
        self.heads = torch.nn.ModuleList(heads if heads is not None else [])

        # self.register_buffer("w", torch.tensor(w))
        self.w = torch.tensor(w)
        self.guidance = True if w > -1.0 else False

        self.training_mode()

    def get_hparams(self) -> Dict:
        """Return hyperparameters sufficient to reconstruct this score model.

        Collects hyperparameters from the translator, representation (via
        :meth:`~agedi.models.translator.Translator.get_representation_hparams`),
        conditionings, and heads, as well as the guidance weight ``w``.

        Returns
        -------
        dict
            Hyperparameter dictionary with a ``_target_`` key and nested
            ``translator``, ``representation``, ``conditionings``, ``heads``,
            and ``w`` entries.
        """
        return {
            "_target_": f"{type(self).__module__}.{type(self).__qualname__}",
            "translator": self.translator.get_hparams(),
            "representation": self.translator.get_representation_hparams(self.representation),
            "conditionings": [c.get_hparams() for c in self.conditionings],
            "heads": [h.get_hparams() for h in self.heads],
            "w": float(self.w),
        }

    def forward(self, batch: Batch) -> Batch:
        """Forward pass of the model.

        Dispatches to :meth:`forward_sample` when the model is in sampling
        mode, and to :meth:`forward_train` otherwise.  This keeps
        ``self.sample`` as a compile-time constant for each compiled subgraph,
        avoiding retracing on mode changes and eliminating the Python-level
        branch from the compiled region.

        Parameters
        ----------
        batch: Batch
            The input batch that will be used to compute the scores.

        Returns
        -------
        Batch
            The output batch containing the scores.

        """
        if self.sample:
            return self.forward_sample(batch)
        return self.forward_train(batch)

    def forward_train(self, batch: Batch) -> Batch:
        """Training-mode forward pass.

        Computes the backbone representation, applies all conditionings
        unconditionally, translates the conditioned batch, and evaluates
        every score head.

        Parameters
        ----------
        batch: Batch
            The input batch.

        Returns
        -------
        Batch
            Batch with score tensors attached.
        """
        translated_batch = self.translator.translate_input(batch)
        rep = self.representation(translated_batch)
        batch = self.translator.add_representation(batch, rep)

        for conditioning in self.conditionings:
            batch = conditioning(batch, empty=False)
        translated_batch = self.translator.translate_with_representation(batch)

        scores = {}
        for head in self.heads:
            scores[head.key] = head(translated_batch)

        batch = self.translator.add_scores(batch, scores)
        return batch

    def forward_sample(self, batch: Batch) -> Batch:
        """Sampling-mode forward pass.

        Computes the backbone representation, applies classifier-free guidance
        (when ``self.guidance`` is ``True``), translates the conditioned batch,
        and evaluates every score head with optional guidance mixing.

        Parameters
        ----------
        batch: Batch
            The input batch.

        Returns
        -------
        Batch
            Batch with score tensors attached.
        """
        translated_batch = self.translator.translate_input(batch)
        rep = self.representation(translated_batch)
        batch = self.translator.add_representation(batch, rep)

        if self.guidance:
            batch_cond = batch.clone()

        for conditioning in self.conditionings:
            batch = conditioning(batch, empty=True)
            if self.guidance:
                batch_cond = conditioning(batch_cond, empty=False)

        translated_batch = self.translator.translate_with_representation(batch)
        if self.guidance:
            translated_batch_cond = self.translator.translate_with_representation(batch_cond)

        scores = {}
        for head in self.heads:
            if self.guidance:
                scores[head.key] = head(translated_batch) + self.w * (
                    head(translated_batch_cond) - head(translated_batch)
                )
            else:
                scores[head.key] = head(translated_batch)

        batch = self.translator.add_scores(batch, scores)
        return batch

    def sample_mode(self) -> None:
        """Switch the model to sampling mode.

        Sets ``self.sample = True`` and calls ``sample_mode()`` on all
        conditioning modules so that classifier-free guidance is applied
        during inference.
        """
        self.sample = True
        for conditioning in self.conditionings:
            conditioning.sample_mode()

    def training_mode(self) -> None:
        """Switch the model to training mode.

        Sets ``self.sample = False`` and calls ``training_mode()`` on all
        conditioning modules so that conditioning is applied unconditionally
        during the forward pass.
        """
        self.sample = False
        for conditioning in self.conditionings:
            conditioning.training_mode()
        

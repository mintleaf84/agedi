from abc import ABC, abstractmethod

from agedi.diffusion.distributions import Distribution
from agedi.data import AtomsGraph

import torch


class Noiser(ABC, torch.nn.Module):
    """Noiser Base class

    Impments a noiser that can noise and denoise a atomistic structure attribute.

    Parameters
    ----------
    distribution: Distribution
        The distribution to be used for the noising.
    prior: Distribution
        The prior to be used for the denoising.
    key: str
        The key of the attribute to be noised and denoised.

    Returns
    -------
    Noiser

    """

    _key: str

    def __init__(
        self,
        distribution: Distribution,
        prior: Distribution,
        loss_scaling: float = 1.0,
        **kwargs
    ):
        """Initializes the Noiser."""
        super().__init__(**kwargs)

        self.distribution = distribution
        self.distribution.key = self.key

        self.prior = prior
        self.prior.key = self.key

        self.loss_scaling = loss_scaling

    @property
    def key(self) -> str:
        """The key of the attribute to be noised and denoised."""
        return self._key

    @abstractmethod
    def _noise(self, batch: AtomsGraph) -> AtomsGraph:
        """Noises the attribute of the atomistic structure.

        Must be implemented by the subclass.

        Parameters
        ----------
        batch: AtomsGraph
            The atomistic structure (or batch hereof) to be noised.

        Returns
        -------
        AtomsGraph
            The noised atomistic structure (or bach hereof).

        """
        pass

    @abstractmethod
    def _denoise(self, batch: AtomsGraph, delta_t: float, last: bool) -> AtomsGraph:
        """Denoises the attribute of the atomistic structure.

        Must be implemented by the subclass.

        Parameters
        ----------
        batch: AtomsGraph
            The atomistic structure (or batch hereof) to be denoised.

        delta_t: float
            The time step to be used for the denoising.

        Returns
        -------
        AtomsGraph
            The denoised atomistic structure (or bach hereof).

        """
        pass

    @abstractmethod
    def _loss(self, batch: AtomsGraph) -> float:
        """Computes the training loss.

        Must be implemented by the subclass.

        Parameters
        ----------
        batch: AtomsGraph
            The atomistic structure (or batch hereof) to be noised and denoised.

        Returns
        -------
        float
            The loss of the noised and denoised atomistic structure.

        """
        pass

    def noise(self, batch: AtomsGraph) -> AtomsGraph:
        """Noises the attribute of the atomistic structure.

        Parameters
        ----------
        batch: AtomsGraph
            The atomistic structure (or batch hereof) to be noised.

        Returns
        -------
        AtomsGraph
            The noised atomistic structure (or bach hereof).

        """
        return self._noise(batch)

    def denoise(self, batch: AtomsGraph, delta_t: float, last: bool) -> AtomsGraph:
        """Denoises the attribute of the atomistic structure.

        Parameters
        ----------
        batch: AtomsGraph
            The atomistic structure (or batch hereof) to be denoised.
        delta_t: float
            The time step to be used for the denoising.
        last: bool
            If the denoising is the last step of the denoising.

        Returns
        -------
        AtomsGraph
            The denoised atomistic structure (or bach hereof).

        """
        return self._denoise(batch, delta_t, last)

    def loss(self, batch: AtomsGraph) -> float:
        """Compute the training loss.

        Parameters
        ----------
        batch: AtomsGraph
            The atomistic structure (or batch hereof) to be noised and denoised.

        Returns
        -------
        float
            The loss of the noised and denoised atomistic structure.

        """
        return self._loss(batch)

    def initialize_graph(self, batch: AtomsGraph) -> None:
        """Initializes the graph with the prior distribution.

        Can be overwritten by subclasses for specific initializations.

        Parameters
        ----------
        batch: AtomsGraph
            The atomistic structure (or batch hereof) to be noised and denoised.

        """
        setattr(
            batch,
            self.key,
            self.prior.get_callable(batch)(),
        )

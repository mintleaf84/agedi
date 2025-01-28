from abc import ABC, abstractmethod
from typing import Callable, Optional

import torch

from agedi.data import AtomsGraph


class Distribution(ABC):
    """Base Class for noise distributions

    Parameters
    ----------
    key : str
        Key to identify the property from the batch

    Returns
    -------
    Distribution

    """

    def __init__(self, key:Optional[str] = None, **kwargs):
        """Initialize the distribution"""
        self.key = key

    @abstractmethod
    def _sample(self, **kwargs) -> torch.Tensor:
        """Sample distribution

        Sample from the distribution and return tensor of shape self.key

        Parameters
        ----------
        kwargs : dict
            The parameters of the distribution

        Returns
        -------
        torch.Tensor
            Sampled tensor
        """
        pass

    def _setup(self, batch: AtomsGraph) -> None:
        """Prepare distribution

        Prepare the distribution for sampling of the batch

        Parameters
        ----------
        batch : AtomsGraph
            Batch of data

        Returns
        -------
        None

        """
        pass

    def get_callable(self, batch: AtomsGraph) -> Callable:
        """Get callable function

        Return a callable function that samples from the distribution

        Parameters
        ----------
        batch : AtomsGraph
            Batch of data

        Returns
        -------
        Callable
            Callable function that samples from the distribution

        """
        self._setup(batch)

        def callable(**kwargs):
            return self._sample(**kwargs)

        return callable

from abc import ABC, abstractmethod
from typing import Callable, Type

import torch
from torch.distributions.categorical import Categorical as Cat

from agedi.data import AtomsGraph
from agedi.utils import TruncatedNormal as TN



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

    def __init__(self, **kwargs):
        """Initialize the distribution

        """
        self.key = None

    @abstractmethod
    def _sample(self, mu: torch.Tensor, sigma: torch.Tensor, **kwargs) -> torch.Tensor:
        """Sample distribution
        
        Sample from the distribution and return tensor of shape self.key

        Parameters
        ----------
        mu : torch.Tensor
            Mean of the distribution
        sigma : torch.Tensor
            Standard deviation of the distribution

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

        def callable(mu, sigma, **kwargs):
            return self._sample(mu, sigma, **kwargs)

        return callable

class StandardNormal(Distribution):
    """Standard Normal Distribution

    """

    def _sample(self, mu, sigma, **kwargs) -> torch.Tensor:
        """Sample from the standard normal distribution

        Parameters
        ----------
        mu : torch.Tensor
            Mean of the distribution
        sigma : torch.Tensor
            Standard deviation of the distribution

        Returns
        -------
        torch.Tensor
            Sampled tensor
        
        """
        shape = mu.shape
        return torch.normal(0.0, 1.0, size=shape)

class Normal(Distribution):
    """Normal Distribution
    
    """

    def _sample(self, mu, sigma, **kwargs) -> torch.Tensor:
        """Sample from the normal distribution

        Parameters
        ----------
        mu : torch.Tensor
            Mean of the distribution
        sigma : torch.Tensor
            Standard deviation of the distribution

        Returns
        -------
        torch.Tensor
            Sampled tensor
        """
        return torch.normal(mu, sigma)

class TruncatedNormal(Distribution):
    """Truncated Normal Distribution

    Parameters
    ----------
    index : int
        The index of the property to truncate
    
    """

    def __init__(self, index: int = 2, **kwargs) -> None:
        """Initialize the distribution

        """
        super().__init__(**kwargs)
        self.index = index

    def _setup(self, batch: AtomsGraph) -> None:
        """Setup the distribution
        
        Prepare the distribution for sampling of the batch

        Parameters
        ----------
        batch : AtomsGraph
            Batch of data

        Returns
        -------
        None

        """

        self.confinement = batch.confinement[batch.batch]
        self.mask = batch.mask

    def _sample(self, mu, sigma, **kwargs) -> torch.Tensor:
        """Sample from the truncated normal distribution

        Parameters
        ----------
        mu : torch.Tensor
            Mean of the distribution
        sigma : torch.Tensor
            Standard deviation of the distribution

        Returns
        -------
        torch.Tensor
            Sampled tensor

        """
        x = []
        for i in range(mu.shape[1]):
            if i == self.index:
                if mu[:, i].isnan().any():
                    raise ValueError("NaN mean (probably position) values.\n" +
                                     "See troubleshooting in the documentation:\n" +
                                     "https://agedi.readthedocs.io/en/latest/troubleshooting.html")
                    
                sampled = TN(
                    mu[:, i][~self.mask],
                    sigma[:, 0][~self.mask],
                    self.confinement[:, 0][~self.mask],
                    self.confinement[:, 1][~self.mask],
                ).sample()

                xi = torch.zeros_like(mu[:, i])
                xi[~self.mask] = sampled
                x.append(xi)
            else:
                x.append(torch.normal(mu[:, i], sigma[:, 0]))
        return torch.stack(x, dim=1)

class WrappedNormal(Distribution):
    pass

class Uniform(Distribution):
    """Uniform Distribution

    Parameters
    ----------
    low : float
        The lower bound of the distribution
    high : float
        The upper bound of the distribution
    
    """

    def __init__(self, low: float = 0.0, high: float = 1.0) -> None:
        """Initialize the distribution

        """
        self.low = low
        self.high = high

    def _sample(self, mu, sigma) -> torch.Tensor:
        """
        Sample from the uniform distribution

        Parameters
        ----------
        mu : torch.Tensor
            Mean of the distribution
        sigma : torch.Tensor
            Standard deviation of the distribution

        Returns
        -------
        torch.Tensor
            Sampled tensor

        """
        shape = self.shape if hasattr(self, "shape") else mu.shape
        return torch.rand(shape) * (self.high - self.low) + self.low

class UniformCell(Uniform):
    """
    Uniform Prior Distribution for cell parameters
    """

    def _setup(self, batch: AtomsGraph) -> None:
        """
        Prepare the distribution for sampling of the batch

        Parameters
        ----------
        batch : AtomsGraph
            Batch of data

        Returns
        -------
        None

        """
        self.cell = batch.cell.clone()
        if batch.batch is not None:
            self.cell = self.cell.view(-1, 3, 3)[batch.batch]
            self.shape = (batch.x.shape[0], 3, 1)
            self.corner = torch.zeros(self.cell.shape[0], 3)
            
        else:
            self.shape = (batch.x.shape[0], 3)
            self.corner = torch.zeros(1, 3)
            
    def _sample(self, mu, sigma) -> torch.Tensor:
        """Sample from the uniform distribution

        Parameters
        ----------
        mu : torch.Tensor
            Mean of the distribution
        sigma : torch.Tensor
            Standard deviation of the distribution

        Returns
        -------
        torch.Tensor
            Sampled tensor
        
        """
        f = super()._sample(mu, sigma)  # (n_atoms, 3)
        if self.cell.shape[0]  == f.shape[0]:
            r = torch.matmul(self.cell, f).view((self.shape[0], self.shape[1])) + self.corner  # (n_atoms, 3)
        else:
            r = f @ self.cell + self.corner

        return r

class UniformCellConfined(UniformCell):
    """
    Uniform Prior Distribution for cell parameters with Z-directional confinement
    """

    def _setup(self, batch: AtomsGraph) -> None:
        """
        Prepare the distribution for sampling of the batch

        Parameters
        ----------
        batch : AtomsGraph
            Batch of data

        Returns
        -------
        None

        """
        super()._setup(batch)
        self.confinement = batch.confinement
        if batch.batch is not None:
            raise NotImplementedError("Batched version not implemented")
        else:
            z_dist = self.confinement[:, 1] - self.confinement[:, 0]
            z_min = self.confinement[:, 0]
            self.cell[2, :2] = torch.tensor([0.0, 0.0])
            self.cell[2,2] = z_dist
            self.corner[0, 2] = z_min

class NormalConfined(UniformCellConfined):
    """Sample from the normal distribution with confinenement 

    Parameters
    ----------
    mu : torch.Tensor
        Mean of the distribution
    sigma : torch.Tensor
        Standard deviation of the distribution

    Returns
    -------
    torch.Tensor
        Sampled tensor

    """
    def _sample(self, mu, sigma) -> torch.Tensor:
        r = super()._sample(mu, sigma)  # (n_atoms, 3)
        if sigma is None:
            sigma = 1.0
        r[:, :2] = torch.normal(r[:, :2], sigma)
        return r
        
            
##### TYPE DIFFUSION STUFF #####
class Constant(Distribution):
    """Integer Distribution

    Parameters
    ----------
    low : int
        The lower bound of the distribution
    high : int
        The upper bound of the distribution
    
    """

    def __init__(self, value: float=0, dtype: Type=torch.int64) -> None:
        """Initialize the distribution

        """
        self.value = value
        self.dtype = dtype

    def _setup(self, batch: AtomsGraph) -> None:
        self.shape = batch.x.shape 

    def _sample(self, mu, sigma) -> torch.Tensor:
        """
        Sample from the integer distribution

        Parameters
        ----------
        mu : torch.Tensor
            Mean of the distribution
        sigma : torch.Tensor
            Standard deviation of the distribution

        Returns
        -------
        torch.Tensor
            Sampled tensor

        """
        shape = self.shape if hasattr(self, "shape") else mu.shape
        return torch.ones(shape, dtype=self.dtype) * self.value

class Categorical(Distribution):
    """Categorical Distribution

    """

    def __init__(self, n_classes: int=100) -> None:
        """Initialize the distribution

        """
        self.n_classes = n_classes


    def _setup(self, batch: AtomsGraph) -> None:
        self.shape = batch.x.shape

    def _sample(self, mu, sigma) -> torch.Tensor:
        """
        Sample from the categorical distribution where
        probabilites define the likelihood of mu value
        to be set to the masked, 0, value

        Parameters
        ----------
        mu : torch.Tensor
            Interpret as the initial values, a_0
        sigma : torch.Tensor
            Interpret as probabilities, a_0 @ Q_t

        Returns
        -------
        torch.Tensor
            Sampled tensor

        """
        m = Cat(sigma).sample()
        return m.view(self.shape)
        

        

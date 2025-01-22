import torch

from typing import Dict
from agedi.data import AtomsGraph
from agedi.diffusion.noisers.base import Noiser

from agedi.diffusion.noisers.distributions import Constant, Categorical
import torch.nn.functional as F
from agedi.diffusion.noisers import SDE, VP, VE



class TypesNoiser(Noiser):
    """Implementation of the type noiser."""

    _key = "x"

    def __init__(self, beta_min: float=0.01, beta_max: float=1.0, prior=Constant(0), distribution=Categorical(), **kwargs) -> None:
        super().__init__(VE, {}, distribution, prior, **kwargs)

        self.beta_min = beta_min
        self.beta_max = beta_max


    def _noise(self, batch: AtomsGraph) -> AtomsGraph:
        """
        """
        print("Types:", batch.x)
        p = self.p(batch.x, batch.time)
        new_x = self.distribution.get_callable(batch)(batch.x, p)
        batch[self.key + "_noise"] =  new_x - batch.x
        batch.x = new_x
        print("Noised Types:", batch.x)
        return batch
        

    def _denoise(self, batch: AtomsGraph) -> AtomsGraph:
        """
        """
        pass

    def _loss(self, batch: AtomsGraph) -> torch.Tensor:
        """
        """
        target = batch[self.key] - batch[self.key + "_noise"]
        pred = batch[self.key + "_score"]

        loss = F.cross_entropy(pred, target)

        print("Predicted Types:", torch.argmax(pred, dim=-1))
        print("Loss:", loss)
        print('-'*20)
        return loss
        

    def p(self, x: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        """
        """
        beta_t = self._beta_t(time)
        p = F.one_hot(x, num_classes=100).float() * (1.0-beta_t)
        p[:, 0] += beta_t.view(-1)
        
        return p

    def _beta_t(self, time):
        """
        Beta function for the type noiser Q
        """
        return self.beta_min + (self.beta_max - self.beta_min) * time
        
        

import torch

from abc import ABC, abstractmethod
from typing import Dict
from agedi.data import AtomsGraph
from agedi.diffusion.noisers import Noiser

from agedi.diffusion.sdes import SDE
from agedi.diffusion.distributions import Distribution


class SDENoiser(Noiser, ABC):
    """Implements a SDE base class that can be inherited by other classes.

    Parameters
    ----------
    sde_class : SDE
        The class of the SDE to be used for the noising.
    sde_kwargs : Dict
        The keyword arguments to be passed to the SDE class.
    distribution : Distribution
        The distribution to be used for the noise.
    prior : Distribution
        The prior distribution to be used for the noise.
    key : str
        The key to be used for the noising.
    **kwargs
        Additional keyword arguments to be passed to the Noiser class.

    Returns
    -------
    Noiser
        The noiser for the atoms positions in Cartesian coordinates.

    """

    _key = None

    def __init__(
        self,
        sde_class: SDE,
        sde_kwargs: Dict,
        distribution: Distribution,
        prior: Distribution,
        **kwargs
    ) -> None:
        super().__init__(distribution, prior, **kwargs)
        self.sde = sde_class(**sde_kwargs)


    @abstractmethod
    def postprocess_score(self, score: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def postprocess_noise(self, noise: torch.Tensor) -> torch.Tensor:
        pass

    def _noise(self, batch: AtomsGraph) -> AtomsGraph:
        """Adds noise to the atomistic structure.

        Added noise is stored in the self.key+"_noise".

        Parameters
        ----------
        batch: AtomsGraph
            The atomistic structure (or batch hereof) to be noised.

        Returns
        -------
        AtomsGraph
            The noised atomistic structure (or bach hereof).

        """
        z = batch[self.key]
        t = batch.time

        w = self.distribution.get_callable(batch)
        batch[self.key] = self.sde.transition_kernel(z, t, w)
        batch[self.key + "_noise"] = batch.apply_mask(self.sde.noise(z, batch[self.key], t))

        return batch

    def _denoise(self, batch: AtomsGraph, delta_t: float, last: bool) -> AtomsGraph:
        """Denoises the positions of the atomistic structure.

        The denoising follows the Euler-Maruyama scheme.
        ::math::
        R_i+1 = R_i +
                \Delta t (f(R_i, t) + g(t)**2 * s(R_i, t)) +
                \sqrt{\Delta t} g(t) * w

        The used score is expected to be stored in the self.key+"_score".


        Parameters
        ----------
        batch: AtomsGraph
            The atomistic structure (or batch hereof) to be denoised.
        delta_t: float
            The time step for the denoising.
        last: bool
            If the denoising is the last step of the denoising.

        Returns
        -------
        AtomsGraph
            The denoised atomistic structure (or bach hereof).

        """
        z = batch[self.key]
        z_score = batch[self.key + "_score"]
        t = batch.time

        drift = self.sde.drift(z, t)
        diffusion = self.sde.diffusion(t)

        w = self.distribution.get_callable(batch)
        if last:
            batch[self.key] = batch[self.key] + delta_t * (diffusion**2 * z_score + drift)
        else:
            batch[self.key] = w(
                batch[self.key] + delta_t * (diffusion**2 * z_score + drift),  # mean
                torch.sqrt(delta_t) * diffusion,  # variance
            )

        return batch

    def _loss(self, batch: AtomsGraph) -> torch.Tensor:
        """Compute the noiser loss.

        Computes the loss of the diffusion model SDE noiser

        Expects the total added noise to be stored in the self.key+"_noise",
        and the predicted score to be stored in the
        self.key+"_score".

        The loss is computed as
        ::math::
        L = \sum_i ||\sigma_t w_i + \sigma_t^2 s(R_i)||^2

        Parameters
        ----------
        batch: AtomsGraph
            The atomistic structure (or batch hereof) to be noised and denoised.

        Returns
        -------
        float
            The loss of the noised and denoised atomistic structure.

        """
        t = batch.time
        z_score = batch[self.key + "_score"]
        z_noise = batch[self.key + "_noise"]

        var = self.sde.var(t)

        z_score = self.postprocess_score(z_score) #batch.apply_mask(r_score)
        z_noise = self.postprocess_noise(z_noise)
        # r_noise = self.periodic_distance(batch.pos, r_noise, batch.cell, batch.batch)

        lt = 1.0  # /var.sqrt()

        loss = torch.mean(
            lt * torch.sum((z_noise + z_score * var) ** 2, dim=-1, keepdim=True)
        )
        return loss


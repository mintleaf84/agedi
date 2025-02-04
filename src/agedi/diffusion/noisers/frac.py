import torch

from typing import Dict
from agedi.data import AtomsGraph
from agedi.diffusion.noisers import Noiser
from agedi.diffusion.sdes import SDE, VE
from agedi.diffusion.distributions import Distribution, Uniform, WrappedNormal
from agedi.utils import OFFSET_LIST


class FractionalNoiser(Noiser):
    """Implements noising of atoms positions in fractional coordinates.

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
        The noiser for the atoms positions in fractional coordinates.

    """

    _key = "frac"

    def __init__(
        self,
        sde_class: SDE = VE,
        sde_kwargs: Dict = {},
        distribution: Distribution = WrappedNormal(),
        prior: Distribution = Uniform(),
        **kwargs
    ) -> None:
        super().__init__(distribution, prior, **kwargs)
        self.sde = sde_class(**sde_kwargs)

    def _noise(self, batch: AtomsGraph) -> AtomsGraph:
        """Initializes the noise for the positions noiser.

        Added noise is stored in the self.key+"_noise", which by default is
        "positions_noise".

        Parameters
        ----------
        batch: AtomsGraph
            The atomistic structure (or batch hereof) to be noised.

        Returns
        -------
        AtomsGraph
            The noised atomistic structure (or bach hereof).

        """
        f = batch[self.key]
        t = batch.time

        w = self.distribution.get_callable(batch)
        batch.pos = self.sde.transition_kernel(f, t, w)
        batch[self.key + "_noise"] = batch.apply_mask(self.sde.noise(f, batch.pos, t))

        return batch

    def _denoise(self, batch: AtomsGraph, delta_t: float, last: bool) -> AtomsGraph:
        """Denoises the positions of the atomistic structure.

        The denoising follows the Euler-Maruyama scheme.
        ::math::
        R_i+1 = R_i +
                \Delta t (f(R_i, t) + g(t)**2 * s(R_i, t)) +
                \sqrt{\Delta t} g(t) * w

        The used score is expected to be stored in the self.key+"_score",
        which by default is "pos_score".

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
        f = batch[self.key]
        f_score = batch[self.key + "_score"]
        t = batch.time

        drift = self.sde.drift(f, t)
        diffusion = self.sde.diffusion(t)

        w = self.distribution.get_callable(batch)
        if last:
            batch[self.key] = batch[self.key] + delta_t * (diffusion**2 * f_score + drift)
        else:
            batch[self.key] = w(
                batch[self.key] + delta_t * (diffusion**2 * f_score + drift),  # mean
                torch.sqrt(delta_t) * diffusion,  # variance
            )

        return batch

    def _loss(self, batch: AtomsGraph) -> torch.Tensor:
        """Compute the noiser loss.

        Computes the loss of the diffusion model for the positions noiser

        Expects the total added positions noise to be stored in the self.key+"_noise",
        which by default is "pos_noise" and the predicted score to be stored in the
        self.key+"_score", which by default is "pos_score".

        The loss is computed as
        ::math::
        L = \sum_i ||\sigma_t w_i + \sigma_t^2 s(R_i)||^2

        With the noise taking into account periodic boundary conditions.

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
        f_score = batch[self.key + "_score"]
        f_noise = batch[self.key + "_noise"]

        var = self.sde.var(t)

        f_score = batch.apply_mask(f_score)

        lt = 1.0

        loss = torch.mean(
            lt * torch.sum((f_noise + f_score * var) ** 2, dim=-1, keepdim=True)
        )
        return loss


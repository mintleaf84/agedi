import torch

from typing import Dict
from agedi.data import AtomsGraph
from agedi.diffusion.noisers import Noiser
from agedi.diffusion.sdes import SDE, VE
from agedi.diffusion.distributions import Distribution, Normal, StandardNormal


class CellNoiser(Noiser):
    """Implements noising of the cell.

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

    _key = "cell"

    def __init__(
        self,
        sde_class: SDE = VE,
        sde_kwargs: Dict = {},
        distribution: Distribution = Normal(),
        prior: Distribution = StandardNormal(),
        **kwargs
    ) -> None:
        super().__init__(distribution, prior, **kwargs)
        self.sde = sde_class(**sde_kwargs)

    def _noise(self, batch: AtomsGraph) -> AtomsGraph:
        """Initializes the noise for the cell noiser.

        Added noise is stored in the self.key+"_noise", which by default is
        "cell_noise".

        Parameters
        ----------
        batch: AtomsGraph
            The atomistic structure (or batch hereof) to be noised.

        Returns
        -------
        AtomsGraph
            The noised atomistic structure (or bach hereof).

        """
        c = batch[self.key]
        t = batch.time

        w = self.distribution.get_callable(batch)
        batch[self.key] = self.sde.transition_kernel(c, t, w)
        batch[self.key + "_noise"] = self.sde.noise(c, batch[self.key], t)

        return batch

    def _denoise(self, batch: AtomsGraph, delta_t: float, last: bool) -> AtomsGraph:
        """Denoises the cell of the atomistic structure.

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
        c = batch[self.key]
        c_score = batch[self.key + "_score"]
        t = batch.time

        drift = self.sde.drift(c, t)
        diffusion = self.sde.diffusion(t)

        w = self.distribution.get_callable(batch)
        if last:
            batch.pos = batch[self.key] + delta_t * (diffusion**2 * c_score + drift)
        else:
            batch.pos = w(
                batch[self.key] + delta_t * (diffusion**2 * c_score + drift),  # mean
                torch.sqrt(delta_t) * diffusion,  # variance
            )

        return batch

    def _loss(self, batch: AtomsGraph) -> torch.Tensor:
        """Compute the noiser loss.

        Computes the loss of the diffusion model for the cell noiser

        Expects the total added cell noise to be stored in the self.key+"_noise",
        which by default is "cell_noise" and the predicted score to be stored in the
        self.key+"_score", which by default is "cell_score".

        The loss is computed as
        ::math::
        L = \sum_i ||\sigma_t w_i + \sigma_t^2 s(C_i)||^2

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
        c_score = batch[self.key + "_score"]
        c_noise = batch[self.key + "_noise"]

        var = self.sde.var(t)

        lt = 1.0  # /var.sqrt()

        loss = torch.mean(
            lt * torch.sum((c_noise + c_score * var) ** 2, dim=-1, keepdim=True)
        )
        return loss


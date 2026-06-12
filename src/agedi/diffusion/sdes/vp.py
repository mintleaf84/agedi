import torch
from typing import Dict, Optional, Type, Union
from agedi.diffusion.sdes import SDE
from .noise_schedules import NoiseSchedule, Cosine


class VP(SDE):
    """Implements variance-preserving (VP) SDE.

    Parameters
    ----------
    beta_min: float
        The minimum value of the beta parameter (noise schedule at t=0).
    beta_max: float
        The maximum value of the beta parameter (noise schedule at t=1).
    noise_schedule : NoiseSchedule instance or class, optional
        Noise schedule to use.  Can be an already-instantiated
        :class:`~agedi.diffusion.sdes.NoiseSchedule` (e.g. from a Hydra
        round-trip) or the class itself.  When *None* (default),
        :class:`~agedi.diffusion.sdes.Cosine` is used with ``beta_min``
        and ``beta_max``.

    Returns
    -------
    VP

    """

    def __init__(
        self,
        beta_min: float = 0.1,
        beta_max: float = 20.0,
        noise_schedule: Optional[Union[NoiseSchedule, Type[NoiseSchedule]]] = None,
    ):
        """Initializes the VP SDE."""
        super().__init__()
        if noise_schedule is None or isinstance(noise_schedule, type):
            schedule_cls = noise_schedule if noise_schedule is not None else Cosine
            self.noise_schedule = schedule_cls(beta_min, beta_max)
        else:
            self.noise_schedule = noise_schedule
        self.beta_min = self.noise_schedule.min
        self.beta_max = self.noise_schedule.max

    def get_hparams(self) -> Dict:
        """Return hyperparameters for this VP SDE."""
        return {
            **super().get_hparams(),
            "noise_schedule": self.noise_schedule.get_hparams(),
        }

    def drift(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Implement VP drift term.

        Defines the drift term of the SDE: f(x, t).

        Parameters
        ----------
        x: torch.Tensor
            The positions of the atoms.
        t: torch.Tensor
            The time at which to calculate the drift term.

        Returns
        -------
        drift: torch.Tensor
            The drift term of the SDE.

        """
        return -0.5 * self.beta(t) * x

    def diffusion(self, t: torch.Tensor) -> torch.Tensor:
        """Implement VP diffusion term.

        Defines the diffusion term of the SDE: g(t).

        Parameters
        ----------
        t: torch.Tensor
            The time at which to calculate the diffusion term.

        Returns
        -------
        diffusion: torch.Tensor
            The diffusion term of the SDE.

        """
        return torch.sqrt(self.beta(t))

    def mean(self, t: torch.Tensor) -> torch.Tensor:
        """Implement VP mean term.

        Calculates the mean of transition kernel at time t: mu(t).

        Parameters
        ----------
        t: torch.Tensor
            The time at which to calculate the mean.

        Returns
        -------
        mean: torch.Tensor
            The mean of the diffusion process.

        """
        return torch.exp(-0.5 * self.alpha(t))

    def var(self, t: torch.Tensor) -> torch.Tensor:
        """Implement VP variance term.

        Calculates the variance of transition kernel at time t: sigma^2(t).

        Parameters
        ----------
        t: torch.Tensor
            The time at which to calculate the variance.

        Returns
        -------
        var: torch.Tensor
            The variance of the diffusion process.

        """
        return 1 - torch.exp(-self.alpha(t))

    def beta(self, t: torch.Tensor) -> torch.Tensor:
        """VP Beta function

        Calculates the value of beta at time t.

        Parameters
        ----------
        t: torch.Tensor
            The time at which to calculate beta.

        Returns
        -------
        beta: torch.Tensor
            The value of beta at time t.

        """

        return self.noise_schedule.f(t)

    def alpha(self, t: torch.Tensor) -> torch.Tensor:
        """VP Alpha function

        Calculates the value of alpha at time t with
        .. math::
        \\alpha(t) = int_{0}^{t} beta(s) ds.

        Parameters
        ----------
        t: torch.Tensor
            The time at which to calculate alpha.

        Returns
        -------
        alpha: torch.Tensor
            The value of alpha at time t.

        """
        return self.noise_schedule.fint(t)

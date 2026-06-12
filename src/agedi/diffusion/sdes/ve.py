import torch
from typing import Dict, Optional, Type, Union
from agedi.diffusion.sdes import SDE
from .noise_schedules import NoiseSchedule, Linear


class VE(SDE):
    """Implements variance-exploding (VE) SDE.

    Parameters
    ----------
    sigma_min: float
        The minimum value of the sigma parameter (noise schedule at t=0).
    sigma_max: float
        The maximum value of the sigma parameter (noise schedule at t=1).
    noise_schedule : NoiseSchedule instance or class, optional
        Noise schedule to use.  Can be an already-instantiated
        :class:`~agedi.diffusion.sdes.NoiseSchedule` (e.g. from a Hydra
        round-trip) or the class itself.  When *None* (default),
        :class:`~agedi.diffusion.sdes.Linear` is used with ``sigma_min``
        and ``sigma_max``.

    Returns
    -------
    VE

    """

    def __init__(
        self,
        sigma_min: float = 1e-2,
        sigma_max: float = 1.0,
        noise_schedule: Optional[Union[NoiseSchedule, Type[NoiseSchedule]]] = None,
    ):
        """Initializes the VE SDE."""
        super().__init__()
        if noise_schedule is None or isinstance(noise_schedule, type):
            schedule_cls = noise_schedule if noise_schedule is not None else Linear
            self.noise_schedule = schedule_cls(sigma_min, sigma_max)
        else:
            self.noise_schedule = noise_schedule
        self.sigma_min = self.noise_schedule.min
        self.sigma_max = self.noise_schedule.max

    def get_hparams(self) -> Dict:
        """Return hyperparameters for this VE SDE."""
        return {
            **super().get_hparams(),
            "noise_schedule": self.noise_schedule.get_hparams(),
        }

    def drift(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Implement VE drift term.

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
        return torch.zeros_like(x)

    def diffusion(self, t: torch.Tensor) -> torch.Tensor:
        """Implement VE diffusion term.

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
        return torch.sqrt(self.noise_schedule.df2dt(t))

    def mean(self, t: torch.Tensor) -> torch.Tensor:
        """Implement VE mean term.

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
        return torch.ones_like(t)

    def var(self, t: torch.Tensor) -> torch.Tensor:
        """Implement VE variance term.

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
        return self.sigma(t) ** 2 - self.sigma(0) ** 2

    def sigma(self, t: torch.Tensor) -> torch.Tensor:
        """VE sigma function

        Calculates the value of sigma at time t.

        Parameters
        ----------
        t: torch.Tensor
            The time at which to calculate sigma.

        Returns
        -------
        sigma: torch.Tensor
            The value of sigma at time t.

        """

        return self.noise_schedule.f(t)

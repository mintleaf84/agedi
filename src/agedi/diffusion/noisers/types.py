import torch
import torch.nn.functional as F
import numpy as np

from agedi.data import AtomsGraph
from agedi.diffusion.noisers.base import Noiser

from agedi.diffusion.noisers.distributions import Constant, Categorical


class NoiseSchedule:
    def __init__(self, beta_min, beta_max):
        super().__init__()
        self.beta_min = beta_min
        self.beta_max = beta_max

    def _beta_t(self, time):
        """
        Beta function for the type noiser Q
        """
        return self.beta_min + (self.beta_max - self.beta_min) * time

    def rate_noise(self, t):
        return (
            self.beta_min ** (1 - t)
            * self.beta_max**t
            * (np.log(self.beta_max) - np.log(self.beta_min))
        )

    def total_noise(self, t):
        return self.beta_min ** (1 - t) * self.beta_max**t

class Transition:
    pass

class TypesNoiser(Noiser):
    """Type Noiser

    Based on score entropy and discrete diffusion model.
    See https://arxiv.org/abs/2310.16834 for more details

    """

    _key = "x"

    def __init__(
        self,
        prior=Constant(0),
        distribution=Categorical(),
        noise_schedule: NoiseSchedule = NoiseSchedule(0.01, 3.0),
        **kwargs
    ) -> None:
        super().__init__(distribution=distribution, prior=prior, **kwargs)

        self.noise_schedule = noise_schedule

    def _noise(self, batch: AtomsGraph) -> AtomsGraph:
        """Noises the attribute of the atomistic structure.

        Performs the noising of the atomic types.

        Parameters
        ----------
        batch: AtomsGraph
            The atomistic structure (or batch hereof) to be noised.

        Returns
        -------
        AtomsGraph
            The noised atomistic structure (or bach hereof).

        """
        time = batch.time
        sigma = self.noise_schedule.total_noise(time)
        types = batch[self.key]
        noised_types = self.sample_transition(types, sigma.reshape(-1))

        batch[self.key + "_noise"] = noised_types - types
        batch[self.key] = noised_types

        return batch

    def _denoise(self, batch: AtomsGraph, delta_t: float, last: bool) -> AtomsGraph:
        """Denoises the attribute of the atomistic structure.

        Denoisis the atomic types.

        Parameters
        ----------
        batch: AtomsGraph
            The atomistic structure (or batch hereof) to be denoised.
        delta_t: float
            The time step to be used for the denoising.
        last: bool
            If the last denoising step is performed.

        Returns
        -------
        AtomsGraph
            The denoised atomistic structure (or bach hereof).



        """
        types = batch[self.key]
        score = batch[self.key + "_score"].exp()
        time = batch.time

        sigma = self.noise_schedule.total_noise(time)
        dsigma = self.noise_schedule.rate_noise(time)
        dist = self.distribution.get_callable(batch)

        if last:
            stag_score = self.staggered_score(score, sigma)
            probs = stag_score * self.transp_transition(types, sigma)
            probs[:, 0] = 0.0  # no adsorb states
            new_types = dist(probs)
        else:
            rev_rate = delta_t * dsigma * self.reverse_rate(types, score)
            new_types = self.sample_rate(dist, types, rev_rate)

        batch[self.key] = new_types
        return batch

    def _loss(self, batch: AtomsGraph) -> torch.Tensor:
        """Computes the training loss.

        The score is with score entropy training as thus given as score=log(s) and then
        for sampling should be used as a concrete score i.e. exp(score)!

        Parameters
        ----------
        batch: AtomsGraph
            The atomistic structure (or batch hereof) to be noised and denoised.

        Returns
        -------
        float
            The loss of the noised and denoised atomistic structure.


        """
        types = batch[self.key] - batch[self.key + "_noise"]
        noised_types = batch[self.key]
        score = batch[self.key + "_score"]  # am I missing a exp here?

        time = batch.time
        sigma = self.noise_schedule.total_noise(time)
        dsigma = self.noise_schedule.rate_noise(time)

        losses = self.score_entropy(score, sigma, noised_types, types)
        loss = (dsigma * losses).mean()

        return loss

    def sample_transition(self, x, sigma):
        move_chance = 1 - (-sigma).exp()
        move_indices = torch.rand(*x.shape, device=x.device) < move_chance
        x_pert = torch.where(move_indices, 0, x)
        return x_pert

    def score_entropy(self, score, sigma, x, x0):
        rel_ind = x == 0
        esigm1 = torch.where(sigma < 0.5, torch.expm1(sigma), torch.exp(sigma) - 1)

        # ratio = 1 / esigm1.expand_as(x)[rel_ind]
        ratio = 1 / esigm1[rel_ind]
        other_ind = x0[rel_ind]

        # negative_term
        neg_term = ratio * torch.gather(score[rel_ind], -1, other_ind[..., None])

        # positive term
        pos_term = score[rel_ind][:, 1:].exp().sum(dim=-1)

        # constant term
        const = ratio * (ratio.log() - 1)

        entropy = torch.zeros(*x.shape, device=x.device)

        entropy[rel_ind] += pos_term - neg_term.reshape(-1) + const.reshape(-1)
        return entropy

    def transp_rate(self, i):
        edge = -F.one_hot(i, num_classes=100)
        edge[i == 0] += 1
        return edge

    def reverse_rate(self, i, score):
        """
        Constructs the reverse rate. Which is score * transp_rate
        """
        normalized_rate = self.transp_rate(i) * score

        normalized_rate.scatter_(-1, i[..., None], torch.zeros_like(normalized_rate))
        normalized_rate.scatter_(
            -1, i[..., None], -normalized_rate.sum(dim=-1, keepdim=True)
        )

        return normalized_rate

    def sample_rate(self, callable, i, rate):
        return callable(F.one_hot(i, num_classes=100).to(rate) + rate)

    def staggered_score(self, score, dsigma):
        score = score.clone()  # yeah yeah whatever we should probably do this
        extra_const = (1 - (dsigma).exp()) * score.sum(dim=-1, keepdim=True)
        score *= dsigma.exp()
        score[..., 0] += extra_const.squeeze(-1)
        return score

    def transp_transition(self, i, sigma):
        # sigma = unsqueeze_as(sigma, i[..., None])
        edge = (-sigma).exp() * F.one_hot(i, num_classes=100)
        edge += torch.where(i == 0, 1 - (-sigma).squeeze(-1).exp(), 0)[..., None]
        return edge

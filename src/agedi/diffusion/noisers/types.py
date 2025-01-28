import torch
import numpy as np

from typing import Dict
from agedi.data import AtomsGraph
from agedi.diffusion.noisers.base import Noiser

from agedi.diffusion.noisers.distributions import Constant, Categorical
import torch.nn.functional as F
from agedi.diffusion.noisers import SDE, VP, VE, SED

def sample_categorical(categorical_probs, method="hard"):
    if method == "hard":
        gumbel_norm = 1e-10 - (torch.rand_like(categorical_probs) + 1e-10).log()
        return (categorical_probs / gumbel_norm).argmax(dim=-1)
    else:
        raise ValueError(f"Method {method} for sampling categorical variables is not valid.")



class TypesNoiser(Noiser):
    """Implementation of the type noiser.

    for now the distribution is not keep independent (so all happens in this class!)
    
    """

    _key = "x"

    def __init__(self, beta_min: float=0.01, beta_max: float=3.0, prior=Constant(0), distribution=Categorical(), **kwargs) -> None:
        super().__init__(SED, {}, distribution, prior, **kwargs)

        self.beta_min = beta_min
        self.beta_max = beta_max


    def _noise(self, batch: AtomsGraph) -> AtomsGraph:
        """
        this does the noising (forward) step

        !!!Implement!!!
        """
        # print("Types:", batch.x)

        time = batch.time
        sigma, dsigma = self.total_noise(time), self.rate_noise(time)

        noised_x = self.sample_transition(batch.x, sigma.reshape(-1))

        # batch.x = noised_x
        
        # breakpoint()
        # p = self.p(batch.x, batch.time)
        # new_x = self.distribution.get_callable(batch)(batch.x, p)
        batch[self.key + "_noise"] =  noised_x - batch[self.key]
        batch.x = noised_x
        # print("Noised Types:", batch.x)
        return batch
        

    def _denoise(self, batch: AtomsGraph, delta_t: float, last: bool) -> AtomsGraph:
        """
        !!!Implement!!!

        - This performs a single denoising step.
        
        """
        x = batch[self.key]
        score = batch[self.key + "_score"].exp()
        time = batch.time
        sigma, dsigma = self.total_noise(time), self.rate_noise(time)

        if last:
            stag_score = self.staggered_score(score, sigma)
            probs = stag_score * self.transp_transition(x, sigma)
            probs[:, 0] = 0.0   # no adsorption
            new_x = sample_categorical(probs)
        else:
            rev_rate = delta_t * dsigma * self.reverse_rate(x, score)
            new_x = self.sample_rate(x, rev_rate)
        
        batch[self.key] = new_x
        return batch

    def _loss(self, batch: AtomsGraph) -> torch.Tensor:
        """
        This does the loss step.

        The score is trained as a score=log(s) and then for sampling should be exp(score)!
        

        !!!Implement!!!
        """
        noised_target = batch[self.key]
        target = batch[self.key] - batch[self.key + "_noise"]
        score = batch[self.key + "_score"]

        time = batch.time
        sigma, dsigma = self.total_noise(time), self.rate_noise(time)
        # breakpoint()

        losses = self.score_entropy(score, sigma, noised_target, target)

        loss = (dsigma*losses).mean()
        # loss = F.cross_entropy(pred, target)

        # print("Predicted Types:", torch.argmax(pred, dim=-1))
        # print("Loss:", loss)
        # print(score.exp().argmax(dim=-1))
        # print(target)
        # print(noised_target)
        # print('-'*30)
        # if loss.item() < 500.0 and loss.item() > -0.0:
        #     breakpoint()
        # print('-'*20)
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
        
    def rate_noise(self, t):
        return self.beta_min ** (1 - t) * self.beta_max ** t * (np.log(self.beta_max) - np.log(self.beta_min))

    def total_noise(self, t):
        return self.beta_min ** (1 - t) * self.beta_max ** t

    def sample_transition(self, i, sigma):
        move_chance = 1 - (-sigma).exp()
        move_indices = torch.rand(*i.shape, device=i.device) < move_chance
        i_pert = torch.where(move_indices, 0, i)
        return i_pert

    def score_entropy(self, score, sigma, x, x0):
        rel_ind = x == 0
        esigm1 = torch.where(
            sigma < 0.5,
            torch.expm1(sigma),
            torch.exp(sigma) - 1
        )

        # ratio = 1 / esigm1.expand_as(x)[rel_ind]
        ratio = 1 / esigm1[rel_ind]
        other_ind = x0[rel_ind]

        # negative_term
        neg_term = ratio * torch.gather(score[rel_ind], -1, other_ind[..., None])

        #positive term
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
    
    def sample_rate(self, i, rate):
        return sample_categorical(F.one_hot(i, num_classes=100).to(rate) + rate)

    def staggered_score(self, score, dsigma):
        score = score.clone()  # yeah yeah whatever we should probably do this
        extra_const = (1 - (dsigma).exp()) * score.sum(dim=-1, keepdim=True)
        score *= dsigma.exp()
        score[..., 0] += extra_const.squeeze(-1)
        return score

    def transp_transition(self, i, sigma):
        # sigma = unsqueeze_as(sigma, i[..., None])
        edge = (-sigma).exp() * F.one_hot(i, num_classes=100)
        edge += torch.where(i == 0, 1 - (-sigma).squeeze(-1).exp(), 0)[
            ..., None
        ]
        return edge
    

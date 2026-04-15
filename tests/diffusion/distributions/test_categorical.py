"""Tests for the Categorical distribution."""
import torch
from agedi.diffusion.distributions.categorical import Categorical


def test_categorical_sample_valid_class():
    d = Categorical()
    probs = torch.zeros(5, 10)
    probs[:, 3] = 1.0
    out = d._sample(probs)
    assert out.shape == (5,)
    assert (out == 3).all()


def test_categorical_get_callable_returns_tensor(batch):
    d = Categorical()
    fn = d.get_callable(batch)
    probs = torch.softmax(torch.randn((batch.num_nodes, 100)), dim=-1)
    out = fn(probs)
    assert out.shape == (batch.num_nodes,)
    assert ((out >= 0) & (out < 100)).all()

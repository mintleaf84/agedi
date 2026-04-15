import torch

from agedi.data import Representation
from agedi.models.conditionings import IntegerConditioning


def test_integer_conditioning_get_conditioning_shape():
    conditioning = IntegerConditioning(property="x")
    x = torch.tensor([[1], [2], [3]])

    c = conditioning.get_conditioning(x)

    assert c.shape == (3, 1, conditioning.output_dim)


def test_integer_conditioning_get_empty_conditioning_shape():
    conditioning = IntegerConditioning(property="x")

    c = conditioning.get_empty_conditioning(4)

    assert c.shape == (4, conditioning.output_dim)
    assert torch.allclose(c, torch.zeros_like(c))


def test_integer_conditioning_forward_concatenates_to_representation(batch):
    conditioning = IntegerConditioning(property="x")
    conditioning.sample_mode()
    n = batch.num_nodes
    batch.representation = Representation(scalar=torch.zeros((n, 2, 1)))

    out = conditioning(batch, empty=False)

    assert out.representation.scalar.shape[1] == 2 + conditioning.output_dim


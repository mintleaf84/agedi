"""Tests for TypesScore and PositionsScore schnetpack heads."""
import torch
import pytest

from agedi.data import Representation
from agedi.models.schnetpack.heads import TypesScore, PositionsScore
from agedi.models.schnetpack.translator import SchNetPackTranslator


def _make_translated_batch(batch, d=32):
    N = batch.pos.shape[0]
    translator = SchNetPackTranslator()
    rep = Representation(
        scalar=torch.rand((N, d, 1)),
        vector=torch.rand((N, d, 3)),
    )
    batch.representation = rep
    return translator(batch)


class TestTypesScore:
    def test_init_default(self):
        head = TypesScore()
        assert head.key == "x"

    def test_forward_shape(self, batch):
        N = batch.pos.shape[0]
        d = 32
        head = TypesScore(input_dim_scalar=d)
        translated = _make_translated_batch(batch, d)
        out = head(translated)
        assert out.shape == (N, 100)

    def test_forward_output_is_finite(self, batch):
        d = 32
        head = TypesScore(input_dim_scalar=d)
        translated = _make_translated_batch(batch, d)
        out = head(translated)
        assert out.isfinite().all()


class TestPositionsScoreWithClip:
    def test_score_clip_applied(self, batch):
        d = 32
        head = PositionsScore(input_dim_scalar=d, input_dim_vector=d, score_clip=0.0)
        translated = _make_translated_batch(batch, d)
        out = head(translated)
        assert torch.allclose(out, torch.zeros_like(out))

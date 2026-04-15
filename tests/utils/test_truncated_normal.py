"""Tests for TruncatedStandardNormal and TruncatedNormal."""
import torch
import pytest
from agedi.utils.truncated_normal import TruncatedStandardNormal, TruncatedNormal


class TestTruncatedStandardNormal:
    def setup_method(self):
        self.dist = TruncatedStandardNormal(a=-2.0, b=2.0)

    def test_mean_inside_bounds(self):
        m = self.dist.mean
        assert -2.0 <= m.item() <= 2.0

    def test_variance_positive(self):
        assert self.dist.variance.item() > 0

    def test_entropy_finite(self):
        assert self.dist.entropy.isfinite()

    def test_auc_in_zero_one(self):
        assert 0.0 < self.dist.auc.item() <= 1.0

    def test_cdf_monotone(self):
        vals = torch.linspace(-1.9, 1.9, 10)
        cdfs = self.dist.cdf(vals)
        assert (cdfs[1:] >= cdfs[:-1]).all()

    def test_icdf_inverse_of_cdf(self):
        vals = torch.linspace(-1.5, 1.5, 6)
        cdfs = self.dist.cdf(vals)
        recovered = self.dist.icdf(cdfs)
        assert torch.allclose(recovered, vals, atol=1e-5)

    def test_log_prob_finite(self):
        vals = torch.linspace(-1.5, 1.5, 6)
        lp = self.dist.log_prob(vals)
        assert lp.isfinite().all()

    def test_rsample_within_bounds(self):
        samples = self.dist.rsample(torch.Size([100]))
        assert (samples >= -2.0).all()
        assert (samples <= 2.0).all()

    def test_mismatched_dtype_raises(self):
        with pytest.raises(ValueError):
            TruncatedStandardNormal(a=torch.tensor(-1.0), b=torch.tensor(1.0, dtype=torch.float64))

    def test_inverted_bounds_raises(self):
        with pytest.raises(ValueError):
            TruncatedStandardNormal(a=1.0, b=-1.0)


class TestTruncatedNormal:
    def setup_method(self):
        self.dist = TruncatedNormal(loc=0.0, scale=1.0, a=-3.0, b=3.0)

    def test_mean_close_to_loc_for_wide_bounds(self):
        assert abs(self.dist.mean.item()) < 0.1

    def test_variance_positive(self):
        assert self.dist.variance.item() > 0

    def test_rsample_within_bounds(self):
        samples = self.dist.rsample(torch.Size([200]))
        assert (samples >= -3.0).all()
        assert (samples <= 3.0).all()

    def test_cdf_monotone(self):
        vals = torch.linspace(-2.0, 2.0, 8)
        cdfs = self.dist.cdf(vals)
        assert (cdfs[1:] >= cdfs[:-1]).all()

    def test_icdf_inverse(self):
        vals = torch.linspace(-2.0, 2.0, 5)
        cdfs = self.dist.cdf(vals)
        recovered = self.dist.icdf(cdfs)
        assert torch.allclose(recovered, vals, atol=1e-5)

    def test_log_prob_finite(self):
        vals = torch.linspace(-2.0, 2.0, 5)
        lp = self.dist.log_prob(vals)
        assert lp.isfinite().all()

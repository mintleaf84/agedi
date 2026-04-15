import torch
import torch.nn as nn

import agedi.models.schnetpack.regressor_heads as regressor_heads


class DummyBlock(nn.Module):
    def __init__(self, n_sin, n_vin, n_sout, n_vout, n_hidden, activation, sactivation):
        super().__init__()
        self.n_sout = n_sout
        self.n_vout = n_vout
        self.sactivation = sactivation

    def forward(self, x):
        n = x[0].shape[0]
        scalar = torch.zeros((n, self.n_sout))
        vector = torch.zeros((n, 3, self.n_vout))
        return scalar, vector


def test_build_gated_equivariant_mlp(monkeypatch):
    monkeypatch.setattr(regressor_heads.snn, "GatedEquivariantBlock", DummyBlock)

    net = regressor_heads.build_gated_equivariant_mlp(
        s_in=64, v_in=64, n_out=1, n_layers=3
    )

    assert isinstance(net, nn.Sequential)
    assert len(net) == 3
    assert net[-1].sactivation is None


def test_forces_key_and_predict(monkeypatch):
    monkeypatch.setattr(regressor_heads.snn, "GatedEquivariantBlock", DummyBlock)
    model = regressor_heads.Forces(input_dim_scalar=16, input_dim_vector=16, gated_blocks=2)
    batch = {
        "scalar_representation": torch.randn((8, 16)),
        "vector_representation": torch.randn((8, 3, 16)),
    }

    out_predict = model.predict(batch)
    out_call = model(batch)

    assert model.key == "forces"
    assert out_predict.shape == (8, 3)
    assert torch.equal(out_predict, out_call)


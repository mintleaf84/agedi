import pytest
import torch

from agedi.diffusion.noisers import PositionsNoiser

@pytest.fixture(params=['true', 'false'])
def last(request: str) -> bool:
    if request.param == 'true':
        return True
    else:
        return False
        
def test_init_noiser() -> None:
    noiser = PositionsNoiser()
    assert noiser is not None

def test_noise_no_time(batch: "Batch") -> None:
    noiser = PositionsNoiser()
    with pytest.raises(TypeError):
        noiser.noise(batch)
        
def test_noise(batch: "Batch") -> None:
    batch.time = torch.rand((batch.num_graphs, 1))[batch.batch]
    noiser = PositionsNoiser()
    noised = noiser.noise(batch)
    assert "pos_noise" in noised.keys()

def test_denoise_no_score(batch: "Batch", last: bool) -> None:
    noiser = PositionsNoiser()
    with pytest.raises(KeyError):
        noiser.denoise(batch, torch.tensor(0.001), last=last)

def test_denoise(batch: "Batch", last: bool) -> None:
    pos = batch.pos.clone()
    batch.time = torch.rand((batch.num_graphs, 1))[batch.batch]
    batch.pos_score = torch.randn_like(batch.pos)
    noiser = PositionsNoiser()
    noiser.denoise(batch, torch.tensor(0.001), last=last)
    assert not torch.allclose(pos, batch.pos)

def test_loss(batch: "Batch") -> None:
    batch.time = torch.rand((batch.num_graphs, 1))[batch.batch]    
    pos = batch.pos.clone()
    
    noiser = PositionsNoiser()

    noised = noiser.noise(batch)
    noised.pos_score = torch.randn_like(batch.pos)
    l = noiser.loss(noised)
    assert l > 0
        

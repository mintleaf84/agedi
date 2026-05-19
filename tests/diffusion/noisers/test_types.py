import torch

from agedi.diffusion.noisers.types import NoiseSchedule, Types


def test_noise_schedule_methods():
    schedule = NoiseSchedule(beta_min=0.01, beta_max=3.0)
    t = torch.tensor([0.0, 0.5, 1.0])

    beta_t = schedule._beta_t(t)
    rate = schedule.rate_noise(t)
    total = schedule.total_noise(t)

    assert torch.allclose(beta_t, torch.tensor([0.01, 1.505, 3.0]))
    assert rate.shape == t.shape
    assert total.shape == t.shape


def test_types_noiser_sample_transition_output_shape_and_bounds():
    noiser = Types()
    x = torch.tensor([1, 2, 3, 4])
    sigma = torch.tensor([0.1, 0.2, 0.3, 0.4])

    out = noiser.sample_transition(x, sigma)

    assert out.shape == x.shape
    assert ((out == 0) | (out == x)).all()


def test_types_noiser_transp_rate_and_reverse_rate():
    noiser = Types()
    x = torch.tensor([0, 5, 7])
    score = torch.ones((3, 100))

    rate = noiser.transp_rate(x)
    reverse = noiser.reverse_rate(x, score)

    assert rate.shape == (3, 100)
    assert reverse.shape == (3, 100)
    assert torch.allclose(reverse.sum(dim=-1), torch.zeros(3))


def test_types_noiser_staggered_score_and_transition():
    noiser = Types()
    x = torch.tensor([0, 5])
    sigma = torch.tensor([[0.2], [0.4]])
    score = torch.softmax(torch.randn((2, 100)), dim=-1)

    staggered = noiser.staggered_score(score, sigma)
    transition = noiser.transp_transition(x, sigma)

    assert staggered.shape == score.shape
    assert transition.shape == score.shape
    assert torch.isfinite(staggered).all()
    assert torch.isfinite(transition).all()


def test_types_noiser_score_entropy_shape():
    noiser = Types()
    score = torch.randn((4, 100))
    sigma = torch.tensor([[0.2], [0.3], [0.4], [0.5]])
    x = torch.tensor([0, 1, 0, 2])
    x0 = torch.tensor([1, 1, 2, 2])

    out = noiser.score_entropy(score, sigma, x, x0)

    assert out.shape == x.shape
    assert torch.isfinite(out).all()


def test_types_noiser_sample_rate():
    noiser = Types()
    x = torch.tensor([1, 2])
    rate = torch.zeros((2, 100))
    sampler = lambda probs: probs.argmax(dim=-1)

    sampled = noiser.sample_rate(sampler, x, rate)

    assert sampled.shape == x.shape


def test_types_noiser_noise_loss_and_denoise(batch):
    noiser = Types()
    batch.time = torch.rand((batch.num_nodes, 1))
    original_types = batch.x.clone()

    noised = noiser._noise(batch)
    assert "x_noise" in noised.keys()
    assert noised.x.shape == original_types.shape

    noised.x_score = torch.log_softmax(torch.randn((batch.num_nodes, 100)), dim=-1)
    loss = noiser._loss(noised)
    assert torch.isfinite(loss)

    noiser.distribution.get_callable = lambda _: (lambda probs: probs.argmax(dim=-1))
    denoised_last = noiser._denoise(noised, delta_t=0.01, last=True)
    assert denoised_last.x.shape == original_types.shape

    denoised_step = noiser._denoise(noised, delta_t=0.01, last=False)
    assert denoised_step.x.shape == original_types.shape


# ---------------------------------------------------------------------------
# Tests for type_map (compact-index) functionality
# ---------------------------------------------------------------------------


def test_types_noiser_type_map_sets_n_classes():
    """Types noiser should set n_classes = len(type_map) when type_map is given."""
    type_map = [0, 1, 6, 8]  # absorbing, H, C, O
    noiser = Types(type_map=type_map)
    assert noiser.n_classes == 4
    assert noiser._type_map == [0, 1, 6, 8]


def test_types_noiser_type_map_none_uses_default_n_classes():
    """Without type_map, default n_classes=100 is used."""
    noiser = Types()
    assert noiser.n_classes == 100
    assert noiser._type_map is None


def test_types_noiser_type_map_noise_remaps_to_compact_indices(batch):
    """With type_map, _noise() should remap atomic numbers to compact indices."""
    # Only atoms with Z in {1, 6, 8} should be in the batch for this test.
    from agedi.data import AtomsGraph
    from ase.build import molecule

    atoms = molecule("H2O")
    atoms.set_cell([10, 10, 10])
    atoms.set_pbc(True)
    atoms.center()
    graph = AtomsGraph.from_atoms(atoms)

    # H=1, O=8 → compact indices: H→1, O→2 with type_map=[0,1,8]
    type_map = [0, 1, 8]  # absorbing, H, O
    noiser = Types(type_map=type_map)

    graph.time = torch.full((graph.num_nodes, 1), 1.0)  # max noise time

    # Inject sigma so all atoms get absorbed (transition to 0)
    import torch_geometric
    original_x = graph.x.clone()
    noised = noiser._noise(graph)

    # After noising, x values should be compact indices (0 or 1 or 2)
    assert noised.x.max().item() < len(type_map), (
        f"Noised x contains index >= n_classes: {noised.x}"
    )


def test_types_noiser_type_map_denoise_last_remaps_to_atomic_numbers():
    """With type_map, the final _denoise step converts compact indices → atomic numbers."""
    type_map = [0, 1, 6, 8]  # absorbing, H, C, O
    noiser = Types(type_map=type_map)
    n_classes = len(type_map)  # 4

    from agedi.data import AtomsGraph
    from ase.build import molecule
    atoms = molecule("H2O")
    atoms.set_cell([10, 10, 10])
    atoms.set_pbc(True)
    atoms.center()
    graph = AtomsGraph.from_atoms(atoms)

    # Set batch.x to compact indices (1=H, 3=O)
    graph.x = torch.tensor([1, 3, 3], dtype=torch.long)  # H, O, O in compact
    graph.time = torch.full((graph.num_nodes, 1), 0.01)

    # Provide a score that just puts all weight on H (index 1)
    graph.x_score = torch.log_softmax(
        torch.zeros(graph.num_nodes, n_classes).scatter_(1, torch.ones(graph.num_nodes, 1, dtype=torch.long), 10.0),
        dim=-1,
    )

    noiser.distribution.get_callable = lambda _: (lambda probs: probs.argmax(dim=-1))
    denoised = noiser._denoise(graph, delta_t=0.01, last=True)

    # After last step, x should contain ATOMIC NUMBERS, not compact indices
    valid_atomic_numbers = set(type_map[1:])  # {1, 6, 8}
    for z in denoised.x.tolist():
        assert z in valid_atomic_numbers, (
            f"Got atomic number {z} not in type_map {type_map}"
        )


def test_types_noiser_type_map_denoise_non_last_stays_compact():
    """Non-final _denoise steps should keep compact indices (no remapping)."""
    type_map = [0, 1, 6, 8]  # absorbing, H, C, O
    noiser = Types(type_map=type_map)
    n_classes = len(type_map)

    from agedi.data import AtomsGraph
    from ase.build import molecule
    atoms = molecule("H2O")
    atoms.set_cell([10, 10, 10])
    atoms.set_pbc(True)
    atoms.center()
    graph = AtomsGraph.from_atoms(atoms)

    graph.x = torch.tensor([1, 3, 3], dtype=torch.long)
    graph.time = torch.full((graph.num_nodes, 1), 0.5)
    graph.x_score = torch.log_softmax(torch.randn(graph.num_nodes, n_classes), dim=-1)

    noiser.distribution.get_callable = lambda _: (lambda probs: probs.argmax(dim=-1))
    denoised = noiser._denoise(graph, delta_t=0.01, last=False)

    # Non-last step: values should be compact indices [0, n_classes)
    assert denoised.x.max().item() < n_classes, (
        f"Non-last denoise returned index >= n_classes: {denoised.x}"
    )


def test_types_noiser_type_map_get_hparams():
    """get_hparams() should include type_map when set."""
    type_map = [0, 1, 6, 8]
    noiser = Types(type_map=type_map)
    hparams = noiser.get_hparams()
    assert "type_map" in hparams
    assert hparams["type_map"] == type_map
    assert hparams["n_classes"] == len(type_map)


def test_types_noiser_no_type_map_get_hparams():
    """get_hparams() should NOT include type_map when not set."""
    noiser = Types()
    hparams = noiser.get_hparams()
    assert "type_map" not in hparams
    assert hparams["n_classes"] == 100

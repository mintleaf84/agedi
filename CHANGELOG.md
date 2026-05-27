# Changelog

All notable changes to AGeDi will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-05-22

### Added
- Predictor-corrector (Langevin corrector) sampling via `corrector_steps` /
  `corrector_step_size` parameters.
- Force-field guided sampling (`ForcefieldGuidanceConfig`, `--ff_guidance` /
  `--ff_zeta` CLI flags, `force_field=True` training option).
- Post-diffusion relaxation loop (auto-triggered when `ff_guidance` is enabled
  and forces exceed `force_threshold`).
- `torch.compile` support for the reverse diffusion step (`compile=True` /
  `--compile`), compiled per-instance to avoid cross-model interference.
- `agedi predict` CLI command and `functional.predict()` API for energy/force
  inference with trained regressor heads.
- `agedi inspect` CLI command for inspecting saved model checkpoints.
- `agedi train-hydra` CLI command and `train_from_config()` API supporting a
  full YAML config file (`conf/train.yaml`).
- Resume training from checkpoint (`--checkpoint` / `checkpoint:` config key).
- Separate `regressor_data_path` support in config-file training for
  non-equilibrium regressor data.
- `Noiser.register()` class method for registering custom noisers.
- `register_model()` function for registering custom GNN backbone factories.
- `save_trajectory` parameter on `sample()` / `functional.sample()` (replaces
  the old `save_path` parameter which has been removed).
- `print_timings` parameter for per-stage sampling timing breakdown.
- Skin-based neighbor-list caching in `update_graph()` to skip full rebuilds
  when atoms have moved less than the skin distance.
- `canonical_cell` option in `from_atoms()` / CLI / config to store cells in
  canonical lower-triangular form.
- `n_classes` option to explicitly set the number of atom-type classes for the
  `Types` noiser.
- `repeat` / `repeat_epoch` cell-repeat data augmentation.
- `batch_naive_neighbor_list` pure-PyTorch fallback neighbor-list (used in the
  compiled path when `nvalchemiops` is not available).

### Changed
- **Breaking**: `save_path` parameter removed from `Diffusion.sample()`,
  `Diffusion._sample()`, and `functional.sample()`.  Use `save_trajectory`
  instead.
- `nvalchemiops` is now an **optional** dependency (`pip install agedi[cuda]`).
  The non-compiled sampling path works without it.
- `Agedi.__init__` `optim_config` and `scheduler_config` default to `None`
  (computed internally) rather than shared mutable dicts.
- `AtomsGraph.from_atoms` cell-canonicalization now emits `warnings.warn`
  instead of `print`.
- Compiled reverse step is now a lazy per-instance property instead of a
  class-level `@torch.compile` decorator.
- CI matrix extended to macOS runners; `.[test,full]` installed in one step.

### Fixed
- Python version badge corrected to `3.12+`.
- `AtomsGraph` class docstring typo corrected.
- `from_atoms` `canonical_cell` parameter docstring corrected (default is
  `False`, not `True`).

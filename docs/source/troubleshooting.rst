Pitfalls and troubleshooting
============================

Most common failure modes are due to missing required sampling inputs or
mismatched configuration between training and sampling.

Sampling input pitfalls
-----------------------

- **Missing ``cell`` without template**

  If you sample without ``template``, provide cell information directly or use
  a run whose ``hparams.yaml`` includes ``cell`` metadata.

- **Missing atom specification**

  You must provide enough information to infer generated atoms:
  ``formula``, or ``n_atoms`` + ``atomic_numbers`` as required by your noisers.

- **Type-only / position-only configurations**

  If no position noiser is active, fixed ``positions`` are required at sampling.
  If no type noiser is active, types must come from ``formula`` or ``atomic_numbers``.

Training pitfalls
-----------------

- **No SchNetPack installed for PaiNN workflows**

  Install with ``agedi[full]``.

- **Mask behavior misunderstood**

  ``MaskFixed`` only freezes atoms marked by ASE ``FixAtoms`` constraints.

- **Confinement mismatch**

  Keep training and sampling confinement bounds consistent for slab/surface tasks.

- **Repeat settings**

  If ``repeat`` is set, ``repeat_epoch`` must also be set.

Data pitfalls
-------------

- Ensure periodic cells/pbc are set consistently in ASE data.
- Extremely small datasets can yield unstable train/val splits and noisy metrics.
- Large cutoffs and batch sizes increase memory usage substantially.

Operational checks
------------------

- Inspect run config with ``agedi inspect <log_dir>``.
- Confirm ``hparams.yaml`` and checkpoints exist before loading.
- Start with smaller ``steps`` / ``batch_size`` to diagnose OOM issues.

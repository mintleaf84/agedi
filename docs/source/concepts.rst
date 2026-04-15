Concepts and model behavior
===========================

Graph representation
--------------------

AGeDi uses ``AtomsGraph`` as the main data object:

- Nodes: atomic numbers (``x``) and positions (``pos``)
- Edges: neighbor graph from periodic cutoff
- Graph-level data: cell, pbc, optional confinement
- Optional mask marks fixed atoms during diffusion updates

Diffusion components
--------------------

``Diffusion`` combines:

- A score model (predicts scores for configured targets)
- One or more noisers (e.g., positions, types)
- Optimizer/scheduler configuration for Lightning training

Supported score/noiser pairing is enforced by key matching.

Sampling semantics
------------------

During sampling, required defaults depend on enabled noisers:

- ``n_atoms`` can come from explicit input, ``atomic_numbers``, or ``formula``
- ``atomic_numbers`` are needed if type noising is not enabled and formula is not provided
- ``positions`` are needed if position noising is not enabled
- ``cell`` is needed unless a template provides it

If a template is provided, generated atoms are appended to template atoms and
template atoms are masked as fixed.

Training outputs
----------------

By default, training writes to ``logs/version_x``:

- ``hparams.yaml``: run hyperparameters and data metadata
- ``checkpoints/``: model checkpoints

``load_diffusion`` reconstructs the model from these artifacts.

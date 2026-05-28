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

``Agedi`` combines:

- A score model (predicts scores for configured targets)
- One or more noisers (e.g., positions, types)
- Optimizer/scheduler configuration for Lightning training

Supported score/noiser pairing is enforced by key matching.

Position noisers
----------------

Three position noisers are available, each with a fixed prior and noise
distribution baked in.  Choose based on the physics of your system:

.. list-table::
   :header-rows: 1
   :widths: 35 25 25 25

   * - Class / identifier
     - Prior
     - Distribution
     - Use case
   * - :class:`~agedi.diffusion.noisers.Positions` / ``"Positions"``
     - :class:`~agedi.diffusion.distributions.StandardNormal`
     - :class:`~agedi.diffusion.distributions.Normal`
     - Gas-phase (molecules, clusters)
   * - :class:`~agedi.diffusion.noisers.CellPositions` / ``"CellPositions"``
     - :class:`~agedi.diffusion.distributions.UniformCell`
     - :class:`~agedi.diffusion.distributions.Normal`
     - Periodic bulk / surface (default)
   * - :class:`~agedi.diffusion.noisers.ConfinedCellPositions` / ``"ConfinedCellPositions"``
     - :class:`~agedi.diffusion.distributions.UniformCellConfined`
     - :class:`~agedi.diffusion.distributions.TruncatedNormal`
     - Surface overlayer/adsorbate

The **prior** is the distribution used to initialise atomic positions at the
start of the reverse (sampling) process.  The **distribution** is the noise
kernel applied during the forward (training) process.  The SDE can still be
chosen freely on all three classes (default: Variance-Exploding, ``"ve"``).

Discrete atom types can be diffused by adding a
:class:`~agedi.diffusion.noisers.Types` to the noiser list.

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

Property conditioning
---------------------

The score model can be conditioned on a per-structure scalar or integer
property so that sampling can be steered towards a target value (e.g.
formation energy or band gap).  Use the ``conditioning`` parameter (CLI:
``--conditioning``) to specify the property name and
``conditioning_type`` (CLI: ``--conditioning_type``) to choose between
``"scalar"`` (continuous, default) and ``"integer"`` (discrete) encoding.

The property value is looked up from ``atoms.info[conditioning]`` or
``atoms.get_<conditioning>()`` for each training structure.  At sampling
time pass the target value in the ``property`` dict:

.. code-block:: python

   structures = sample(diffusion, n_samples=10, formula="Pd4O4",
                       property={"energy": -3.5})

Data augmentation (cell repeat)
---------------------------------

For periodic systems it can be beneficial to augment the training data by
tiling each structure along the first two cell vectors.  Enable this with
``repeat`` (CLI: ``--repeat``) and set the epoch interval at which the
repetition level increases with ``repeat_epoch`` (CLI: ``--repeat_epoch``).

For example, ``repeat=3, repeat_epoch=50`` starts training on the original
cells, increases to 2×2×1 at epoch 50, then to 3×3×1 at epoch 100.


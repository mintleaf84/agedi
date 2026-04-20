Command-line interface
======================

AGeDi installs a CLI entrypoint named ``agedi``.

Discover commands
-----------------

.. code-block:: console

   agedi --help

Main commands
-------------

- ``agedi train``: train a diffusion model from ASE trajectory data
- ``agedi train-hydra``: train from a YAML configuration file
- ``agedi sample``: sample structures from a saved training run
- ``agedi inspect``: print ``hparams.yaml`` from a run directory

To get information about options for each use

.. code-block:: console

   agedi train --help

for ``train`` and likewise for ``sample``, ``train-hydra``, and ``inspect``.

Training
--------

Choose one of the three position noisers to match your system type:

.. list-table:: Position noisers
   :header-rows: 1
   :widths: 30 25 25 20

   * - Noiser
     - Prior
     - Distribution
     - Use case
   * - ``positions``
     - StandardNormal
     - Normal
     - Gas-phase clusters
   * - ``cell_positions``
     - UniformCell
     - Normal
     - Periodic bulk / surface (default)
   * - ``confined_cell_positions``
     - UniformCellConfined
     - TruncatedNormal
     - Z-confined surface / slab

Minimal training example for a **surface system with Z-confinement**:

.. code-block:: console

   agedi train --noisers confined_cell_positions --mask MaskFixed --confinement 2 10 training_data.traj

Minimal training example for a **periodic bulk or surface** system:

.. code-block:: console

   agedi train --noisers cell_positions training_data.traj

Minimal training example for a **gas-phase cluster**:

.. code-block:: console

   agedi train --noisers positions training_data.traj

Important options:

- ``--max_time/-t`` or ``--epochs/-e``: stopping criteria
- ``--noisers``: ``cell_positions`` (default), ``confined_cell_positions``, ``positions``, ``types``
- ``--sde``: ``ve`` (default), ``vp``
- ``--mask MaskFixed``: freezes atoms tagged with ASE ``FixAtoms``
- ``--confinement zmin zmax``: z-direction confinement bounds (required for ``confined_cell_positions``)


Sampling
--------

.. code-block:: console

   agedi sample logs/version_0 -f Pd2O2 --template_path template.traj --steps 500 --confinement 2 10

This samples using the ``last_model.ckpt`` checkpoint found in
``logs/version_0``. If you want to use a different checkpoint, you can
specify the exact path to it.

Important options:

- ``-f/--formula`` or ``-a/--n_atoms``
- ``--template_path`` for template-guided generation
- ``--steps``, ``--eps`` for reverse diffusion resolution

Inspect run metadata
--------------------

.. code-block:: console

   agedi inspect logs/version_0

This prints the saved hyperparameters from the run directory (for example, the parsed contents of ``hparams.yaml``).

Logging options
--------------------

AGeDi saves TensorBoard logs by default. WandB can be saved instead
using the ``--logger wandb`` option when training.

To follow training use

.. code-block:: console

   tensorboard --logdir .

This hosts TensorBoard at localhost. Remember to forward a specific
port to your local machine if using HPC. You can use the ``--port
xxxx`` option for TensorBoard to host at this specific port.


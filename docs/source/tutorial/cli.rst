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
- ``agedi sample``: sample structures from a saved training run
- ``agedi inspect``: print ``hparams.yaml`` from a run directory

Training
--------

Minimal training example:

.. code-block:: console

   agedi train -t 3 --style surface --mask MaskFixed --confinement 2 10 PdO_training_data.traj

Important options:

- ``--style``: ``Default``, ``surface``, ``cluster``
- ``--noisers``: choose diffusion targets (typically ``positions`` and/or ``types``)
- ``--mask MaskFixed``: freeze atoms tagged with ASE ``FixAtoms``
- ``--confinement zmin zmax``: z-direction confinement bounds
- ``--max_time/-t`` and ``--epochs/-e``: stopping criteria

Sampling
--------

.. code-block:: console

   agedi sample logs/version_0 -f Pd2O2 --template_path template.traj --confinement 2 10

Key options:

- ``-f/--formula`` or ``-a/--n_atoms``
- ``--template_path`` for template-guided generation
- ``--steps``, ``--eps`` for reverse diffusion resolution
- ``--save_trajectory`` to save full denoising trajectories

Inspect run metadata
--------------------

.. code-block:: console

   agedi inspect logs/version_0

This prints the saved hyperparameters from the run directory (for example, the parsed contents of ``hparams.yaml``).

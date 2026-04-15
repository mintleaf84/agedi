PdO on Pd — Python API
======================

This page repeats the :doc:`PdO tutorial <pdo>` using the AGeDi
**functional Python API** instead of the command-line interface.
The API is useful when you want to integrate AGeDi into a larger
script, change hyper-parameters programmatically, or chain training
and sampling in one file.

Data
----

Download the same training data and surface template used in the CLI
tutorial:

.. code-block:: console

   wget https://github.com/nronne/agedi/raw/refs/heads/main/docs/tutorial_data/PdO_training_data.traj
   wget https://github.com/nronne/agedi/raw/refs/heads/main/docs/tutorial_data/template.traj

Training
--------

The script below reproduces the CLI command

.. code-block:: console

   agedi train -t 240 --mask MaskFixed --confinement 2 10 PdO_training_data.traj

using the :func:`agedi.train_from_atoms` convenience function:

.. code-block:: python

   from ase.io import read
   from agedi import train_from_atoms

   # Load the training structures
   data = read("PdO_training_data.traj", ":")

   # Build, configure, and train the model in one call.
   # MaskFixed respects ASE FixAtoms constraints so the substrate
   # atoms are kept frozen during diffusion.
   diffusion, dataset, trainer = train_from_atoms(
       data,
       noisers=("positions",),
       style="surface",
       mask="MaskFixed",
       confinement=(2.0, 10.0),   # z-confinement in Å
       time_hours=4,              # train for 4 hours (240 minutes)
       log_dir="logs",
   )

After training, AGeDi saves checkpoints under ``logs/version_0/checkpoints/``
and hyper-parameters to ``logs/version_0/hparams.yaml``.  Training progress
can be monitored with ``tensorboard --logdir logs``.

If you prefer to control each step individually, the same result can be
achieved by composing the lower-level helpers:

.. code-block:: python

   from ase.io import read
   from agedi import create_diffusion, create_dataset, create_trainer, train
   from agedi.diffusion.noisers import PositionsNoiser
   from agedi.diffusion.distributions import TruncatedNormal, UniformCellConfined

   data = read("PdO_training_data.traj", ":")

   # Build the noiser explicitly with the surface-confinement distributions
   noiser = PositionsNoiser(
       distribution=TruncatedNormal(),
       prior=UniformCellConfined(),
   )

   diffusion = create_diffusion(
       noisers=[noiser],
       confinement=(2.0, 10.0),
   )

   dataset = create_dataset(
       data,
       mask="MaskFixed",
       confinement=(2.0, 10.0),
   )

   trainer = create_trainer(time_hours=4, log_dir="logs")
   train(diffusion, dataset, trainer=trainer)

Sampling
--------

After training, load the checkpoint and sample new PdO/Pd structures.
The template file provides the fixed Pd substrate on top of which new
atoms will be placed:

.. code-block:: python

   import numpy as np
   from ase import Atoms
   from ase.io import read, write

   from agedi import load_diffusion, sample
   from agedi.data import AtomsGraph

   # Re-build the model from the saved log directory
   diffusion = load_diffusion("logs/version_0")

   # Build the template graph from the Pd substrate file
   template_atoms = read("template.traj")
   template = AtomsGraph.from_atoms(
       template_atoms, initialize_mask=False, confinement=(2.0, 10.0)
   )

   # Sample 12 structures with the Pd2O2 stoichiometry
   formula = Atoms("Pd2O2")
   structures = sample(
       diffusion,
       n_samples=12,
       atomic_numbers=formula.get_atomic_numbers(),
       cell=np.array(template_atoms.cell),
       template=template,
       confinement=(2.0, 10.0),
       steps=500,
   )

   write("sampled.traj", structures)

The output file ``sampled.traj`` contains the sampled ASE
:class:`~ase.Atoms` objects and can be inspected with
``ase gui sampled.traj``.

References
----------
[1] N. Rønne, A. Aspuru-Guzik and B. Hammer. *Phys. Rev. B* **110**, 235427 (2024):
https://doi.org/10.1103/PhysRevB.110.235427

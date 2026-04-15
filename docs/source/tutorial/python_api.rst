Python API workflow
===================

This page shows the script-based workflow using :mod:`agedi.functional`.

One-call training
-----------------

.. code-block:: python

   from ase.io import read
   from agedi import train_from_atoms

   data = read("PdO_training_data.traj", ":")

   diffusion, dataset, trainer = train_from_atoms(
       data,
       noisers=("positions",),
       style="surface",
       mask="MaskFixed",
       confinement=(2.0, 10.0),
       max_time=3,  # hours
       log_dir="logs",
   )

Composed workflow
-----------------

.. code-block:: python

   from ase.io import read
   from agedi import create_diffusion, create_dataset, create_trainer, train

   data = read("PdO_training_data.traj", ":")
   diffusion = create_diffusion(noisers=("positions",), style="surface", confinement=(2.0, 10.0))
   dataset = create_dataset(data, mask="MaskFixed", confinement=(2.0, 10.0))
   trainer = create_trainer(max_time=3, log_dir="logs")  # 3 hours
   train(diffusion, dataset, trainer=trainer)

Sampling with template
----------------------

.. code-block:: python

   from ase.io import read, write
   from agedi import load_diffusion, sample, AtomsGraph

   diffusion = load_diffusion("logs/version_0")
   template = AtomsGraph.from_atoms(read("template.traj"), confinement=(2.0, 10.0))

   structures = sample(
       diffusion,
       n_samples=12,
       formula="Pd2O2",
       template=template,
       confinement=(2.0, 10.0),
       steps=500,
   )

   write("sampled.traj", structures)

Core public functions
---------------------

- ``create_diffusion``
- ``create_dataset``
- ``create_trainer``
- ``train``
- ``train_from_atoms``
- ``load_diffusion``
- ``sample``

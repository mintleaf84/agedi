Python API workflow
===================

This page shows the script-based workflow using functions from
:mod:`agedi.functional`, re-exported at the top-level :mod:`agedi` package.
Using the functional API allows for more customization than relying on
the CLI. 

     
Training
-----------------

Here, we show the same example as with the CLI, but now using the
Python API and the ``train_from_atoms`` functionality

.. code-block:: python

   from ase.io import read
   from agedi import train_from_atoms

   data = read("training_data.traj", ":")

   diffusion, dataset, trainer = train_from_atoms(
       data,
       noisers=("positions",),
       style="surface",
       mask="MaskFixed",
       confinement=(2.0, 10.0),
       max_time=2,  # hours
       log_dir="logs",
   )

More detailed workflow
-----------------

Here, we show a more detailed example setting up the diffusion model,
the dataset and the trainer individually and using the ``train``
functionality of AGeDi.

.. code-block:: python

   from ase.io import read
   from agedi import create_diffusion, create_dataset, create_trainer, train

   data = read("training_data.traj", ":")
   
   diffusion = create_diffusion(
       noisers=("positions",),
       style="surface",
       confinement=(2.0, 10.0)
   )
   
   dataset = create_dataset(
       data,
       mask="MaskFixed",
       confinement=(2.0, 10.0)
   )
   
   trainer = create_trainer(
       max_time=2, # hours
       log_dir="logs"
   )
   
   train(diffusion, dataset, trainer=trainer)

Sampling with template
----------------------

To sample, whether model is trained using CLI og Python API, we use
the ``sample`` functionality

.. code-block:: python

   from ase.io import read, write
   from agedi import load_diffusion, sample, AtomsGraph

   diffusion = load_diffusion("logs/version_0")
   
   template = AtomsGraph.from_atoms(read("template.traj"), confinement=(2.0, 10.0))

   structures = sample(
       diffusion,
       n_samples=12,
       formula="X2Y3",
       template=template,
       confinement=(2.0, 10.0),
       steps=500,
   )

   write("sampled.traj", structures)

Similar to the CLI, this samples using the ``last_model.ckpt`` checkpoint found in
``logs/version_0``. If you want to use a different checkpoint, you can
specify the exact path to it, when calling ``load_diffusion``.
   
   
Core public functions
---------------------

- ``create_diffusion``
- ``create_dataset``
- ``create_trainer``
- ``train``
- ``train_from_atoms``
- ``load_diffusion``
- ``sample``

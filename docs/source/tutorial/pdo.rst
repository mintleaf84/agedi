PdO on Pd
=========
Following [1] we want to train a diffusion model for
surface structure generation of a PdO thin-film. Data for this can be
downloaded by
.. code-block:: console
   wget https://github.com/nronne/agedi/tree/main/docs/tutorial_data/...

This will download a ASE Traj file that contains small PdO-Pd
structures and will be our training data. Notice how some of the
atoms in the training data is constrained using ASE ``FixAtoms``,
which will be translated into a masking in the diffusion model.


To train the diffusion model simply call:
.. code-block:: console
   agedi train -t 240 --mask MaskFixed --confinement 2 10 --noiser_distributions TruncatedNormal --prior_distributions UniformCellConfined PdO_training_data.traj

Following [1] we use a z-directional confinement using a truncated
normal distribution and choose to train the model for 240 minutes. Training is
most efficiently done using GPU. In the ``logs`` directory 
created the settings are written to the ``hparams.yaml`` file and
training checkpoints are stored in the ``checkpoints`` directory.

The training can easily be monitored using ``Tensorboard``.

After training sampling requires you to setup a surface template. For
now we will get it through
.. code-block:: console
   wget https://github.com/nronne/agedi/tree/main/docs/tutorial_data/...

Now sampling simply becomes   
.. code-block:: console
   agedi sample logs/version_0 -f Pd2O2 --template_path template.traj --confinement 2 10

that will write the ``sampled.traj`` ASE Traj file with the sampled
structures from the diffusion model.







References
----------
[1] N. Rønne, A. Aspuru-Guzik and B. Hammer. Phys. Rev. B 110, 235427 (2024): https://doi.org/10.1103/PhysRevB.110.235427


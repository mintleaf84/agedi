PdO-on-Pd end-to-end example
============================

This reproduces the surface-generation workflow used in Ref. [1].

Download tutorial data
----------------------

.. code-block:: console

   wget https://github.com/nronne/agedi/raw/refs/heads/main/docs/tutorial_data/PdO_training_data.traj
   wget https://github.com/nronne/agedi/raw/refs/heads/main/docs/tutorial_data/template.traj

Train
-----

.. code-block:: console

   agedi train -t 3 --style surface --mask MaskFixed --confinement 2 10 PdO_training_data.traj

Notes:

- ``MaskFixed`` maps ASE ``FixAtoms`` constraints to the graph mask.
- Confinement applies to z-coordinates and is useful for slab/surface tasks.
- Outputs are written in ``logs/version_0`` (or next available version).

Sample
------

.. code-block:: console

   agedi sample logs/version_0 -f Pd2O2 --template_path template.traj --confinement 2 10

This writes sampled structures to ``sampled.traj``.

References
----------

[1] N. Rønne, A. Aspuru-Guzik and B. Hammer. *Phys. Rev. B* **110**, 235427 (2024):
   https://doi.org/10.1103/PhysRevB.110.235427

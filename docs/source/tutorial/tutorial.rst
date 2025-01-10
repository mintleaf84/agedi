Tutorial
=======

This section gives a simple introduction to the AGeDi package by
training and sampling with a diffusion model using the AGeDi
CLI.

.. toctree::
   :maxdepth: 1

   pdo



Command Line Interface
----------------------
After installing AGeDi try running

.. code-block:: console
   agedi --help

The will look like
.. click:: agedi:agedi
   :prog: agedi
   :nested: short

The CLI exposes the two main functionalities namely training and
sampling the model. To inspect the possibilities within try

.. code-block:: console
   agedi train --help

and

.. code-block:: console
   agedi sample --help





Installation
============

Requirements
------------

- Python >= 3.10

Install from GitHub
-------------------

Minimal install directly from Github:

.. code-block:: console

   pip install "agedi @ git+https://github.com/nronne/agedi.git"

This installs the core package only. For the current release, training and
sampling require the PaiNN backend, which depends on SchNetPack.

Install with full model dependencies (required for training/sampling)
directly from Github:

.. code-block:: console

   pip install "agedi[full] @ git+https://github.com/nronne/agedi.git"

Developer install
----------------------

Clone from Github

.. code-block:: console

   git clone https://github.com/nronne/agedi.git

.. code-block:: console

   cd agedi
   
.. code-block:: console

   pip install -e ".[full]"

  
Verify installation
-------------------

.. code-block:: console

   agedi --help

and

.. code-block:: console

   python -c "import agedi; print(agedi.__all__)"

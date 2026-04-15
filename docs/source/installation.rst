Installation
============

Requirements
------------

- Python >= 3.7
- PyTorch-compatible environment (CPU or GPU)

Install from GitHub
-------------------

Minimal install:

.. code-block:: console

   pip install "agedi @ git+https://github.com/nronne/agedi.git"

Install with full model dependencies (recommended for PaiNN workflows):

.. code-block:: console

   pip install "agedi[full] @ git+https://github.com/nronne/agedi.git"

Developer/test install
----------------------

.. code-block:: console

   pip install -e .[test] -e .[full]

Verify installation
-------------------

.. code-block:: console

   agedi --help

and

.. code-block:: console

   python -c "import agedi; print(agedi.__all__)"

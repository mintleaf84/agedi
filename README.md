<p align="center">
  <img height="250" src="https://raw.githubusercontent.com/nronne/agedi/refs/heads/main/docs/agedi.svg?sanitize=true" />
</p>

______________________________________________________________________


[![Build Status](https://github.com/nronne/agedi/actions/workflows/python-package.yml/badge.svg)](https://github.com/nronne/agedi/actions/workflows/python-package.yml)
[![Documentation Status](https://readthedocs.org/projects/agedi/badge/?version=latest)](https://agedi.readthedocs.io/en/latest/?badge=latest)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)


**[Documentation](https://agedi.readthedocs.io)**


**AGeDI** pronounced "A Jedi" is a library for **A**tomistic **Ge**nerative
**Di**ffusion build on PyG, Lightning and ASE and offers customizable
diffusion models for periodic atomistic material generation.

> [!CAUTION]
> This project is under active development.

## Interfaced Equivariant Models
At the moment only PaiNN is possible to use as a score model
architecture.

We expect to implement interfaces to GemNet-dQ, NequIP and possibly Mace. 

## Implemented Noisers and Scorers
Below is an overview of the different available noisers and for which
models there is an score-model implementation.

|                                       | Cartesian Coordinates | Fractional Coordinates | Atomic Types         | Cell                 |
| ------------------------------------- | --------------------- | ---------------------- | -------------------- | -------------------- |
| PaiNN                                 | :white_check_mark:    | :white_large_square:   | :white_check_mark:   | :white_large_square: |
| GemNet-dQ                             | :white_large_square:  | :white_large_square:   | :white_large_square: | :white_large_square: |
| NequIP                                | :white_large_square:  | :white_large_square:   | :white_large_square: | :white_large_square: |

The diffusion model is based on continuous-time diffusion for all
implementation. Specifically for atomic coordinates, we use the SDE
diffusion formulation trained with score-matching. For the atomic types, we
use the discrete score-entropy diffusion formulation with the concrete
scores trained using the score entropy loss. 

<p align="center">
  <img height="250" src="https://raw.githubusercontent.com/nronne/agedi/refs/heads/main/docs/diff.gif?sanitize=true" />
</p>

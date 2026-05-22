# Contributing to AGeDi

Thank you for your interest in contributing to AGeDi!  This document describes
how to set up a development environment and the conventions we follow.

## Setting up the development environment

```bash
git clone https://github.com/nronne/agedi.git
cd agedi
pip install -e ".[test,full]"
```

The `full` extra installs the SchNetPack backend (requires Git).  The `cuda`
extra (`pip install -e ".[cuda]"`) installs the optional NVIDIA nvalchemiops
package needed for `torch.compile`-accelerated sampling.

## Running the tests

```bash
pytest
```

## Code style

We use [Ruff](https://docs.astral.sh/ruff/) for linting and formatting (line
length 88).  Before opening a pull request please run:

```bash
ruff check src/
ruff format src/
```

## Docstrings

Use [NumPy-style docstrings](https://numpydoc.readthedocs.io/en/latest/format.html)
for all public classes and functions.

## Opening a pull request

1. Fork the repository and create a feature branch from `main`.
2. Add tests for any new functionality.
3. Run the full test suite (`pytest`) and ensure it passes.
4. Update `CHANGELOG.md` under the `[Unreleased]` section.
5. Open a pull request against `main` with a clear description of the change.

## Reporting bugs

Please open an issue using the **Bug report** template and include:

- AGeDi version (`python -c "import agedi; print(agedi.__version__)"`)
- Python version and operating system
- Minimal reproducible example
- Full traceback

## Feature requests

Open an issue using the **Feature request** template.  Describe the use-case
and, if possible, a sketch of the proposed API.

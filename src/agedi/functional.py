"""Backward-compatibility shim.

All public symbols are now implemented in :mod:`agedi.api`.
This module re-exports them so that existing code using
``from agedi.functional import X`` continues to work unchanged.
"""

from agedi.api import (  # noqa: F401
    create_dataset,
    create_diffusion,
    create_trainer,
    load_diffusion,
    predict,
    register_model,
    sample,
    train,
    train_from_atoms,
    train_from_config,
)

__all__ = [
    "create_dataset",
    "create_diffusion",
    "create_trainer",
    "load_diffusion",
    "predict",
    "register_model",
    "sample",
    "train",
    "train_from_atoms",
    "train_from_config",
]



try:
    import torch
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "xlnstorch requires PyTorch but it is not installed.\n"
        "See https://pytorch.org/get-started/locally for instructions."
    ) from e

from .tensor import LNSTensor, lnstensor, implements, set_default_implementation, override_impl, apply_lns_op
from . import operators

__all__ = [
    "LNSTensor",
    "lnstensor"
    "implements",
    "set_default_implementation",
    "override_impl",
    "apply_lns_op"
]
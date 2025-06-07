try:
    import torch
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "xlnstorch requires PyTorch but it is not installed.\n"
        "See https://pytorch.org/get-started/locally for instructions."
    ) from e

LNS_ZERO = torch.tensor(-2**53, dtype=torch.float64)

from .dispatch_table import (
    implements,
    get_implementation,
    set_default_implementation,
    get_default_implementation_key,
    override_implementation,
    apply_lns_op
)
from .tensor import (
    LNSTensor,
    lnstensor,
)
from .base import (
    align_lnstensor_bases,
    format_lnstensor_operands,
)
from . import operators

__all__ = [
    "LNS_ZERO",
    "LNSTensor",
    "lnstensor",
    "implements",
    "get_implementation",
    "set_default_implementation",
    "get_default_implementation_key",
    "override_implementation",
    "apply_lns_op",
    "align_lnstensor_bases",
    "format_lnstensor_operands",
]
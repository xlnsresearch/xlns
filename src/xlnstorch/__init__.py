try:
    import torch
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "xlnstorch requires PyTorch but it is not installed.\n"
        "See https://pytorch.org/get-started/locally for instructions."
    ) from e

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
from .lns_ops import (
    lns_add,
    lns_sub,
    lns_mul,
    lns_div,
    lns_neg,
    lns_abs,
    lns_sqrt,
    lns_square,
    lns_pow,
    lns_reciprocal,
    lns_sign,
    lns_positive,
)
from . import operators

__all__ = [
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
    # LNS operation functions
    "lns_add",
    "lns_sub",
    "lns_mul",
    "lns_div",
    "lns_neg",
    "lns_abs",
    "lns_sqrt",
    "lns_square",
    "lns_pow",
    "lns_reciprocal",
    "lns_sign",
    "lns_positive",
]
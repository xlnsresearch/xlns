import logging
try:
    import torch
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "xlnstorch requires PyTorch but it is not installed.\n"
        "See https://pytorch.org/get-started/locally for instructions."
    ) from e

# These constants are independent of base so we can precompute
# their internal representations.
LNS_ZERO = torch.tensor(-2**53 | 1, dtype=torch.float64)
LNS_ONE = torch.tensor(0, dtype=torch.float64)
LNS_NEG_ONE = torch.tensor(1, dtype=torch.float64)

try:
    from . import _C
    _C_AVAILABLE = True
except ImportError as e:
    logging.info("xlnstorch c++ extension not found. Reverting to pure Python implementation.")
    _C_AVAILABLE = False

from . import autograd

from .dispatch_table import (
    implements,
    get_implementation,
    set_default_implementation,
    get_default_implementation_key,
    override_implementation,
    apply_lns_op
)
from .autograd import LNSFunction
from .tensor_utils import (
    align_lnstensor_bases,
    format_lnstensor_operands,
)
from .tensor import (
    LNSTensor,
    lnstensor,
    zeros,
    zeros_like,
    ones,
    ones_like,
    full,
    full_like,
    rand,
    rand_like,
    randn,
    randn_like,
)
from . import operators
from . import nn
from . import optim
from . import viz
from . import benchmark

__all__ = [
    "LNS_ZERO",
    "LNS_ONE",
    "LNS_NEG_ONE",
    "_C_AVAILABLE",

    "LNSTensor",
    "lnstensor",
    "zeros",
    "zeros_like",
    "ones",
    "ones_like",
    "full",
    "full_like",
    "rand",
    "rand_like",
    "randn",
    "randn_like",

    "implements",
    "get_implementation",
    "set_default_implementation",
    "get_default_implementation_key",
    "override_implementation",
    "apply_lns_op",

    "LNSFunction",
    "align_lnstensor_bases",
    "format_lnstensor_operands",
]
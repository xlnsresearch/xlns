import xlnstorch
from xlnstorch import set_default_implementation
import torch

from . import addition_ops
from . import arithmetic_ops
from . import unary_ops
from . import comparison_ops
from . import loss_ops
from . import activation_ops
from . import layer_ops
from . import misc_ops

from . import _C
from . import implementations

def toggle_cpp_implementations(use_cpp: bool) -> None:
    """
    Toggle the use of C++ implementations for operators that have them.

    Parameters
    ----------
    use_cpp : bool
        If ``True``, use C++ implementations where available.
        If ``False``, use pure Python implementations.

    Raises
    ------
    RuntimeError
        If C++ extensions are not available.
    """
    if use_cpp and not xlnstorch.CSRC_AVAILABLE:
        raise RuntimeError("C++ extensions are not available. Cannot enable C++ implementations.")

    for torch_op, (py_key, cpp_key) in _C.CPP_IMPLEMENTED_OPERATORS.items():
        impl_key = cpp_key if use_cpp else py_key
        set_default_implementation(torch_op, impl_key)

    xlnstorch.tensor_utils.toggle_cpp_tensor_utils(use_cpp)

__all__ = [
    "toggle_cpp_implementations",
    "lns_sum_to_size",
]
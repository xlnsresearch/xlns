from typing import Callable, Any

import torch
from . import LNSTensor, apply_lns_op

def _create_lns_op_func(
    op_name: str,
    torch_op: Callable,
    docstring: str | None = None,
) -> Callable:
    """
    Factory function to create LNS operation functions. These functions are
    the operations that are performed on the internal representations of
    LNSTensors.

    Parameters
    ----------
    op_name : str
        Name of the operation (e.g., 'add', 'mul')
    torch_op : Callable
        The corresponding PyTorch operation function
    docstring : str, optional
        Docstring for the function

    Returns
    -------
    Callable
        The created LNS operation function
    """
    def func(*args: Any, **kwargs: Any) -> LNSTensor:
        return apply_lns_op(torch_op, *args, **kwargs)

    func.__name__ = f"lns_{op_name}"
    func.__doc__ = f"See docs for `{torch_op.__module__}.{torch_op.__name__}` for more details."
    if docstring:
        func.__doc__ = f"{docstring}\n\n{func.__doc__}"

    return func

lns_add = _create_lns_op_func('add', torch.add)
lns_sub = _create_lns_op_func('sub', torch.sub)
lns_mul = _create_lns_op_func('mul', torch.mul)
lns_div = _create_lns_op_func('div', torch.div)
lns_neg = _create_lns_op_func('neg', torch.neg)
lns_abs = _create_lns_op_func('abs', torch.abs)
lns_sqrt = _create_lns_op_func('sqrt', torch.sqrt)
lns_square = _create_lns_op_func('square', torch.square)
lns_pow = _create_lns_op_func('pow', torch.pow)
lns_reciprocal = _create_lns_op_func('reciprocal', torch.reciprocal)
lns_sign = _create_lns_op_func('sign', torch.sign)
lns_positive = _create_lns_op_func('positive', torch.positive)
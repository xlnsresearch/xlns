from typing import Callable, Any

import torch
from .. import LNSTensor, apply_lns_op

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
lns_sum = _create_lns_op_func('sum', torch.sum)
lns_matmul = _create_lns_op_func('matmul', torch.matmul)

lns_equal = _create_lns_op_func('equal', torch.equal)
lns_eq = _create_lns_op_func('eq', torch.eq)
lns_ne = _create_lns_op_func('ne', torch.ne)
lns_ge = _create_lns_op_func('ge', torch.ge)
lns_gt = _create_lns_op_func('gt', torch.gt)
lns_le = _create_lns_op_func('le', torch.le)
lns_lt = _create_lns_op_func('lt', torch.lt)
lns_isclose = _create_lns_op_func('isclose', torch.isclose)
lns_allclose = _create_lns_op_func('allclose', torch.allclose)
lns_any = _create_lns_op_func('any', torch.any)
lns_all = _create_lns_op_func('all', torch.all)
lns_isin = _create_lns_op_func('isin', torch.isin)
lns_sort = _create_lns_op_func('sort', torch.sort)
lns_argsort = _create_lns_op_func('argsort', torch.argsort)
lns_kthvalue = _create_lns_op_func('kthvalue', torch.kthvalue)
lns_maximum = _create_lns_op_func('maximum', torch.maximum)
lns_minimum = _create_lns_op_func('minimum', torch.minimum)
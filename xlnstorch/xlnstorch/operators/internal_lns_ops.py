from __future__ import annotations
import inspect
from typing import Callable, Any, Iterable, Tuple, Union, Optional

import torch
from xlnstorch import apply_lns_op

_KIND_MAP: dict[str, inspect._ParameterKind] = {
    "po": inspect.Parameter.POSITIONAL_ONLY,
    "pk": inspect.Parameter.POSITIONAL_OR_KEYWORD,
    "ko": inspect.Parameter.KEYWORD_ONLY,
    "*": inspect.Parameter.VAR_POSITIONAL,
    "**": inspect.Parameter.VAR_KEYWORD,
    "positional_only": inspect.Parameter.POSITIONAL_ONLY,
    "positional_or_keyword": inspect.Parameter.POSITIONAL_OR_KEYWORD,
    "keyword_only": inspect.Parameter.KEYWORD_ONLY,
    "var_positional": inspect.Parameter.VAR_POSITIONAL,
    "var_keyword": inspect.Parameter.VAR_KEYWORD,
}

def _build_signature(
        params: Iterable[Tuple],
        return_type: Any = inspect._empty,
    ):

    parameters: list[inspect.Parameter] = []
    for entry in params:
        if len(entry) not in (3, 4):
            raise ValueError(
                "Each parameter tuple must have 3 or 4 items "
                "(name, kind, annotation [, default])."
            )

        name, kind, annotation, *rest = entry
        default = rest[0] if rest else inspect._empty

        if isinstance(kind, str):
            try:
                kind = _KIND_MAP[kind.lower()]
            except KeyError as e:
                raise ValueError(f"Unknown parameter kind string '{kind}'.") from e

        parameters.append(
            inspect.Parameter(
                name,
                kind,
                annotation=annotation,
                default=default,
            )
        )

    return inspect.Signature(parameters, return_annotation=return_type)

def _create_lns_op_func(
    op_name: str,
    torch_op: Callable,
    *,
    docstring: Optional[str] = None,
    signature: Optional[inspect.Signature] = None,
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
    signature : inspect.Signature, optional
        Signature of the function, if provided.
    return_type : Any, optional
        Return type of the function, if provided.

    Returns
    -------
    Callable
        The created LNS operation function
    """
    def func(*args: Any, **kwargs: Any):
        return apply_lns_op(torch_op, *args, **kwargs)

    func.__name__ = f"lns_{op_name}"

    if docstring:
        func.__doc__ = docstring
    else:
        func.__doc__ = (
            f"See docs for :py:func:`{torch_op.__module__}.{torch_op.__name__}` for "
            f"parameter/return details. Typically, torch.Tensor arguments are the "
            f"equivalent internal representations of LNStensors."
        )

    if signature is not None:
        func.__signature__ = signature

    return func

lns_add = _create_lns_op_func('add', torch.add, signature=_build_signature([
    ("x", "pk", torch.Tensor),
    ("y", "pk", torch.Tensor),
    ("base", "pk", torch.Tensor)],
    torch.Tensor,
))
lns_sub = _create_lns_op_func('sub', torch.sub, signature=_build_signature([
    ("x", "pk", torch.Tensor),
    ("y", "pk", torch.Tensor),
    ("base", "pk", torch.Tensor)],
    torch.Tensor,
))
lns_mul = _create_lns_op_func('mul', torch.mul, signature=_build_signature([
    ("x", "pk", torch.Tensor),
    ("y", "pk", torch.Tensor),
    ("base", "pk", torch.Tensor)],
    torch.Tensor,
))
lns_div = _create_lns_op_func('div', torch.div, signature=_build_signature([
    ("x", "pk", torch.Tensor),
    ("y", "pk", torch.Tensor),
    ("base", "pk", torch.Tensor)],
    torch.Tensor,
))
lns_neg = _create_lns_op_func('neg', torch.neg, signature=_build_signature([
    ("x", "pk", torch.Tensor)],
    torch.Tensor,
))
lns_abs = _create_lns_op_func('abs', torch.abs, signature=_build_signature([
    ("x", "pk", torch.Tensor)],
    torch.Tensor,
))
lns_sqrt = _create_lns_op_func('sqrt', torch.sqrt, signature=_build_signature([
    ("x", "pk", torch.Tensor),
    ("base", "pk", torch.Tensor)],
    torch.Tensor,
))
lns_square = _create_lns_op_func('square', torch.square, signature=_build_signature([
    ("x", "pk", torch.Tensor),
    ("base", "pk", torch.Tensor)],
    torch.Tensor,
))
lns_pow = _create_lns_op_func('pow', torch.pow, signature=_build_signature([
    ("x", "pk", torch.Tensor),
    ("n", "pk", torch.Tensor),
    ("base", "pk", torch.Tensor)],
    torch.Tensor,
))
lns_exp = _create_lns_op_func('exp', torch.exp, signature=_build_signature([
    ("x", "pk", torch.Tensor),
    ("base", "pk", torch.Tensor)],
    torch.Tensor,
))
lns_log = _create_lns_op_func('log', torch.log, signature=_build_signature([
    ("x", "pk", torch.Tensor),
    ("base", "pk", torch.Tensor)],
    torch.Tensor,
))
lns_reciprocal = _create_lns_op_func('reciprocal', torch.reciprocal, signature=_build_signature([
    ("x", "pk", torch.Tensor),
    ("base", "pk", torch.Tensor)],
    torch.Tensor,
))
lns_sign = _create_lns_op_func('sign', torch.sign, signature=_build_signature([
    ("x", "pk", torch.Tensor),
    ("base", "pk", torch.Tensor)],
    torch.Tensor,
))
lns_positive = _create_lns_op_func('positive', torch.positive, signature=_build_signature([
    ("x", "pk", torch.Tensor)],
    torch.Tensor,
))
lns_sum = _create_lns_op_func('sum', torch.sum, signature=_build_signature([
    ("x", "pk", torch.Tensor),
    ("base", "pk", torch.Tensor),
    ("dim", "pk", Union[int, Tuple[int]], None),
    ("keepdim", "pk", bool, False)],
    torch.Tensor,
))
lns_prod = _create_lns_op_func('prod', torch.prod, signature=_build_signature([
    ("x", "pk", torch.Tensor),
    ("base", "pk", torch.Tensor),
    ("dim", "pk", Union[int, Tuple[int]], None),
    ("keepdim", "pk", bool, False)],
    torch.Tensor,
))
lns_mean = _create_lns_op_func('mean', torch.mean, signature=_build_signature([
    ("x", "pk", torch.Tensor),
    ("base", "pk", torch.Tensor),
    ("dim", "pk", Union[int, Tuple[int]], None),
    ("keepdim", "pk", bool, False)],
    torch.Tensor,
))
lns_var = _create_lns_op_func('var', torch.var, signature=_build_signature([
    ("x", "pk", torch.Tensor),
    ("base", "pk", torch.Tensor),
    ("correction", "pk", torch.Tensor),
    ("dim", "pk", Union[int, Tuple[int]], None),
    ("keepdim", "pk", bool, False)],
    torch.Tensor,
))
lns_matmul = _create_lns_op_func('matmul', torch.matmul, signature=_build_signature([
    ("A", "pk", torch.Tensor),
    ("B", "pk", torch.Tensor),
    ("base", "pk", torch.Tensor)],
    torch.Tensor,
))
lns_transpose = _create_lns_op_func('transpose', torch.transpose, signature=_build_signature([
    ("A", "pk", torch.Tensor),
    ("dim0", "pk", int),
    ("dim1", "pk", int)],
    torch.Tensor,
))

lns_equal = _create_lns_op_func('equal', torch.equal, signature=_build_signature([
    ("x", "pk", torch.Tensor),
    ("y", "pk", torch.Tensor)],
    bool,
))
lns_eq = _create_lns_op_func('eq', torch.eq, signature=_build_signature([
    ("x", "pk", torch.Tensor),
    ("y", "pk", torch.Tensor)],
    torch.Tensor,
))
lns_ne = _create_lns_op_func('ne', torch.ne, signature=_build_signature([
    ("x", "pk", torch.Tensor),
    ("y", "pk", torch.Tensor)],
    torch.Tensor,
))
lns_ge = _create_lns_op_func('ge', torch.ge, signature=_build_signature([
    ("x", "pk", torch.Tensor),
    ("y", "pk", torch.Tensor)],
    torch.Tensor,
))
lns_gt = _create_lns_op_func('gt', torch.gt, signature=_build_signature([
    ("x", "pk", torch.Tensor),
    ("y", "pk", torch.Tensor)],
    torch.Tensor,
))
lns_le = _create_lns_op_func('le', torch.le, signature=_build_signature([
    ("x", "pk", torch.Tensor),
    ("y", "pk", torch.Tensor)],
    torch.Tensor,
))
lns_lt = _create_lns_op_func('lt', torch.lt, signature=_build_signature([
    ("x", "pk", torch.Tensor),
    ("y", "pk", torch.Tensor)],
    torch.Tensor,
))
lns_isclose = _create_lns_op_func('isclose', torch.isclose, signature=_build_signature([
    ("x", "pk", torch.Tensor),
    ("y", "pk", torch.Tensor),
    ("base", "pk", torch.Tensor),
    ("rtol", "pk", torch.Tensor),
    ("atol", "pk", torch.Tensor)],
    torch.Tensor,
))
lns_allclose = _create_lns_op_func('allclose', torch.allclose, signature=_build_signature([
    ("x", "pk", torch.Tensor),
    ("y", "pk", torch.Tensor),
    ("base", "pk", torch.Tensor),
    ("rtol", "pk", torch.Tensor),
    ("atol", "pk", torch.Tensor)],
    bool,
))
lns_any = _create_lns_op_func('any', torch.any, signature=_build_signature([
    ("x", "pk", torch.Tensor),
    ("dim", "pk", Union[int, Tuple[int]], None),
    ("keepdim", "pk", bool, False)],
    torch.Tensor,
))
lns_all = _create_lns_op_func('all', torch.all, signature=_build_signature([
    ("x", "pk", torch.Tensor),
    ("dim", "pk", Union[int, Tuple[int]], None),
    ("keepdim", "pk", bool, False)],
    torch.Tensor,
))
lns_isin = _create_lns_op_func('isin', torch.isin, signature=_build_signature([
    ("x", "pk", torch.Tensor),
    ("y", "pk", torch.Tensor),
    ("assume_unique", "pk", bool, False),
    ("invert", "pk", bool, False)],
    torch.Tensor,
))
lns_sort = _create_lns_op_func('sort', torch.sort, signature=_build_signature([
    ("x", "pk", torch.Tensor),
    ("dim", "pk", int, -1),
    ("descending", "pk", bool, False),
    ("stable", "pk", bool, False)],
    torch.return_types.sort,
))
lns_argsort = _create_lns_op_func('argsort', torch.argsort, signature=_build_signature([
    ("x", "pk", torch.Tensor),
    ("dim", "pk", int, -1),
    ("descending", "pk", bool, False),
    ("stable", "pk", bool, False)],
    torch.Tensor,
))
lns_kthvalue = _create_lns_op_func('kthvalue', torch.kthvalue, signature=_build_signature([
    ("x", "pk", torch.Tensor),
    ("k", "pk", int),
    ("dim", "pk", int, -1),
    ("keepdim", "pk", bool, False)],
    torch.return_types.kthvalue,
))
lns_maximum = _create_lns_op_func('maximum', torch.maximum, signature=_build_signature([
    ("x", "pk", torch.Tensor),
    ("y", "pk", torch.Tensor),
    ("base", "pk", torch.Tensor)],
    torch.Tensor,
))
lns_minimum = _create_lns_op_func('minimum', torch.minimum, signature=_build_signature([
    ("x", "pk", torch.Tensor),
    ("y", "pk", torch.Tensor),
    ("base", "pk", torch.Tensor)],
    torch.Tensor,
))
lns_max = _create_lns_op_func('max', torch.max, signature=_build_signature([
    ("x", "pk", torch.Tensor),
    ("base", "pk", torch.Tensor),
    ("dim", "pk", Union[int, Tuple[int]], None),
    ("keepdim", "pk", bool, False)],
    Union[torch.return_types.max, torch.Tensor],
))
lns_argmax = _create_lns_op_func('argmax', torch.argmax, signature=_build_signature([
    ("x", "pk", torch.Tensor),
    ("dim", "pk", int, None),
    ("keepdim", "pk", bool, False)],
    torch.Tensor,
))
lns_min = _create_lns_op_func('min', torch.min, signature=_build_signature([
    ("x", "pk", torch.Tensor),
    ("base", "pk", torch.Tensor),
    ("dim", "pk", Union[int, Tuple[int]], None),
    ("keepdim", "pk", bool, False)],
    Union[torch.return_types.min, torch.Tensor],
))
lns_argmin = _create_lns_op_func('argmin', torch.argmin, signature=_build_signature([
    ("x", "pk", torch.Tensor),
    ("dim", "pk", int, None),
    ("keepdim", "pk", bool, False)],
    torch.Tensor,
))
lns_clamp = _create_lns_op_func('clamp', torch.clamp, signature=_build_signature([
    ("x", "pk", torch.Tensor),
    ("min", "pk", Optional[torch.Tensor], None),
    ("max", "pk", Optional[torch.Tensor], None)],
    torch.Tensor,
))

lns_broadcast_to = _create_lns_op_func('broadcast_to', torch.broadcast_to, signature=_build_signature([
    ("x", "pk", torch.Tensor),
    ("shape", "pk", Union[int, Tuple[int]]),
    ("base", "pk", torch.Tensor)],
    torch.Tensor,
))
lns_clone = _create_lns_op_func('clone', torch.clone, signature=_build_signature([
    ("x", "pk", torch.Tensor),
    ("memory_format", "pk", torch.memory_format, torch.preserve_format)],
    torch.Tensor,
))
lns_squeeze = _create_lns_op_func('squeeze', torch.squeeze, signature=_build_signature([
    ("x", "pk", torch.Tensor),
    ("dim", "pk", Optional[int], None)],
    torch.Tensor,
))
lns_unsqueeze = _create_lns_op_func('unsqueeze', torch.unsqueeze, signature=_build_signature([
    ("x", "pk", torch.Tensor),
    ("dim", "pk", int)],
    torch.Tensor,
))
lns_stack = _create_lns_op_func('stack', torch.stack, signature=_build_signature([
    ("dim", "pk", int),
    ("tensors", "*", Iterable[torch.Tensor])],
    torch.Tensor,
))
lns_cat = _create_lns_op_func('cat', torch.cat, signature=_build_signature([
    ("dim", "pk", int),
    ("tensors", "*", Iterable[torch.Tensor])],
    torch.Tensor,
))
lns_chunk = _create_lns_op_func('chunk', torch.chunk, signature=_build_signature([
    ("x", "pk", torch.Tensor),
    ("chunks", "pk", int),
    ("dim", "pk", int, 0)],
    Tuple[torch.Tensor, ...],
))
lns_where = _create_lns_op_func('where', torch.where, signature=_build_signature([
    ("condition", "pk", torch.Tensor),
    ("x", "pk", torch.Tensor),
    ("y", "pk", torch.Tensor),
    ("base", "pk", torch.Tensor)],
    torch.Tensor,
))
lns_pad = _create_lns_op_func('pad', torch.nn.functional.pad, signature=_build_signature([
    ("x", "pk", torch.Tensor),
    ("base", "pk", torch.Tensor),
    ("pad", "pk", Tuple[int]),
    ("mode", "pk", str, "constant"),
    ("value", "pk", Union[float, torch.Tensor, None], 0.0)],
    torch.Tensor,
))

lns_mse_loss = _create_lns_op_func('mse_loss', torch.nn.functional.mse_loss, signature=_build_signature([
    ("x", "pk", torch.Tensor),
    ("y", "pk", torch.Tensor),
    ("base", "pk", torch.Tensor),
    ("size_average", "pk", bool, None),
    ("reduce", "pk", bool, None),
    ("reduction", "pk", str, "mean"),
    ("weight", "pk", torch.Tensor, None)],
    torch.Tensor,
))
lns_l1_loss = _create_lns_op_func('l1_loss', torch.nn.functional.l1_loss, signature=_build_signature([
    ("x", "pk", torch.Tensor),
    ("y", "pk", torch.Tensor),
    ("base", "pk", torch.Tensor),
    ("size_average", "pk", bool, None),
    ("reduce", "pk", bool, None),
    ("reduction", "pk", str, "mean")],
    torch.Tensor,
))
lns_binary_cross_entropy = _create_lns_op_func('binary_cross_entropy', torch.nn.functional.binary_cross_entropy, signature=_build_signature([
    ("x", "pk", torch.Tensor),
    ("y", "pk", torch.Tensor),
    ("base", "pk", torch.Tensor),
    ("weight", "pk", torch.Tensor, None),
    ("size_average", "pk", bool, None),
    ("reduce", "pk", bool, None),
    ("reduction", "pk", str, "mean")],
    torch.Tensor,
))
lns_binary_cross_entropy_with_logits = _create_lns_op_func('binary_cross_entropy_with_logits', torch.nn.functional.binary_cross_entropy_with_logits, signature=_build_signature([
    ("x", "pk", torch.Tensor),
    ("y", "pk", torch.Tensor),
    ("base", "pk", torch.Tensor),
    ("weight", "pk", torch.Tensor, None),
    ("size_average", "pk", bool, None),
    ("reduce", "pk", bool, None),
    ("reduction", "pk", str, "mean"),
    ("pos_weight", "pk", torch.Tensor, None)],
    torch.Tensor,
))
lns_nll_loss = _create_lns_op_func('nll_loss', torch.nn.functional.nll_loss, signature=_build_signature([
    ("x", "pk", torch.Tensor),
    ("y", "pk", torch.Tensor),
    ("base", "pk", torch.Tensor),
    ("weight", "pk", torch.Tensor, None),
    ("size_average", "pk", bool, None),
    ("ignore_index", "pk", int, -100),
    ("reduce", "pk", bool, None),
    ("reduction", "pk", str, "mean")],
    torch.Tensor,
))
lns_poisson_nll_loss = _create_lns_op_func('poisson_nll_loss', torch.nn.functional.poisson_nll_loss, signature=_build_signature([
    ("x", "pk", torch.Tensor),
    ("y", "pk", torch.Tensor),
    ("base", "pk", torch.Tensor),
    ("log_input", "pk", bool, True),
    ("full", "pk", bool, False),
    ("size_average", "pk", bool, None),
    ("eps", "pk", torch.Tensor, 1e-8),
    ("reduce", "pk", bool, None),
    ("reduction", "pk", str, "mean")],
    torch.Tensor,
))
lns_hinge_embedding_loss = _create_lns_op_func('hinge_embedding_loss', torch.nn.functional.hinge_embedding_loss, signature=_build_signature([
    ("x", "pk", torch.Tensor),
    ("y", "pk", torch.Tensor),
    ("margin", "pk", torch.Tensor),
    ("base", "pk", torch.Tensor),
    ("size_average", "pk", bool, None),
    ("reduce", "pk", bool, None),
    ("reduction", "pk", str, "mean")],
    torch.Tensor,
))
lns_kl_div = _create_lns_op_func('kl_div', torch.nn.functional.kl_div, signature=_build_signature([
    ("x", "pk", torch.Tensor),
    ("y", "pk", torch.Tensor),
    ("base", "pk", torch.Tensor),
    ("size_average", "pk", bool, None),
    ("reduce", "pk", bool, None),
    ("reduction", "pk", str, "mean"),
    ("log_target", "pk", bool, False)],
    torch.Tensor,
))
lns_margin_ranking_loss = _create_lns_op_func('margin_ranking_loss', torch.nn.functional.margin_ranking_loss, signature=_build_signature([
    ("x1", "pk", torch.Tensor),
    ("x2", "pk", torch.Tensor),
    ("y", "pk", torch.Tensor),
    ("margin", "pk", torch.Tensor),
    ("base", "pk", torch.Tensor),
    ("size_average", "pk", bool, None),
    ("reduce", "pk", bool, None),
    ("reduction", "pk", str, "mean")],
    torch.Tensor,
))
lns_gaussian_nll_loss = _create_lns_op_func('gaussian_nll_loss', torch.nn.functional.gaussian_nll_loss, signature=_build_signature([
    ("x", "pk", torch.Tensor),
    ("y", "pk", torch.Tensor),
    ("var", "pk", torch.Tensor),
    ("eps", "pk", torch.Tensor),
    ("base", "pk", torch.Tensor),
    ("full", "pk", bool, False),
    ("reduction", "pk", str, "mean")],
    torch.Tensor,
))
lns_huber_loss = _create_lns_op_func('huber_loss', torch.nn.functional.huber_loss, signature=_build_signature([
    ("x", "pk", torch.Tensor),
    ("y", "pk", torch.Tensor),
    ("delta", "pk", torch.Tensor),
    ("base", "pk", torch.Tensor),
    ("reduction", "pk", str, "mean"),
    ("weight", "pk", torch.Tensor, None)],
    torch.Tensor,
))
lns_smooth_l1_loss = _create_lns_op_func('smooth_l1_loss', torch.nn.functional.smooth_l1_loss, signature=_build_signature([
    ("x", "pk", torch.Tensor),
    ("y", "pk", torch.Tensor),
    ("beta", "pk", torch.Tensor),
    ("base", "pk", torch.Tensor),
    ("size_average", "pk", bool, None),
    ("reduce", "pk", bool, None),
    ("reduction", "pk", str, "mean")],
    torch.Tensor,
))
lns_cross_entropy = _create_lns_op_func('cross_entropy', torch.nn.functional.cross_entropy, signature=_build_signature([
    ("x", "pk", torch.Tensor),
    ("target", "pk", torch.Tensor),
    ("base", "pk", torch.Tensor),
    ("weight", "pk", torch.Tensor, None),
    ("size_average", "pk", bool, None),
    ("ignore_index", "pk", int, -100),
    ("reduce", "pk", bool, None),
    ("reduction", "pk", str, "mean"),
    ("label_smoothing", "pk", float, 0.0)],
    torch.Tensor,
))

lns_relu = _create_lns_op_func('relu', torch.nn.functional.relu, signature=_build_signature([
    ("x", "pk", torch.Tensor),
    ("base", "pk", torch.Tensor)],
    torch.Tensor,
))
lns_relu_ = _create_lns_op_func('relu_', torch.nn.functional.relu_, signature=_build_signature([
    ("x", "pk", torch.Tensor),
    ("base", "pk", torch.Tensor)],
    torch.Tensor,
))
lns_leaky_relu = _create_lns_op_func('leaky_relu', torch.nn.functional.leaky_relu, signature=_build_signature([
    ("x", "pk", torch.Tensor),
    ("negative_slope", "pk", torch.Tensor),
    ("base", "pk", torch.Tensor)],
    torch.Tensor,
))
lns_leaky_relu_ = _create_lns_op_func('leaky_relu_', torch.nn.functional.leaky_relu_, signature=_build_signature([
    ("x", "pk", torch.Tensor),
    ("negative_slope", "pk", torch.Tensor),
    ("base", "pk", torch.Tensor)],
    torch.Tensor,
))
lns_threshold = _create_lns_op_func('threshold', torch.nn.functional.threshold, signature=_build_signature([
    ("x", "pk", torch.Tensor),
    ("threshold", "pk", torch.Tensor),
    ("value", "pk", torch.Tensor),
    ("base", "pk", torch.Tensor)],
    torch.Tensor,
))
lns_threshold_ = _create_lns_op_func('threshold_', torch.nn.functional.threshold_, signature=_build_signature([
    ("x", "pk", torch.Tensor),
    ("threshold", "pk", torch.Tensor),
    ("value", "pk", torch.Tensor),
    ("base", "pk", torch.Tensor)],
    torch.Tensor,
))
lns_tanh = _create_lns_op_func('tanh', torch.nn.functional.tanh, signature=_build_signature([
    ("x", "pk", torch.Tensor),
    ("base", "pk", torch.Tensor)],
    torch.Tensor,
))
lns_sigmoid = _create_lns_op_func('sigmoid', torch.nn.functional.sigmoid, signature=_build_signature([
    ("x", "pk", torch.Tensor),
    ("base", "pk", torch.Tensor)],
    torch.Tensor,
))
lns_logsigmoid = _create_lns_op_func('logsigmoid', torch.nn.functional.logsigmoid, signature=_build_signature([
    ("x", "pk", torch.Tensor),
    ("base", "pk", torch.Tensor)],
    torch.Tensor,
))
lns_softmin = _create_lns_op_func('softmin', torch.nn.functional.softmin, signature=_build_signature([
    ("x", "pk", torch.Tensor),
    ("base", "pk", torch.Tensor),
    ("dim", "pk", int, None)],
    torch.Tensor,
))
lns_softmax = _create_lns_op_func('softmax', torch.nn.functional.softmax, signature=_build_signature([
    ("x", "pk", torch.Tensor),
    ("base", "pk", torch.Tensor),
    ("dim", "pk", int, None)],
    torch.Tensor,
))
lns_log_softmax = _create_lns_op_func('log_softmax', torch.nn.functional.log_softmax, signature=_build_signature([
    ("x", "pk", torch.Tensor),
    ("base", "pk", torch.Tensor),
    ("dim", "pk", int, None)],
    torch.Tensor,
))
lns_hardtanh = _create_lns_op_func('hardtanh', torch.nn.functional.hardtanh, signature=_build_signature([
    ("x", "pk", torch.Tensor),
    ("min_val", "pk", torch.Tensor),
    ("max_val", "pk", torch.Tensor),
    ("base", "pk", torch.Tensor)],
    torch.Tensor,
))
lns_hardswish = _create_lns_op_func('hardswish', torch.nn.functional.hardswish, signature=_build_signature([
    ("x", "pk", torch.Tensor),
    ("base", "pk", torch.Tensor)],
    torch.Tensor,
))
lns_elu = _create_lns_op_func('elu', torch.nn.functional.elu, signature=_build_signature([
    ("x", "pk", torch.Tensor),
    ("alpha", "pk", torch.Tensor),
    ("base", "pk", torch.Tensor)],
    torch.Tensor,
))
lns_selu = _create_lns_op_func('selu', torch.nn.functional.selu, signature=_build_signature([
    ("x", "pk", torch.Tensor),
    ("base", "pk", torch.Tensor)],
    torch.Tensor,
))
lns_celu = _create_lns_op_func('celu', torch.nn.functional.celu, signature=_build_signature([
    ("x", "pk", torch.Tensor),
    ("alpha", "pk", torch.Tensor),
    ("base", "pk", torch.Tensor)],
    torch.Tensor,
))
lns_prelu = _create_lns_op_func('prelu', torch.nn.functional.prelu, signature=_build_signature([
    ("x", "pk", torch.Tensor),
    ("a", "pk", torch.Tensor),
    ("base", "pk", torch.Tensor)],
    torch.Tensor,
))
lns_rrelu = _create_lns_op_func('rrelu', torch.nn.functional.rrelu, signature=_build_signature([
    ("x", "pk", torch.Tensor),
    ("a", "pk", torch.Tensor),
    ("base", "pk", torch.Tensor)],
    torch.Tensor,
))
lns_glu = _create_lns_op_func('glu', torch.nn.functional.glu, signature=_build_signature([
    ("x", "pk", torch.Tensor),
    ("base", "pk", torch.Tensor),
    ("dim", "pk", int, -1)],
    torch.Tensor,
))
lns_hardshrink = _create_lns_op_func('hardshrink', torch.nn.functional.hardshrink, signature=_build_signature([
    ("x", "pk", torch.Tensor),
    ("lambd", "pk", torch.Tensor),
    ("base", "pk", torch.Tensor)],
    torch.Tensor,
))
lns_tanhshrink = _create_lns_op_func('tanhshrink', torch.nn.functional.tanhshrink, signature=_build_signature([
    ("x", "pk", torch.Tensor),
    ("base", "pk", torch.Tensor)],
    torch.Tensor,
))
lns_softsign = _create_lns_op_func('softsign', torch.nn.functional.softsign, signature=_build_signature([
    ("x", "pk", torch.Tensor),
    ("base", "pk", torch.Tensor)],
    torch.Tensor,
))
lns_softplus = _create_lns_op_func('softplus', torch.nn.functional.softplus, signature=_build_signature([
    ("x", "pk", torch.Tensor),
    ("beta", "pk", torch.Tensor),
    ("threshold", "pk", torch.Tensor),
    ("base", "pk", torch.Tensor)],
    torch.Tensor,
))
lns_softshrink = _create_lns_op_func('softshrink', torch.nn.functional.softshrink, signature=_build_signature([
    ("x", "pk", torch.Tensor),
    ("lambd", "pk", torch.Tensor),
    ("base", "pk", torch.Tensor)],
    torch.Tensor,
))
lns_hardsigmoid = _create_lns_op_func('hardsigmoid', torch.nn.functional.hardsigmoid, signature=_build_signature([
    ("x", "pk", torch.Tensor),
    ("base", "pk", torch.Tensor)],
    torch.Tensor,
))
lns_silu = _create_lns_op_func('silu', torch.nn.functional.silu, signature=_build_signature([
    ("x", "pk", torch.Tensor),
    ("base", "pk", torch.Tensor)],
    torch.Tensor,
))

lns_linear = _create_lns_op_func('linear', torch.nn.functional.linear, signature=_build_signature([
    ("x", "pk", torch.Tensor),
    ("A", "pk", torch.Tensor),
    ("base", "pk", torch.Tensor),
    ("bias", "pk", torch.Tensor, None)],
    torch.Tensor,
))
lns_bilinear = _create_lns_op_func('bilinear', torch.nn.functional.bilinear, signature=_build_signature([
    ("x", "pk", torch.Tensor),
    ("y", "pk", torch.Tensor),
    ("A", "pk", torch.Tensor),
    ("base", "pk", torch.Tensor),
    ("bias", "pk", torch.Tensor, None)],
    torch.Tensor,
))
lns_dropout = _create_lns_op_func('dropout', torch.nn.functional.dropout, signature=_build_signature([
    ("x", "pk", torch.Tensor),
    ("base", "pk", torch.Tensor),
    ("p", "pk", float, 0.5)],
    torch.Tensor,
))
lns_dropout1d = _create_lns_op_func('dropout1d', torch.nn.functional.dropout1d, signature=_build_signature([
    ("x", "pk", torch.Tensor),
    ("base", "pk", torch.Tensor),
    ("p", "pk", float, 0.5)],
    torch.Tensor,
))
lns_dropout2d = _create_lns_op_func('dropout2d', torch.nn.functional.dropout2d, signature=_build_signature([
    ("x", "pk", torch.Tensor),
    ("base", "pk", torch.Tensor),
    ("p", "pk", float, 0.5)],
    torch.Tensor,
))
lns_dropout3d = _create_lns_op_func('dropout3d', torch.nn.functional.dropout3d, signature=_build_signature([
    ("x", "pk", torch.Tensor),
    ("base", "pk", torch.Tensor),
    ("p", "pk", float, 0.5)],
    torch.Tensor,
))
lns_conv1d = _create_lns_op_func('conv1d', torch.nn.functional.conv1d, signature=_build_signature([
    ("x", "pk", torch.Tensor),
    ("weight", "pk", torch.Tensor),
    ("bias", "pk", torch.Tensor),
    ("base", "pk", torch.Tensor),
    ("stride", "pk", int, 1),
    ("padding", "pk", int, 0),
    ("dilation", "pk", int, 1),
    ("groups", "pk", int, 1)],
    torch.Tensor,
))
lns_conv2d = _create_lns_op_func('conv2d', torch.nn.functional.conv2d, signature=_build_signature([
    ("x", "pk", torch.Tensor),
    ("weight", "pk", torch.Tensor),
    ("bias", "pk", torch.Tensor),
    ("base", "pk", torch.Tensor),
    ("stride", "pk", Union[int, Tuple[int]], 1),
    ("padding", "pk", Union[int, Tuple[int]], 0),
    ("dilation", "pk", Union[int, Tuple[int]], 1),
    ("groups", "pk", int, 1)],
    torch.Tensor,
))
lns_conv3d = _create_lns_op_func('conv3d', torch.nn.functional.conv3d, signature=_build_signature([
    ("x", "pk", torch.Tensor),
    ("weight", "pk", torch.Tensor),
    ("bias", "pk", torch.Tensor),
    ("base", "pk", torch.Tensor),
    ("stride", "pk", Union[int, Tuple[int]], 1),
    ("padding", "pk", Union[int, Tuple[int]], 0),
    ("dilation", "pk", Union[int, Tuple[int]], 1),
    ("groups", "pk", int, 1)],
    torch.Tensor,
))
lns_avg_pool1d = _create_lns_op_func('avg_pool1d', torch.nn.functional.avg_pool1d, signature=_build_signature([
    ("x", "pk", torch.Tensor),
    ("kernel_size", "pk", int),
    ("stride", "pk", Union[int, Tuple[int], None], None),
    ("padding", "pk", Union[int, Tuple[int]], 0),
    ("ceil_mode", "pk", bool, False),
    ("count_include_pad", "pk", bool, True)],
    torch.Tensor,
))
lns_avg_pool2d = _create_lns_op_func('avg_pool2d', torch.nn.functional.avg_pool2d, signature=_build_signature([
    ("x", "pk", torch.Tensor),
    ("kernel_size", "pk", int),
    ("stride", "pk", Union[int, Tuple[int], None], None),
    ("padding", "pk", Union[int, Tuple[int]], 0),
    ("ceil_mode", "pk", bool, False),
    ("count_include_pad", "pk", bool, True),
    ("divisor_override", "pk", Optional[torch.Tensor], None)],
    torch.Tensor,
))
lns_avg_pool3d = _create_lns_op_func('avg_pool3d', torch.nn.functional.avg_pool3d, signature=_build_signature([
    ("x", "pk", torch.Tensor),
    ("kernel_size", "pk", int),
    ("stride", "pk", Union[int, Tuple[int], None], None),
    ("padding", "pk", Union[int, Tuple[int]], 0),
    ("ceil_mode", "pk", bool, False),
    ("count_include_pad", "pk", bool, True),
    ("divisor_override", "pk", Optional[torch.Tensor], None)],
    torch.Tensor,
))
lns_adaptive_avg_pool1d = _create_lns_op_func('adaptive_avg_pool1d', torch.nn.functional.adaptive_avg_pool1d, signature=_build_signature([
    ("x", "pk", torch.Tensor),
    ("output_size", "pk", Union[int, Tuple[int]])],
    torch.Tensor,
))
lns_adaptive_avg_pool2d = _create_lns_op_func('adaptive_avg_pool2d', torch.nn.functional.adaptive_avg_pool2d, signature=_build_signature([
    ("x", "pk", torch.Tensor),
    ("output_size", "pk", Union[int, Tuple[int, int]])],
    torch.Tensor,
))
lns_adaptive_avg_pool3d = _create_lns_op_func('adaptive_avg_pool3d', torch.nn.functional.adaptive_avg_pool3d, signature=_build_signature([
    ("x", "pk", torch.Tensor),
    ("output_size", "pk", Union[int, Tuple[int, int, int]])],
    torch.Tensor,
))
lns_batch_norm = _create_lns_op_func('batch_norm', torch.nn.functional.batch_norm, signature=_build_signature([
    ("x", "pk", torch.Tensor),
    ("running_mean", "pk", torch.Tensor),
    ("running_var", "pk", torch.Tensor),
    ("momentum", "pk", torch.Tensor),
    ("eps", "pk", torch.Tensor),
    ("base", "pk", torch.Tensor),
    ("weight", "pk", Optional[torch.Tensor], None),
    ("bias", "pk", Optional[torch.Tensor], None),
    ("training", "pk", bool, False)],
    torch.Tensor,
))
lns_layer_norm = _create_lns_op_func('layer_norm', torch.nn.functional.layer_norm, signature=_build_signature([
    ("x", "pk", torch.Tensor),
    ("normalized_shape", "pk", Union[int, Tuple[int]]),
    ("base", "pk", torch.Tensor),
    ("weight", "pk", Optional[torch.Tensor], None),
    ("bias", "pk", Optional[torch.Tensor], None),
    ("eps", "pk", torch.Tensor, 1e-5)],
    torch.Tensor,
))
lns_max_pool1d = _create_lns_op_func('max_pool1d', torch.nn.functional.max_pool1d, signature=_build_signature([
    ("x", "pk", torch.Tensor),
    ("kernel_size", "pk", Union[int, Tuple[int]]),
    ("base", "pk", torch.Tensor),
    ("stride", "pk", Union[int, Tuple[int], None], None),
    ("padding", "pk", Union[int, Tuple[int]], 0),
    ("dilation", "pk", Union[int, Tuple[int]], 1),
    ("ceil_mode", "pk", bool, False),
    ("return_indices", "pk", bool, False)],
    torch.Tensor,
))
lns_max_pool2d = _create_lns_op_func('max_pool2d', torch.nn.functional.max_pool2d, signature=_build_signature([
    ("x", "pk", torch.Tensor),
    ("kernel_size", "pk", Union[int, Tuple[int]]),
    ("base", "pk", torch.Tensor),
    ("stride", "pk", Union[int, Tuple[int], None], None),
    ("padding", "pk", Union[int, Tuple[int]], 0),
    ("dilation", "pk", Union[int, Tuple[int]], 1),
    ("ceil_mode", "pk", bool, False),
    ("return_indices", "pk", bool, False)],
    torch.Tensor,
))
lns_max_pool3d = _create_lns_op_func('max_pool3d', torch.nn.functional.max_pool3d, signature=_build_signature([
    ("x", "pk", torch.Tensor),
    ("kernel_size", "pk", Union[int, Tuple[int]]),
    ("base", "pk", torch.Tensor),
    ("stride", "pk", Union[int, Tuple[int], None], None),
    ("padding", "pk", Union[int, Tuple[int]], 0),
    ("dilation", "pk", Union[int, Tuple[int]], 1),
    ("ceil_mode", "pk", bool, False),
    ("return_indices", "pk", bool, False)],
    torch.Tensor,
))
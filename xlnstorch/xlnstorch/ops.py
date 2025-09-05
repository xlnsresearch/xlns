from __future__ import annotations
import inspect
from typing import Callable, Any, Iterable, Tuple, Union, Optional

import torch
from xlnstorch import apply_lns_op, LNS_ZERO, LNS_ONE
from xlnstorch.tensor_utils import float_to_lns_forward, float_to_lns_backward

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
        include_self: bool = True,
    ):

    if include_self:
        params = (("self", "pk", None),) + tuple(params)

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
            f"more information. LongTensor parameters/return types refer to int64 "
            "internal representation tensors."
        )

    if signature is not None:
        func.__signature__ = signature

    return func

class LNSOps:

    def __init__(self, base: torch.Tensor):
        self.base = base

    # ====================
    #    Helper Methods
    # ====================

    def to_lns(self, value: Union[int, float, torch.Tensor]) -> torch.LongTensor:
        tensor_value = torch.tensor(value, dtype=torch.float64) if not isinstance(value, torch.Tensor) else value.to(torch.float64)
        return float_to_lns_forward(tensor_value, self.base)

    def from_lns(self, lns_value: torch.LongTensor) -> torch.Tensor:
        return float_to_lns_backward(lns_value, self.base)

    def zeros(self, *size: int):
        return torch.full(size, LNS_ZERO.item(), dtype=torch.int64)

    def zeros_like(self, tensor: torch.LongTensor):
        return torch.full_like(tensor, LNS_ZERO.item(), dtype=torch.int64)

    def ones(self, *size: int):
        return torch.full(size, LNS_ONE.item(), dtype=torch.int64)

    def ones_like(self, tensor: torch.LongTensor):
        return torch.full_like(tensor, LNS_ONE.item(), dtype=torch.int64)

    def full(self, *size: int, fill_value: Union[int, float]):
        lns_fill_value = self.to_lns(fill_value)
        return torch.full(size, lns_fill_value.item(), dtype=torch.int64)

    def full_like(self, tensor: torch.LongTensor, fill_value: Union[int, float]):
        lns_fill_value = self.to_lns(fill_value)
        return torch.full_like(tensor, lns_fill_value.item(), dtype=torch.int64)

    def sum_to_size(self, tensor: torch.LongTensor, target_size: torch.Size) -> torch.LongTensor:
        """
        Sum-reduce a tensor to a target size by summing over excess dimensions.

        Parameters
        ----------
        tensor : torch.LongTensor
            The input internal representation tensor to be reduced.
        target_size : torch.Size
            The desired target size after reduction.

        Returns
        -------
        torch.LongTensor
            The reduced internal representation tensor with the specified target size.

        Raises
        ------
        ValueError
            If the target size is not compatible with the input tensor size.
        """
        if list(tensor.shape) == list(target_size):
            return tensor

        tensor_shape = list(tensor.shape)
        tgt_shape = list(target_size)
        if tensor.dim() > len(tgt_shape):
            tgt_shape = [1] * (tensor.dim() - len(tgt_shape)) + tgt_shape

        # reduce dimensions that were broadcasted
        leading = tensor.dim() - len(tgt_shape)
        if leading > 0:
            tensor = self.sum(tensor, dim=tuple(range(leading)), keepdim=False)
            tensor_shape = tensor_shape[leading:]

        # reduce dimensions where target size is 1 but tensor has a larger size
        reduce_dims = [i for i, (ts, gs) in enumerate(zip(tensor_shape, tgt_shape)) if gs == 1 and ts != 1]
        if reduce_dims:
            tensor = self.sum(tensor, dim=tuple(reduce_dims), keepdim=True)

        return tensor.reshape(target_size)



    # ===========================
    #    Arithmetic Operations
    # ===========================

    add = _create_lns_op_func('add', torch.add, signature=_build_signature([
        ("x", "pk", torch.LongTensor),
        ("y", "pk", torch.LongTensor)],
        torch.LongTensor,
    ))

    sub = _create_lns_op_func('sub', torch.sub, signature=_build_signature([
        ("x", "pk", torch.LongTensor),
        ("y", "pk", torch.LongTensor)],
        torch.LongTensor,
    ))

    mul = _create_lns_op_func('mul', torch.mul, signature=_build_signature([
        ("x", "pk", torch.LongTensor),
        ("y", "pk", torch.LongTensor)],
        torch.LongTensor,
    ))

    div = _create_lns_op_func('div', torch.div, signature=_build_signature([
        ("x", "pk", torch.LongTensor),
        ("y", "pk", torch.LongTensor)],
        torch.LongTensor,
    ))

    neg = _create_lns_op_func('neg', torch.neg, signature=_build_signature([
        ("x", "pk", torch.LongTensor)],
        torch.LongTensor,
    ))

    abs = _create_lns_op_func('abs', torch.abs, signature=_build_signature([
        ("x", "pk", torch.LongTensor)],
        torch.LongTensor,
    ))

    sqrt = _create_lns_op_func('sqrt', torch.sqrt, signature=_build_signature([
        ("x", "pk", torch.LongTensor)],
        torch.LongTensor,
    ))

    square = _create_lns_op_func('square', torch.square, signature=_build_signature([
        ("x", "pk", torch.LongTensor)],
        torch.LongTensor,
    ))

    pow = _create_lns_op_func('pow', torch.pow, signature=_build_signature([
        ("x", "pk", torch.LongTensor),
        ("n", "pk", torch.Tensor)],
        torch.LongTensor,
    ))

    exp = _create_lns_op_func('exp', torch.exp, signature=_build_signature([
        ("x", "pk", torch.LongTensor)],
        torch.LongTensor,
    ))

    log = _create_lns_op_func('log', torch.log, signature=_build_signature([
        ("x", "pk", torch.LongTensor)],
        torch.LongTensor,
    ))

    reciprocal = _create_lns_op_func('reciprocal', torch.reciprocal, signature=_build_signature([
        ("x", "pk", torch.LongTensor)],
        torch.LongTensor,
    ))

    sign = _create_lns_op_func('sign', torch.sign, signature=_build_signature([
        ("x", "pk", torch.LongTensor)],
        torch.LongTensor,
    ))

    positive = _create_lns_op_func('positive', torch.positive, signature=_build_signature([
        ("x", "pk", torch.LongTensor)],
        torch.LongTensor,
    ))

    sum = _create_lns_op_func('sum', torch.sum, signature=_build_signature([
        ("x", "pk", torch.LongTensor),
        ("dim", "ko", Union[int, Tuple[int, ...]], None),
        ("keepdim", "ko", bool, False)],
        torch.LongTensor,
    ))

    prod = _create_lns_op_func('prod', torch.prod, signature=_build_signature([
        ("x", "pk", torch.LongTensor),
        ("dim", "pk", Union[int, Tuple[int, ...]], None),
        ("keepdim", "pk", bool, False)],
        torch.LongTensor,
    ))

    mean = _create_lns_op_func('mean', torch.mean, signature=_build_signature([
        ("x", "pk", torch.LongTensor),
        ("dim", "pk", Union[int, Tuple[int, ...]], None),
        ("keepdim", "pk", bool, False)],
        torch.LongTensor,
    ))

    var = _create_lns_op_func('var', torch.var, signature=_build_signature([
        ("x", "pk", torch.LongTensor),
        ("correction", "pk", torch.LongTensor),
        ("dim", "pk", Union[int, Tuple[int, ...]], None),
        ("keepdim", "pk", bool, False)],
        torch.LongTensor,
    ))

    matmul = _create_lns_op_func('matmul', torch.matmul, signature=_build_signature([
        ("A", "pk", torch.LongTensor),
        ("B", "pk", torch.LongTensor)],
        torch.LongTensor,
    ))

    transpose = _create_lns_op_func('transpose', torch.transpose, signature=_build_signature([
        ("A", "pk", torch.LongTensor),
        ("dim0", "pk", int),
        ("dim1", "pk", int)],
        torch.LongTensor,
    ))



    # ===========================
    #    Comparison Operations
    # ===========================

    equal = _create_lns_op_func('eq', torch.equal, signature=_build_signature([
        ("x", "pk", torch.LongTensor),
        ("y", "pk", torch.LongTensor)],
        bool,
    ))

    eq = _create_lns_op_func('eq', torch.eq, signature=_build_signature([
        ("x", "pk", torch.LongTensor),
        ("y", "pk", torch.LongTensor)],
        torch.BoolTensor,
    ))

    ne = _create_lns_op_func('ne', torch.ne, signature=_build_signature([
        ("x", "pk", torch.LongTensor),
        ("y", "pk", torch.LongTensor)],
        torch.BoolTensor,
    ))

    ge = _create_lns_op_func('ge', torch.ge, signature=_build_signature([
        ("x", "pk", torch.LongTensor),
        ("y", "pk", torch.LongTensor)],
        torch.BoolTensor,
    ))

    gt = _create_lns_op_func('gt', torch.gt, signature=_build_signature([
        ("x", "pk", torch.LongTensor),
        ("y", "pk", torch.LongTensor)],
        torch.BoolTensor,
    ))

    le = _create_lns_op_func('le', torch.le, signature=_build_signature([
        ("x", "pk", torch.LongTensor),
        ("y", "pk", torch.LongTensor)],
        torch.BoolTensor,
    ))

    lt = _create_lns_op_func('lt', torch.lt, signature=_build_signature([
        ("x", "pk", torch.LongTensor),
        ("y", "pk", torch.LongTensor)],
        torch.BoolTensor,
    ))

    isclose = _create_lns_op_func('isclose', torch.isclose, signature=_build_signature([
        ("x", "pk", torch.LongTensor),
        ("y", "pk", torch.LongTensor),
        ("rtol", "pk", torch.LongTensor),
        ("atol", "pk", torch.LongTensor)],
        torch.BoolTensor,
    ))

    allclose = _create_lns_op_func('allclose', torch.allclose, signature=_build_signature([
        ("x", "pk", torch.LongTensor),
        ("y", "pk", torch.LongTensor),
        ("rtol", "pk", torch.LongTensor),
        ("atol", "pk", torch.LongTensor)],
        bool,
    ))

    any = _create_lns_op_func('any', torch.any, signature=_build_signature([
        ("x", "pk", torch.BoolTensor),
        ("dim", "ko", Union[int, Tuple[int, ...]], None),
        ("keepdim", "ko", bool, False)],
        torch.BoolTensor,
    ))

    all = _create_lns_op_func('all', torch.all, signature=_build_signature([
        ("x", "pk", torch.BoolTensor),
        ("dim", "ko", Union[int, Tuple[int, ...]], None),
        ("keepdim", "ko", bool, False)],
        torch.BoolTensor,
    ))

    isin = _create_lns_op_func('isin', torch.isin, signature=_build_signature([
        ("x", "pk", torch.LongTensor),
        ("y", "pk", torch.LongTensor),
        ("assume_unique", "pk", bool, False),
        ("invert", "pk", bool, False)],
        torch.BoolTensor,
    ))

    sort = _create_lns_op_func('sort', torch.sort, signature=_build_signature([
        ("x", "pk", torch.LongTensor),
        ("dim", "pk", int, -1),
        ("descending", "pk", bool, False),
        ("stable", "pk", bool, False)],
        torch.return_types.sort,
    ))

    argsort = _create_lns_op_func('argsort', torch.argsort, signature=_build_signature([
        ("x", "pk", torch.LongTensor),
        ("dim", "pk", int, -1),
        ("descending", "pk", bool, False),
        ("stable", "pk", bool, False)],
        torch.Tensor,
    ))

    kthvalue = _create_lns_op_func('kthvalue', torch.kthvalue, signature=_build_signature([
        ("x", "pk", torch.LongTensor),
        ("k", "pk", int),
        ("dim", "pk", int, -1),
        ("keepdim", "pk", bool, False)],
        torch.return_types.kthvalue,
    ))

    maximum = _create_lns_op_func('maximum', torch.maximum, signature=_build_signature([
        ("x", "pk", torch.LongTensor),
        ("y", "pk", torch.LongTensor)],
        torch.LongTensor,
    ))

    minimum = _create_lns_op_func('minimum', torch.minimum, signature=_build_signature([
        ("x", "pk", torch.LongTensor),
        ("y", "pk", torch.LongTensor)],
        torch.LongTensor,
    ))

    max = _create_lns_op_func('max', torch.max, signature=_build_signature([
        ("x", "pk", torch.LongTensor),
        ("dim", "pk", Union[int, Tuple[int]], None),
        ("keepdim", "pk", bool, False)],
        Union[torch.return_types.max, torch.LongTensor],
    ))

    argmax = _create_lns_op_func('argmax', torch.argmax, signature=_build_signature([
        ("x", "pk", torch.LongTensor),
        ("dim", "pk", int, None),
        ("keepdim", "pk", bool, False)],
        torch.Tensor,
    ))

    min = _create_lns_op_func('min', torch.min, signature=_build_signature([
        ("x", "pk", torch.LongTensor),
        ("dim", "pk", Union[int, Tuple[int]], None),
        ("keepdim", "pk", bool, False)],
        Union[torch.return_types.min, torch.LongTensor],
    ))

    argmin = _create_lns_op_func('argmin', torch.argmin, signature=_build_signature([
        ("x", "pk", torch.LongTensor),
        ("dim", "pk", int, None),
        ("keepdim", "pk", bool, False)],
        torch.Tensor,
    ))

    clamp = _create_lns_op_func('clamp', torch.clamp, signature=_build_signature([
        ("x", "pk", torch.LongTensor),
        ("min", "pk", Optional[torch.LongTensor], None),
        ("max", "pk", Optional[torch.LongTensor], None)],
        torch.LongTensor,
    ))

    # ==============================
    #    Miscellaneous Operations
    # ==============================

    broadcast_to = _create_lns_op_func('broadcast_to', torch.broadcast_to, signature=_build_signature([
        ("x", "pk", torch.LongTensor),
        ("shape", "pk", Union[int, Tuple[int]])],
        torch.LongTensor,
    ))

    clone = _create_lns_op_func('clone', torch.clone, signature=_build_signature([
        ("x", "pk", torch.LongTensor),
        ("memory_format", "pk", torch.memory_format, torch.preserve_format)],
        torch.LongTensor,
    ))

    squeeze = _create_lns_op_func('squeeze', torch.squeeze, signature=_build_signature([
        ("x", "pk", torch.LongTensor),
        ("dim", "pk", Optional[int], None)],
        torch.LongTensor,
    ))

    unsqueeze = _create_lns_op_func('unsqueeze', torch.unsqueeze, signature=_build_signature([
        ("x", "pk", torch.LongTensor),
        ("dim", "pk", int)],
        torch.LongTensor,
    ))

    stack = _create_lns_op_func('stack', torch.stack, signature=_build_signature([
        ("dim", "pk", int),
        ("tensors", "*", Iterable[torch.LongTensor])],
        torch.LongTensor,
    ))

    cat = _create_lns_op_func('cat', torch.cat, signature=_build_signature([
        ("dim", "pk", int),
        ("tensors", "*", Iterable[torch.LongTensor])],
        torch.LongTensor,
    ))

    chunk = _create_lns_op_func('chunk', torch.chunk, signature=_build_signature([
        ("x", "pk", torch.LongTensor),
        ("chunks", "pk", int),
        ("dim", "pk", int, 0)],
        Tuple[torch.LongTensor, ...],
    ))

    where = _create_lns_op_func('where', torch.where, signature=_build_signature([
        ("condition", "pk", torch.BoolTensor),
        ("x", "pk", torch.LongTensor),
        ("y", "pk", torch.LongTensor)],
        torch.LongTensor,
    ))

    pad = _create_lns_op_func('pad', torch.nn.functional.pad, signature=_build_signature([
        ("x", "pk", torch.LongTensor),
        ("pad", "pk", Tuple[int]),
        ("mode", "pk", str, "constant"),
        ("value", "pk", Union[float, torch.LongTensor, None], 0.0)],
        torch.LongTensor,
    ))



    # ====================
    #    Loss Functions
    # ====================

    mse_loss = _create_lns_op_func('mse_loss',
        torch.nn.functional.mse_loss, signature=_build_signature([
        ("x", "pk", torch.LongTensor),
        ("y", "pk", torch.LongTensor),
        ("reduction", "pk", str, "mean"),
        ("weight", "pk", Optional[torch.LongTensor], None)],
        torch.LongTensor,
    ))

    l1_loss = _create_lns_op_func('l1_loss',
        torch.nn.functional.l1_loss, signature=_build_signature([
        ("x", "pk", torch.LongTensor),
        ("y", "pk", torch.LongTensor),
        ("reduction", "pk", str, "mean")],
        torch.LongTensor,
    ))

    binary_cross_entropy = _create_lns_op_func('binary_cross_entropy',
        torch.nn.functional.binary_cross_entropy, signature=_build_signature([
        ("x", "pk", torch.LongTensor),
        ("y", "pk", torch.LongTensor),
        ("weight", "pk", Optional[torch.LongTensor], None),
        ("reduction", "pk", str, "mean")],
        torch.LongTensor,
    ))

    binary_cross_entropy_with_logits = _create_lns_op_func('binary_cross_entropy_with_logits',
        torch.nn.functional.binary_cross_entropy_with_logits, signature=_build_signature([
        ("x", "pk", torch.LongTensor),
        ("y", "pk", torch.LongTensor),
        ("weight", "pk", Optional[torch.LongTensor], None),
        ("reduction", "pk", str, "mean"),
        ("pos_weight", "pk", Optional[torch.LongTensor], None)],
        torch.LongTensor,
    ))

    nll_loss = _create_lns_op_func('nll_loss',
        torch.nn.functional.nll_loss, signature=_build_signature([
        ("x", "pk", torch.LongTensor),
        ("y", "pk", torch.Tensor),
        ("weight", "pk", Optional[torch.LongTensor], None),
        ("reduction", "pk", str, "mean")],
        torch.LongTensor,
    ))

    poisson_nll_loss = _create_lns_op_func('poisson_nll_loss',
        torch.nn.functional.poisson_nll_loss, signature=_build_signature([
        ("x", "pk", torch.LongTensor),
        ("y", "pk", torch.LongTensor),
        ("eps", "pk", torch.LongTensor),
        ("log_input", "pk", bool, True),
        ("full", "pk", bool, False),
        ("reduction", "pk", str, "mean")],
        torch.LongTensor,
    ))

    hinge_embedding_loss = _create_lns_op_func('hinge_embedding_loss',
        torch.nn.functional.hinge_embedding_loss, signature=_build_signature([
        ("x", "pk", torch.LongTensor),
        ("y", "pk", torch.LongTensor),
        ("margin", "pk", torch.LongTensor),
        ("reduction", "pk", str, "mean")],
        torch.LongTensor,
    ))

    kl_div = _create_lns_op_func('kl_div',
        torch.nn.functional.kl_div, signature=_build_signature([
        ("x", "pk", torch.LongTensor),
        ("y", "pk", torch.LongTensor),
        ("reduction", "pk", str, "mean"),
        ("log_target", "pk", bool, False)],
        torch.LongTensor,
    ))

    margin_ranking_loss = _create_lns_op_func('margin_ranking_loss',
        torch.nn.functional.margin_ranking_loss, signature=_build_signature([
        ("x1", "pk", torch.LongTensor),
        ("x2", "pk", torch.LongTensor),
        ("y", "pk", torch.LongTensor),
        ("margin", "pk", torch.LongTensor),
        ("reduction", "pk", str, "mean")],
        torch.LongTensor,
    ))

    gaussian_nll_loss = _create_lns_op_func('gaussian_nll_loss',
        torch.nn.functional.gaussian_nll_loss, signature=_build_signature([
        ("x", "pk", torch.LongTensor),
        ("y", "pk", torch.LongTensor),
        ("var", "pk", torch.LongTensor),
        ("eps", "pk", torch.LongTensor),
        ("full", "pk", bool, False),
        ("reduction", "pk", str, "mean")],
        torch.LongTensor,
    ))

    huber_loss = _create_lns_op_func('huber_loss',
        torch.nn.functional.huber_loss, signature=_build_signature([
        ("x", "pk", torch.LongTensor),
        ("y", "pk", torch.LongTensor),
        ("delta", "pk", torch.LongTensor),
        ("reduction", "pk", str, "mean"),
        ("weight", "pk", Optional[torch.LongTensor], None)],
        torch.LongTensor,
    ))

    smooth_l1_loss = _create_lns_op_func('smooth_l1_loss',
        torch.nn.functional.smooth_l1_loss, signature=_build_signature([
        ("x", "pk", torch.LongTensor),
        ("y", "pk", torch.LongTensor),
        ("beta", "pk", torch.LongTensor),
        ("reduction", "pk", str, "mean")],
        torch.LongTensor,
    ))

    cross_entropy = _create_lns_op_func('cross_entropy',
        torch.nn.functional.cross_entropy, signature=_build_signature([
        ("x", "pk", torch.LongTensor),
        ("target", "pk", torch.LongTensor),
        ("weight", "pk", Optional[torch.LongTensor], None),
        ("reduction", "pk", str, "mean")],
        torch.LongTensor,
    ))



    # ==========================
    #    Activation Functions
    # ==========================

    relu = _create_lns_op_func('relu',
        torch.nn.functional.relu, signature=_build_signature([
        ("x", "pk", torch.LongTensor)],
        torch.LongTensor,
    ))

    leaky_relu = _create_lns_op_func('leaky_relu',
        torch.nn.functional.leaky_relu, signature=_build_signature([
        ("x", "pk", torch.LongTensor),
        ("negative_slope", "pk", torch.LongTensor)],
        torch.LongTensor,
    ))

    threshold = _create_lns_op_func('threshold',
        torch.nn.functional.threshold, signature=_build_signature([
        ("x", "pk", torch.LongTensor),
        ("threshold", "pk", torch.LongTensor),
        ("value", "pk", torch.LongTensor)],
        torch.LongTensor,
    ))

    tanh = _create_lns_op_func('tanh',
        torch.nn.functional.tanh, signature=_build_signature([
        ("x", "pk", torch.LongTensor)],
        torch.LongTensor,
    ))

    sigmoid = _create_lns_op_func('sigmoid',
        torch.nn.functional.sigmoid, signature=_build_signature([
        ("x", "pk", torch.LongTensor)],
        torch.LongTensor,
    ))

    logsigmoid = _create_lns_op_func('logsigmoid',
        torch.nn.functional.logsigmoid, signature=_build_signature([
        ("x", "pk", torch.LongTensor)],
        torch.LongTensor,
    ))

    softmin = _create_lns_op_func('softmin',
        torch.nn.functional.softmin, signature=_build_signature([
        ("x", "pk", torch.LongTensor),
        ("dim", "pk", int, None)],
        torch.LongTensor,
    ))

    softmax = _create_lns_op_func('softmax',
        torch.nn.functional.softmax, signature=_build_signature([
        ("x", "pk", torch.LongTensor),
        ("dim", "pk", int, None)],
        torch.LongTensor,
    ))

    log_softmax = _create_lns_op_func('log_softmax',
        torch.nn.functional.log_softmax, signature=_build_signature([
        ("x", "pk", torch.LongTensor),
        ("dim", "pk", int, None)],
        torch.LongTensor,
    ))

    hardtanh = _create_lns_op_func('hardtanh',
        torch.nn.functional.hardtanh, signature=_build_signature([
        ("x", "pk", torch.LongTensor),
        ("min_val", "pk", torch.LongTensor),
        ("max_val", "pk", torch.LongTensor)],
        torch.LongTensor,
    ))

    hardswish = _create_lns_op_func('hardswish',
        torch.nn.functional.hardswish, signature=_build_signature([
        ("x", "pk", torch.LongTensor)],
        torch.LongTensor,
    ))

    elu = _create_lns_op_func('elu',
        torch.nn.functional.elu, signature=_build_signature([
        ("x", "pk", torch.LongTensor),
        ("alpha", "pk", torch.LongTensor)],
        torch.LongTensor,
    ))

    selu = _create_lns_op_func('selu',
        torch.nn.functional.selu, signature=_build_signature([
        ("x", "pk", torch.LongTensor)],
        torch.LongTensor,
    ))

    celu = _create_lns_op_func('celu',
        torch.nn.functional.celu, signature=_build_signature([
        ("x", "pk", torch.LongTensor),
        ("alpha", "pk", torch.LongTensor)],
        torch.LongTensor,
    ))

    prelu = _create_lns_op_func('prelu',
        torch.nn.functional.prelu, signature=_build_signature([
        ("x", "pk", torch.LongTensor),
        ("a", "pk", torch.LongTensor)],
        torch.LongTensor,
    ))

    rrelu = _create_lns_op_func('rrelu',
        torch.nn.functional.rrelu, signature=_build_signature([
        ("x", "pk", torch.LongTensor),
        ("a", "pk", torch.LongTensor)],
        torch.LongTensor,
    ))

    glu = _create_lns_op_func('glu',
        torch.nn.functional.glu, signature=_build_signature([
        ("x", "pk", torch.LongTensor),
        ("dim", "pk", int, -1)],
        torch.LongTensor,
    ))

    hardshrink = _create_lns_op_func('hardshrink',
        torch.nn.functional.hardshrink, signature=_build_signature([
        ("x", "pk", torch.LongTensor),
        ("lambd", "pk", torch.LongTensor)],
        torch.LongTensor,
    ))

    tanhshrink = _create_lns_op_func('tanhshrink',
        torch.nn.functional.tanhshrink, signature=_build_signature([
        ("x", "pk", torch.LongTensor)],
        torch.LongTensor,
    ))

    softsign = _create_lns_op_func('softsign',
        torch.nn.functional.softsign, signature=_build_signature([
        ("x", "pk", torch.LongTensor)],
        torch.LongTensor,
    ))

    softplus = _create_lns_op_func('softplus',
        torch.nn.functional.softplus, signature=_build_signature([
        ("x", "pk", torch.LongTensor),
        ("beta", "pk", torch.LongTensor),
        ("threshold", "pk", torch.LongTensor)],
        torch.LongTensor,
    ))

    softshrink = _create_lns_op_func('softshrink',
        torch.nn.functional.softshrink, signature=_build_signature([
        ("x", "pk", torch.LongTensor),
        ("lambd", "pk", torch.LongTensor)],
        torch.LongTensor,
    ))

    hardsigmoid = _create_lns_op_func('hardsigmoid',
        torch.nn.functional.hardsigmoid, signature=_build_signature([
        ("x", "pk", torch.LongTensor)],
        torch.LongTensor,
    ))

    silu = _create_lns_op_func('silu',
        torch.nn.functional.silu, signature=_build_signature([
        ("x", "pk", torch.LongTensor)],
        torch.LongTensor,
    ))



    # ======================
    #    Layer Operations
    # ======================

    lns_linear = _create_lns_op_func('linear',
        torch.nn.functional.linear, signature=_build_signature([
        ("x", "pk", torch.LongTensor),
        ("A", "pk", torch.LongTensor),
        ("bias", "pk", Optional[torch.LongTensor], None)],
        torch.LongTensor,
    ))

    lns_bilinear = _create_lns_op_func('bilinear',
        torch.nn.functional.bilinear, signature=_build_signature([
        ("x", "pk", torch.LongTensor),
        ("y", "pk", torch.LongTensor),
        ("A", "pk", torch.LongTensor),
        ("bias", "pk", Optional[torch.LongTensor], None)],
        torch.LongTensor,
    ))

    lns_dropout = _create_lns_op_func('dropout',
        torch.nn.functional.dropout, signature=_build_signature([
        ("x", "pk", torch.LongTensor),
        ("p", "pk", float, 0.5)],
        torch.LongTensor,
    ))

    lns_dropout1d = _create_lns_op_func('dropout1d',
        torch.nn.functional.dropout1d, signature=_build_signature([
        ("x", "pk", torch.LongTensor),
        ("p", "pk", float, 0.5)],
        torch.LongTensor,
    ))

    lns_dropout2d = _create_lns_op_func('dropout2d',
        torch.nn.functional.dropout2d, signature=_build_signature([
        ("x", "pk", torch.LongTensor),
        ("p", "pk", float, 0.5)],
        torch.LongTensor,
    ))

    lns_dropout3d = _create_lns_op_func('dropout3d',
        torch.nn.functional.dropout3d, signature=_build_signature([
        ("x", "pk", torch.LongTensor),
        ("p", "pk", float, 0.5)],
        torch.LongTensor,
    ))

    lns_conv1d = _create_lns_op_func('conv1d',
        torch.nn.functional.conv1d, signature=_build_signature([
        ("x", "pk", torch.LongTensor),
        ("weight", "pk", torch.LongTensor),
        ("bias", "pk", torch.LongTensor),
        ("stride", "pk", int, 1),
        ("padding", "pk", int, 0),
        ("dilation", "pk", int, 1),
        ("groups", "pk", int, 1)],
        torch.LongTensor,
    ))

    lns_conv2d = _create_lns_op_func('conv2d',
        torch.nn.functional.conv2d, signature=_build_signature([
        ("x", "pk", torch.LongTensor),
        ("weight", "pk", torch.LongTensor),
        ("bias", "pk", torch.LongTensor),
        ("stride", "pk", Union[int, Tuple[int]], 1),
        ("padding", "pk", Union[int, Tuple[int]], 0),
        ("dilation", "pk", Union[int, Tuple[int]], 1),
        ("groups", "pk", int, 1)],
        torch.LongTensor,
    ))

    lns_conv3d = _create_lns_op_func('conv3d',
        torch.nn.functional.conv3d, signature=_build_signature([
        ("x", "pk", torch.LongTensor),
        ("weight", "pk", torch.LongTensor),
        ("bias", "pk", torch.LongTensor),
        ("stride", "pk", Union[int, Tuple[int]], 1),
        ("padding", "pk", Union[int, Tuple[int]], 0),
        ("dilation", "pk", Union[int, Tuple[int]], 1),
        ("groups", "pk", int, 1)],
        torch.LongTensor,
    ))

    lns_avg_pool1d = _create_lns_op_func('avg_pool1d',
        torch.nn.functional.avg_pool1d, signature=_build_signature([
        ("x", "pk", torch.LongTensor),
        ("kernel_size", "pk", int),
        ("stride", "pk", Union[int, Tuple[int], None], None),
        ("padding", "pk", Union[int, Tuple[int]], 0),
        ("ceil_mode", "pk", bool, False),
        ("count_include_pad", "pk", bool, True)],
        torch.LongTensor,
    ))

    lns_avg_pool2d = _create_lns_op_func('avg_pool2d',
        torch.nn.functional.avg_pool2d, signature=_build_signature([
        ("x", "pk", torch.LongTensor),
        ("kernel_size", "pk", int),
        ("stride", "pk", Union[int, Tuple[int], None], None),
        ("padding", "pk", Union[int, Tuple[int]], 0),
        ("ceil_mode", "pk", bool, False),
        ("count_include_pad", "pk", bool, True),
        ("divisor_override", "pk", Optional[torch.LongTensor], None)],
        torch.LongTensor,
    ))

    lns_avg_pool3d = _create_lns_op_func('avg_pool3d',
        torch.nn.functional.avg_pool3d, signature=_build_signature([
        ("x", "pk", torch.LongTensor),
        ("kernel_size", "pk", int),
        ("stride", "pk", Union[int, Tuple[int], None], None),
        ("padding", "pk", Union[int, Tuple[int]], 0),
        ("ceil_mode", "pk", bool, False),
        ("count_include_pad", "pk", bool, True),
        ("divisor_override", "pk", Optional[torch.LongTensor], None)],
        torch.LongTensor,
    ))

    lns_adaptive_avg_pool1d = _create_lns_op_func('adaptive_avg_pool1d',
        torch.nn.functional.adaptive_avg_pool1d, signature=_build_signature([
        ("x", "pk", torch.LongTensor),
        ("output_size", "pk", Union[int, Tuple[int]])],
        torch.LongTensor,
    ))

    lns_adaptive_avg_pool2d = _create_lns_op_func('adaptive_avg_pool2d',
        torch.nn.functional.adaptive_avg_pool2d, signature=_build_signature([
        ("x", "pk", torch.LongTensor),
        ("output_size", "pk", Union[int, Tuple[int, int]])],
        torch.LongTensor,
    ))

    lns_adaptive_avg_pool3d = _create_lns_op_func('adaptive_avg_pool3d',
        torch.nn.functional.adaptive_avg_pool3d, signature=_build_signature([
        ("x", "pk", torch.LongTensor),
        ("output_size", "pk", Union[int, Tuple[int, int, int]])],
        torch.LongTensor,
    ))

    lns_batch_norm = _create_lns_op_func('batch_norm',
        torch.nn.functional.batch_norm, signature=_build_signature([
        ("x", "pk", torch.LongTensor),
        ("running_mean", "pk", torch.LongTensor),
        ("running_var", "pk", torch.LongTensor),
        ("momentum", "pk", torch.LongTensor),
        ("eps", "pk", torch.LongTensor),
        ("weight", "pk", Optional[torch.LongTensor], None),
        ("bias", "pk", Optional[torch.LongTensor], None),
        ("training", "pk", bool, False)],
        torch.LongTensor,
    ))

    lns_layer_norm = _create_lns_op_func('layer_norm',
        torch.nn.functional.layer_norm, signature=_build_signature([
        ("x", "pk", torch.LongTensor),
        ("eps", "pk", torch.LongTensor),
        ("normalized_shape", "pk", Union[int, Tuple[int]]),
        ("weight", "pk", Optional[torch.LongTensor], None),
        ("bias", "pk", Optional[torch.LongTensor], None)],
        torch.LongTensor,
    ))

    lns_max_pool1d = _create_lns_op_func('max_pool1d',
        torch.nn.functional.max_pool1d, signature=_build_signature([
        ("x", "pk", torch.LongTensor),
        ("kernel_size", "pk", Union[int, Tuple[int]]),
        ("stride", "pk", Union[int, Tuple[int], None], None),
        ("padding", "pk", Union[int, Tuple[int]], 0),
        ("dilation", "pk", Union[int, Tuple[int]], 1),
        ("ceil_mode", "pk", bool, False),
        ("return_indices", "pk", bool, False)],
        Union[torch.LongTensor, Tuple[torch.LongTensor, torch.Tensor]],
    ))

    lns_max_pool2d = _create_lns_op_func('max_pool2d',
        torch.nn.functional.max_pool2d, signature=_build_signature([
        ("x", "pk", torch.LongTensor),
        ("kernel_size", "pk", Union[int, Tuple[int]]),
        ("stride", "pk", Union[int, Tuple[int], None], None),
        ("padding", "pk", Union[int, Tuple[int]], 0),
        ("dilation", "pk", Union[int, Tuple[int]], 1),
        ("ceil_mode", "pk", bool, False),
        ("return_indices", "pk", bool, False)],
        Union[torch.LongTensor, Tuple[torch.LongTensor, torch.Tensor]],
    ))

    lns_max_pool3d = _create_lns_op_func('max_pool3d',
        torch.nn.functional.max_pool3d, signature=_build_signature([
        ("x", "pk", torch.LongTensor),
        ("kernel_size", "pk", Union[int, Tuple[int]]),
        ("stride", "pk", Union[int, Tuple[int], None], None),
        ("padding", "pk", Union[int, Tuple[int]], 0),
        ("dilation", "pk", Union[int, Tuple[int]], 1),
        ("ceil_mode", "pk", bool, False),
        ("return_indices", "pk", bool, False)],
        Union[torch.LongTensor, Tuple[torch.LongTensor, torch.Tensor]],
    ))
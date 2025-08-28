import xlnstorch
from xlnstorch import set_default_implementation
import torch

from .internal_lns_ops import (
    lns_add,
    lns_sub,
    lns_mul,
    lns_div,
    lns_neg,
    lns_abs,
    lns_sqrt,
    lns_square,
    lns_pow,
    lns_exp,
    lns_log,
    lns_reciprocal,
    lns_sign,
    lns_positive,
    lns_sum,
    lns_prod,
    lns_mean,
    lns_var,
    lns_matmul,
    lns_transpose,

    lns_equal,
    lns_eq,
    lns_ne,
    lns_ge,
    lns_gt,
    lns_le,
    lns_lt,
    lns_isclose,
    lns_allclose,
    lns_any,
    lns_all,
    lns_isin,
    lns_sort,
    lns_argsort,
    lns_kthvalue,
    lns_maximum,
    lns_minimum,
    lns_max,
    lns_argmax,
    lns_min,
    lns_argmin,
    lns_clamp,

    lns_broadcast_to,
    lns_clone,
    lns_squeeze,
    lns_unsqueeze,
    lns_stack,
    lns_cat,
    lns_chunk,
    lns_where,
    lns_pad,

    lns_mse_loss,
    lns_l1_loss,
    lns_binary_cross_entropy,
    lns_binary_cross_entropy_with_logits,
    lns_nll_loss,
    lns_poisson_nll_loss,
    lns_hinge_embedding_loss,
    lns_kl_div,
    lns_margin_ranking_loss,
    lns_gaussian_nll_loss,
    lns_huber_loss,
    lns_smooth_l1_loss,
    lns_cross_entropy,

    lns_relu,
    lns_relu_,
    lns_leaky_relu,
    lns_leaky_relu_,
    lns_threshold,
    lns_threshold_,
    lns_tanh,
    lns_sigmoid,
    lns_logsigmoid,
    lns_softmin,
    lns_softmax,
    lns_log_softmax,
    lns_hardtanh,
    lns_hardswish,
    lns_elu,
    lns_selu,
    lns_celu,
    lns_prelu,
    lns_rrelu,
    lns_glu,
    lns_hardshrink,
    lns_tanhshrink,
    lns_softsign,
    lns_softplus,
    lns_softshrink,
    lns_hardsigmoid,
    lns_silu,

    lns_linear,
    lns_bilinear,
    lns_dropout,
    lns_dropout1d,
    lns_dropout2d,
    lns_dropout3d,
    lns_conv1d,
    lns_conv2d,
    lns_conv3d,
    lns_avg_pool1d,
    lns_avg_pool2d,
    lns_avg_pool3d,
    lns_adaptive_avg_pool1d,
    lns_adaptive_avg_pool2d,
    lns_adaptive_avg_pool3d,
    lns_batch_norm,
    lns_layer_norm,
    lns_max_pool1d,
    lns_max_pool2d,
    lns_max_pool3d,
)

def lns_sum_to_size(tensor: torch.Tensor, base: torch.Tensor, target_size: torch.Size) -> torch.Tensor:
    """
    Sum-reduce a tensor to a target size by summing over excess dimensions.

    Parameters
    ----------
    tensor : LNSTensor
        The input LNSTensor to be reduced.
    target_size : torch.Size
        The desired target size after reduction.

    Returns
    -------
    LNSTensor
        The reduced LNSTensor with the specified target size.

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
        tensor = lns_sum(tensor, base, dim=tuple(range(leading)), keepdim=False)
        tensor_shape = tensor_shape[leading:]

    # reduce dimensions where target size is 1 but tensor has a larger size
    reduce_dims = [i for i, (ts, gs) in enumerate(zip(tensor_shape, tgt_shape)) if gs == 1 and ts != 1]
    if reduce_dims:
        tensor = lns_sum(tensor, base, dim=tuple(reduce_dims), keepdim=True)

    return tensor.reshape(target_size)

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

    "implement_sbdb",
    "set_default_sbdb_implementation",
    "override_sbdb_implementation",
    "register_xlnsconf_implementation",
    "sbdb",

    #  internal LNS operation functions
    "lns_add",
    "lns_sub",
    "lns_mul",
    "lns_div",
    "lns_neg",
    "lns_abs",
    "lns_sqrt",
    "lns_square",
    "lns_pow",
    "lns_exp",
    "lns_log",
    "lns_reciprocal",
    "lns_sign",
    "lns_positive",
    "lns_sum",
    "lns_prod",
    "lns_mean",
    "lns_var",
    "lns_matmul",
    "lns_transpose",

    "lns_equal",
    "lns_eq",
    "lns_ne",
    "lns_ge",
    "lns_gt",
    "lns_le",
    "lns_lt",
    "lns_isclose",
    "lns_allclose",
    "lns_any",
    "lns_all",
    "lns_isin",
    "lns_sort",
    "lns_argsort",
    "lns_kthvalue",
    "lns_maximum",
    "lns_minimum",
    "lns_max",
    "lns_argmax",
    "lns_min",
    "lns_argmin",
    "lns_clamp",

    "lns_broadcast_to",
    "lns_clone",
    "lns_squeeze",
    "lns_unsqueeze",
    "lns_stack",
    "lns_cat",
    "lns_chunk",
    "lns_where",
    "lns_pad",

    "lns_mse_loss",
    "lns_l1_loss",
    "lns_binary_cross_entropy",
    "lns_binary_cross_entropy_with_logits",
    "lns_nll_loss",
    "lns_poisson_nll_loss",
    "lns_hinge_embedding_loss",
    "lns_kl_div",
    "lns_margin_ranking_loss",
    "lns_gaussian_nll_loss",
    "lns_huber_loss",
    "lns_smooth_l1_loss",
    "lns_cross_entropy",

    "lns_relu",
    "lns_relu_",
    "lns_leaky_relu",
    "lns_leaky_relu_",
    "lns_threshold",
    "lns_threshold_",
    "lns_tanh",
    "lns_sigmoid",
    "lns_logsigmoid",
    "lns_softmin",
    "lns_softmax",
    "lns_log_softmax",
    "lns_hardtanh",
    "lns_hardswish",
    "lns_elu",
    "lns_selu",
    "lns_celu",
    "lns_prelu",
    "lns_rrelu",
    "lns_glu",
    "lns_hardshrink",
    "lns_tanhshrink",
    "lns_softsign",
    "lns_softplus",
    "lns_softshrink",
    "lns_hardsigmoid",
    "lns_silu",

    "lns_linear",
    "lns_bilinear",
    "lns_dropout",
    "lns_dropout1d",
    "lns_dropout2d",
    "lns_dropout3d",
    "lns_conv1d",
    "lns_conv2d",
    "lns_conv3d",
    "lns_avg_pool1d",
    "lns_avg_pool2d",
    "lns_avg_pool3d",
    "lns_adaptive_avg_pool1d",
    "lns_adaptive_avg_pool2d",
    "lns_adaptive_avg_pool3d",
    "lns_batch_norm",
    "lns_layer_norm",
    "lns_max_pool1d",
    "lns_max_pool2d",
    "lns_max_pool3d",
]
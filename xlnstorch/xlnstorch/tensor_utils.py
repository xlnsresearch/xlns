"""
Utility functions for LNSTensor operations and autograd functions.
"""
from __future__ import annotations
from typing import Any, Tuple, List, Sequence, TYPE_CHECKING, Optional, Union
from contextlib import nullcontext
import math
import re
import torch
import xlns as xl

# Import constants and base classes that don't cause circular imports
from xlnstorch import LNS_ZERO, CSRC_AVAILABLE
from xlnstorch.autograd import LNSFunction
import xlnstorch.csrc

# Precomputed table of bases from precisions
# base = 2^(2^(-f)) for f in [1, 40]
# f=32 gives base ≈ 1.0000000023283064365, which is very close to 1
# Going beyond f=32 risks numerical precision issues

# Create tensor of precision values f from 1 to 40
PRECISION_VALUES = torch.arange(1, 41, dtype=torch.float64)
PRECISION_BASES = torch.pow(2.0, torch.pow(2.0, -PRECISION_VALUES))

OVF_MAX = None
OVF_MIN = None

def get_base_from_precision(f: int) -> torch.Tensor:
    """
    Get the logarithmic base for a given precision.

    Parameters
    ----------
    f : int
        The precision (number of fractional exponent bits).
        Must be in range [1, 40].

    Returns
    -------
    torch.Tensor
        The corresponding logarithmic base (2^(2^(-f))).

    Raises
    ------
    ValueError
        If precision f is outside the supported range [1, 40].
    """
    if f < 1 or f > 40:
        raise ValueError(f"Precision f={f} not supported. Must be in range [1, 40].")
    return PRECISION_BASES[f - 1]

def get_precision_from_base(base: torch.Tensor, tolerance: Union[float, torch.Tensor] = 0) -> Optional[int]:
    """
    Get the precision for a given logarithmic base, if it matches a precomputed base.

    Parameters
    ----------
    base : torch.Tensor
        The logarithmic base to check.
    tolerance : float, torch.Tensor, optional
        Tolerance for floating-point comparison. Default is 0 as bases should
        match exactly since we use precomputed values.

    Returns
    -------
    int or None
        The corresponding precision if the base matches a precomputed value,
        otherwise None.
    """
    # Check if base matches any precomputed base within tolerance
    differences = torch.abs(PRECISION_BASES - base)
    matches = differences <= tolerance
    if matches.any():
        # Return the first match (precision = index + 1)
        return matches.nonzero()[0].item() + 1
    return None

# Use TYPE_CHECKING for type hints only
if TYPE_CHECKING:
    from .tensor import LNSTensor

# Lazy import cache to avoid repeated imports
_tensor_module = None
_operator_module = None

def _get_tensor_module():
    """Lazy import of tensor module to avoid circular imports."""
    global _tensor_module
    if _tensor_module is None:
        from . import tensor
        _tensor_module = tensor
    return _tensor_module

def _get_operator_module():
    global _operator_module
    if _operator_module is None:
        from . import operators
        _operator_module = operators
    return _operator_module

float_to_lns_forward = None
float_to_lns_backward = None
change_base_forward = None
change_base_backward = None

def _float_to_lns_forward_python(x: torch.Tensor, base: torch.Tensor) -> torch.Tensor:

    log_base = torch.log(base)
    log_data = torch.log(torch.abs(x)) / log_base
    exponent = log_data.round().to(torch.int64)

    sign_bit = (x < 0).to(torch.int64)
    packed_int = (exponent << 1) | sign_bit
    packed = packed_int.to(torch.float64)
    packed = torch.where(torch.eq(x, 0), LNS_ZERO, packed)

    return packed

def _float_to_lns_backward_python(grad_output: torch.Tensor, base: torch.Tensor) -> torch.Tensor:
    packed_grad_output = grad_output.to(torch.int64)

    exponent = (packed_grad_output >> 1).to(torch.float64)
    sign = torch.where((packed_grad_output & 1).bool(), -1.0, 1.0)

    return torch.where(torch.eq(packed_grad_output | 1, LNS_ZERO), 0.0, sign * torch.pow(base, exponent))

def _change_base_forward_python(x: torch.Tensor, old_base: torch.Tensor, new_base: torch.Tensor) -> torch.Tensor:
    packed_int = x.to(torch.int64)
    sign_bit = packed_int & 1
    exponent = (packed_int >> 1).to(torch.float64)

    exponent_new = exponent * torch.log(old_base) / torch.log(new_base)
    new_packed_int = (exponent_new.round().to(torch.int64) << 1) | sign_bit
    new_tensor = new_packed_int.to(torch.float64)

    return new_tensor

def _change_base_backward_python(grad_output: torch.Tensor, old_base: torch.Tensor, new_base: torch.Tensor) -> torch.Tensor:
    packed_int = grad_output.to(torch.int64)
    sign_bit = packed_int & 1
    exponent = (packed_int >> 1).to(torch.float64)

    exponent_new = exponent * torch.log(new_base) / torch.log(old_base)
    old_packed_int = (exponent_new.round().to(torch.int64) << 1) | sign_bit
    old_tensor = old_packed_int.to(torch.float64)

    return old_tensor

def toggle_cpp_tensor_utils(use_cpp: bool) -> None:
    """
    Toggle the use of C++ implementations for tensor utility functions. This
    function is called by `xlnstorch.operators.toggle_cpp_implementations()`.

    In particular, this toggles the implementations for float to and from LNS
    conversions and base change operations.

    Parameters
    ----------
    use_cpp : bool
        If True, use C++ implementations where available. If False, use
        pure Python implementations.

    Raises
    ------
    RuntimeError
        If C++ extensions are not available and `use_cpp` is True.
    """
    global float_to_lns_forward, float_to_lns_backward
    global change_base_forward, change_base_backward

    if use_cpp and not xlnstorch.CSRC_AVAILABLE:
        raise RuntimeError("C++ extensions are not available. Cannot enable C++ tensor utils.")

    if use_cpp:
        float_to_lns_forward = xlnstorch.csrc.float_to_lns_forward
        float_to_lns_backward = xlnstorch.csrc.float_to_lns_backward
        change_base_forward = xlnstorch.csrc.change_base_forward
        change_base_backward = xlnstorch.csrc.change_base_backward

    else:
        float_to_lns_forward = _float_to_lns_forward_python
        float_to_lns_backward = _float_to_lns_backward_python
        change_base_forward = _change_base_forward_python
        change_base_backward = _change_base_backward_python

class FloatToLNS(LNSFunction):

    @staticmethod
    def forward(x, base):
        return float_to_lns_forward(x, base)

    @staticmethod
    def setup_context(ctx, inputs, output):
        _, base = inputs
        ctx.save_for_backward(base)

    @staticmethod
    def backward(ctx, grad_output):
        base, = ctx.saved_tensors
        return float_to_lns_backward(grad_output, base), None


class LNSChangeBaseFunction(LNSFunction):

    @staticmethod
    def forward(tensor, old_base, new_base):
        return change_base_forward(tensor, old_base, new_base)

    @staticmethod
    def setup_context(ctx, inputs, outputs):
        _, old_base, new_base = inputs
        ctx.save_for_backward(old_base, new_base)

    @staticmethod
    def backward(ctx, grad_output):
        old_base, new_base = ctx.saved_tensors
        return change_base_backward(grad_output, old_base, new_base), None, None


class LNSGetItemFunction(LNSFunction):

    @staticmethod
    def forward(x, idx):
        return x[idx]

    @staticmethod
    def setup_context(ctx, inputs, output):
        x, idx = inputs
        ctx.is_idx_tensor = torch.is_tensor(idx)

        if ctx.is_idx_tensor:
            ctx.save_for_backward(x, idx)
        else:
            ctx.save_for_backward(x)
            ctx.idx = idx

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.is_idx_tensor:
            x, idx = ctx.saved_tensors
        else:
            x, = ctx.saved_tensors
            idx = ctx.idx

        grad_x = torch.full_like(x, LNS_ZERO)
        grad_x[idx] = grad_output

        return grad_x, None


class LNSToFunction(LNSFunction):

    @staticmethod
    def forward(x, device):
        return x.to(device)

    @staticmethod
    def setup_context(ctx, inputs, output):
        x, _ = inputs
        ctx.orig_device = x.device

    @staticmethod
    def backward(ctx, grad_output):
        grad_x = grad_output.to(ctx.orig_device)
        return grad_x, None


class LNSViewFunction(LNSFunction):

    @staticmethod
    def forward(x, shape):
        return x.view(*shape)

    @staticmethod
    def setup_context(ctx, inputs, output):
        x, shape = inputs
        ctx.original_shape = x.shape
        ctx.n_shape = len(shape)

    @staticmethod
    def backward(ctx, grad_output):
        grad_x = grad_output.contiguous().view(ctx.original_shape)

        return grad_x, None


class LNSContiguousFunction(LNSFunction):

    @staticmethod
    def forward(x, memory_format):
        return x.contiguous(memory_format=memory_format)

    @staticmethod
    def setup_context(ctx, inputs, output):
        pass

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


class LNSRepeatFunction(LNSFunction):

    @staticmethod
    def forward(x, base, repeats):
        return x.repeat(*repeats)

    @staticmethod
    def setup_context(ctx, inputs, output):
        x, base, repeats = inputs
        ctx.save_for_backward(base)
        ctx.input_shape = tuple(x.shape)
        ctx.repeats = tuple(repeats)

    @staticmethod
    def backward(ctx, grad_output):
        ops = _get_operator_module()
        base, = ctx.saved_tensors
        grad_x = grad_output

        for dim, rep in enumerate(ctx.repeats):
            if rep == 1:
                continue

            new_shape = list(grad_x.shape)
            new_shape[dim] = ctx.input_shape[dim]
            new_shape.insert(dim + 1, rep)

            grad_x = ops.lns_sum(grad_x.view(*new_shape), base, dim=dim+1)

        return grad_x, None, None


class LNSOverflowFunction(LNSFunction):

    @staticmethod
    def forward(x, base, max=None, min=None):
        # import here to avoid circular imports
        tensor_module = _get_tensor_module()
        ops = _get_operator_module()

        result = x.to(torch.int64)

        if max is not None:
            max_packed = tensor_module.LNSTensor.get_internal_tensor(max, base)
            result = torch.where(ops.lns_gt(ops.lns_abs(result), max_packed),
                                 ops.lns_mul(ops.lns_sign(result, base), max_packed, base), result)

        if min is not None:
            min_packed = tensor_module.LNSTensor.get_internal_tensor(min, base)
            result = torch.where(ops.lns_lt(ops.lns_abs(result), min_packed),
                                 LNS_ZERO, result)

        return result.to(torch.float64)

    @staticmethod
    def setup_context(ctx, inputs, output):
        x, base, max, min = inputs
        ctx.save_for_backward(x, base)
        ctx.max = max
        ctx.min = min

    @staticmethod
    def backward(ctx, grad_output):
        x, base = ctx.saved_tensors
        tensor_module = _get_tensor_module()
        ops = _get_operator_module()

        x_packed, grad_x = x.to(torch.int64), grad_output.to(torch.int64)

        if ctx.max is not None:
            max_packed = tensor_module.LNSTensor.get_internal_tensor(ctx.max, base)
            grad_x = torch.where(ops.lns_gt(ops.lns_abs(x_packed), max_packed), LNS_ZERO, grad_x)

        if ctx.min is not None:
            min_packed = tensor_module.LNSTensor.get_internal_tensor(ctx.min, base)
            grad_x = torch.where(ops.lns_lt(ops.lns_abs(x_packed), min_packed), LNS_ZERO, grad_x)

        return grad_x, None, None, None

def set_overflow_limits(max: float = None, min: float = None):
    """
    Set global overflow limits for the handle_overflow function.

    Parameters
    ----------
    max : float, optional
        Maximum representable value in LNS. Values exceeding this will be
        clamped to this maximum. If None, no maximum limit is set.
    min : float, optional
        Minimum representable value in LNS. Values below this will be
        clamped to zero. If None, no minimum limit is set.
    """
    global OVF_MAX, OVF_MIN

    if max is not None:
        OVF_MAX = max

    if min is not None:
        OVF_MIN = min

def handle_overflow(*tensors: LNSTensor, inplace=False, no_grad=False) -> List[LNSTensor]:
    """
    Handle overflow for a variable number of LNSTensors based on global overflow limits.

    Parameters
    ----------
    tensors : LNSTensor
        Variable number of LNSTensor objects to check for overflow.
    inplace : bool, optional
        If True, modifies the input tensors in place. Default is False.
    no_grad : bool, optional
        If True, disables gradient tracking during overflow handling. Default is False.
    """
    tensor_module = _get_tensor_module()
    handled_tensors = []

    with torch.no_grad() if no_grad else nullcontext():

        for tensor in tensors:
            handled_tensors.append(LNSOverflowFunction.apply(tensor, tensor.base, OVF_MAX, OVF_MIN))

    if inplace:

        for i in range(len(tensors)):
            if no_grad:
                tensors[i]._lns.data.copy_(handled_tensors[i].data)
            else:
                tensors[i]._inplace_copy(handled_tensors[i])

        if len(tensors) == 1:
            return tensors[0]
        return tensors

    for i in range(len(tensors)):
        handled_tensors[i] = tensor_module.lnstensor(handled_tensors[i], from_lns=True, b=tensors[i].base)

    if len(handled_tensors) == 1:
        return handled_tensors[0]
    return handled_tensors


def align_lnstensor_bases(
        *tensors: LNSTensor,
        base: Optional[torch.Tensor] = None
    ) -> Tuple[LNSTensor, ...]:
    """
    Aligns the bases of a sequence of LNSTensors to a common base.

    Parameters
    ----------
    tensors : LNSTensor
        Variable number of LNSTensor objects to be aligned.
    base : torch.Tensor, optional
        The target base to which all tensors should be aligned.
        If None, the default base from `xl.xlnsB` will be used.

    Returns
    -------
    Tuple[LNSTensor, ...]
        A tuple containing the LNSTensors with their bases aligned
        to the specified base or default base. Tensors that already
        match the base will be returned unchanged.

    Notes
    -----
    This function ensures compatibility for operations requiring a
    common logarithmic base. This operation is tracked by PyTorch's
    autograd system to allow for correct gradient computation on the
    original tensors in their original bases.
    """
    tensor_module = _get_tensor_module()
    
    if base is None:
        new_base = torch.tensor(xl.xlnsB, dtype=torch.float64)
    else:
        new_base = base.detach()

    aligned_tensors = []
    for tensor in tensors:

        if tensor is None:
            aligned_tensors.append(None)
        elif torch.eq(tensor.base, new_base):
            aligned_tensors.append(tensor)
        else:
            aligned_tensor = LNSChangeBaseFunction.apply(tensor, tensor.base, new_base)
            aligned_tensors.append(tensor_module.lnstensor(aligned_tensor, from_lns=True, b=new_base))

    return tuple(aligned_tensors)


def format_lnstensor_operands(*operands: Any, base=None) -> Tuple[LNSTensor, ...]:
    """
    Converts a variable number of operands to LNSTensor objects, aligning
    all operands to the base of the first operand that is an LNSTensor.

    Parameters
    ----------
    operands : Any
        Variable number of operands, which can be LNSTensor objects or
        other array-like objects that can be converted to LNSTensor.
    base : torch.Tensor, optional
        The base to force conversion to if supplied.

    Returns
    -------
    Tuple[LNSTensor, ...]
        A tuple of LNSTensor objects with their bases aligned to the base
        of the first LNSTensor operand. If no LNSTensor is found, all
        operands are converted to LNSTensors with the default base.
    """
    tensor_module = _get_tensor_module()
    
    if base is None:
        for operand in operands:
            if isinstance(operand, tensor_module.LNSTensor):
                base = operand.base
                break
        else:
            base = torch.tensor(xl.xlnsB, dtype=torch.float64)

    elif torch.is_tensor(base):
        base = base.detach()

    else:
        base = torch.tensor(base, dtype=torch.float64)

    converted_operands = []
    for operand in operands:
        if isinstance(operand, tensor_module.LNSTensor):
            converted_operands.append(operand)
        elif operand is None:
            converted_operands.append(None)
        else:
            converted_operands.append(tensor_module.lnstensor(operand, detach=False, b=base))

    return align_lnstensor_bases(*converted_operands, base=base)

def get_internal_lnstensor_operands(*operands: Any, base=None) -> Tuple[torch.Tensor, ...]:
    """
    Converts a variable number of operands to LNSTensor objects, aligns
    their bases, and extracts their internal packed tensor representations.

    Parameters
    ----------
    operands : Any
        Variable number of operands, which can be LNSTensor objects or
        other array-like objects that can be converted to LNSTensor.
    base : torch.Tensor, optional
        The base to force conversion to if supplied.

    Returns
    -------
    Tuple[torch.Tensor, ...]
        A tuple of Tensor objects, representing the internal representations
        of LNSTensors with their bases aligned to the base of the first
        LNSTensor operand. If no LNSTensor is found, all operands are converted
        to LNSTensors with the default base.
    """
    lnstensor_operands = format_lnstensor_operands(*operands, base=base)
    return tuple(op._lns for op in lnstensor_operands)

def make_index_tensors(
        index: Any,
        shape: Union[torch.Size, Sequence[int]]
    ) -> Tuple[torch.Tensor, ...]:

    flat_count = math.prod(shape)
    labels = torch.arange(flat_count).reshape(shape)

    selected = labels[index]

    flat_idx = selected.reshape(-1)
    coord_tuples = torch.unravel_index(flat_idx, shape)

    return coord_tuples

def _format_scientific(log10_val: torch.Tensor):
    """
    Format a logarithmic value in scientific notation.

    Parameters
    ----------
    log10_val : torch.Tensor
        The logarithmic value to format.
    precision : int, optional
        The number of decimal places for the mantissa.
        Default is 2.

    Returns
    -------
    str
        The formatted scientific notation string.
    """
    exponent = torch.floor(log10_val).to(torch.int).item()
    mantissa = torch.pow(10, log10_val % 1).item()

    precision = torch._tensor_str.PRINT_OPTS.precision
    if exponent >= 0:
        return f"{mantissa:.{precision}f}e+{int(exponent)}"

    return f"{mantissa:.{precision}f}e{int(exponent)}"

def _lns_tensor_str(tensor: LNSTensor, indent: int) -> str:
    """
    Custom string representation for LNSTensor that handles large values
    and scientific notation for overflow cases.

    Parameters
    ----------
    tensor : LNSTensor
        The LNSTensor to convert to string.
    indent : int
        The number of spaces to indent the value string.
    overflow_precision : int, optional
        The precision for scientific notation when values overflow.
        Default is 2.

    Returns
    -------
    str
        The string representation of the LNSTensor value, with special handling
        for large values and scientific notation for overflow cases."""

    # store at the start as each call will recompute it
    value = tensor.value

    inf_mask = torch.isinf(value)
    if not torch.any(inf_mask):
        return torch._tensor_str._tensor_str(value, indent)

    # instead of overflowing to inf, we can calculate the mantissa
    # and exponent for scientific notation
    log10_scale = math.log(10) / torch.log(tensor.base)
    log10_values = (tensor._lns.to(torch.int64) >> 1) / log10_scale

    if value.dim() == 0:
        if torch.isinf(value):
            sci_notation = _format_scientific(log10_values)
            result_str = torch._tensor_str._tensor_str(value, indent)
            result_str = result_str.replace('inf', sci_notation)
            result_str = result_str.replace('-inf', '-' + sci_notation)

    else:
        required_prec = 0
        flat_value = value.flatten()
        flat_log10 = log10_values.flatten()

        # calculate required precision for scientific notation
        # to line up correctly with the original values
        for i in range(flat_value.numel()):
            if torch.isinf(flat_value[i]):
                sci_str = _format_scientific(flat_log10[i])
                required_prec = max(required_prec, len(sci_str) - 2)
            else:
                num_str = torch._tensor_str._tensor_str(flat_value[i], 0)
                decimal_index = num_str.find('.')
                required_prec = max(required_prec, len(num_str) - decimal_index)

        # generate the string representation
        original_prec = torch._tensor_str.PRINT_OPTS.precision
        torch.set_printoptions(precision=required_prec)
        result_str = torch._tensor_str._tensor_str(value, indent)
        torch.set_printoptions(precision=original_prec)

        # only look for spaces not all whitespace
        inf_pattern = r'( *)(-?)inf'

        def replacer(match):
            nonlocal inf_counter
            is_negative = match.group(2) == '-'

            while inf_counter < len(flat_value):
                if torch.isinf(flat_value[inf_counter]):
                    if (is_negative and flat_value[inf_counter] < 0) or \
                    (not is_negative and flat_value[inf_counter] > 0):

                        sci_notation = _format_scientific(flat_log10[inf_counter])
                        inf_counter += 1
                        return (('-' if is_negative else '') + sci_notation).rjust(len(match.group(0)))

                inf_counter += 1

            # fallback incase there were no inf values
            return match.group(0)

        inf_counter = 0
        result_str = re.sub(inf_pattern, replacer, result_str)

    return result_str
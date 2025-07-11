"""
Utility functions for LNSTensor operations and autograd functions.
"""
from __future__ import annotations
from typing import Any, Tuple, TYPE_CHECKING
import torch
import xlns as xl

# Import constants and base classes that don't cause circular imports
from . import LNS_ZERO
from .autograd import LNSFunction

# Precomputed table of bases from precisions
# base = 2^(2^(-f)) for f in [1, 40]
# f=32 gives base ≈ 1.0000000023283064365, which is very close to 1
# Going beyond f=32 risks numerical precision issues

# Create tensor of precision values f from 1 to 40
PRECISION_VALUES = torch.arange(1, 41, dtype=torch.float64)
PRECISION_BASES = torch.pow(2.0, torch.pow(2.0, -PRECISION_VALUES))

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

def get_precision_from_base(base: torch.Tensor, tolerance: float | torch.Tensor = 0) -> int | None:
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

def _get_tensor_module():
    """Lazy import of tensor module to avoid circular imports."""
    global _tensor_module
    if _tensor_module is None:
        from . import tensor
        _tensor_module = tensor
    return _tensor_module


class FloatToLNS(LNSFunction):

    @staticmethod
    def forward(x, base):
        log_base = torch.log(base)
        log_data = torch.log(torch.abs(x)) / log_base
        exponent = log_data.round().to(torch.int64)

        sign_bit = (x < 0).to(torch.int64)
        packed_int = (exponent << 1) | sign_bit
        packed = packed_int.to(torch.float64)
        packed = torch.where(torch.eq(x, 0), LNS_ZERO, packed)

        return packed
    
    @staticmethod
    def setup_context(ctx, inputs, output):
        _, base = inputs
        ctx.save_for_backward(base)

    @staticmethod
    def backward(ctx, grad_output):
        base, = ctx.saved_tensors
        packed_grad_output = grad_output.to(torch.int64)

        exponent = (packed_grad_output >> 1).to(torch.float64)
        sign = torch.where((packed_grad_output & 1).bool(), -1.0, 1.0)

        return torch.where(torch.eq(packed_grad_output | 1, LNS_ZERO), 0.0, sign * torch.pow(base, exponent)), None


class LNSChangeBaseFunction(LNSFunction):

    @staticmethod
    def forward(tensor, old_base, new_base):
        packed_int = tensor.to(torch.int64)
        sign_bit = packed_int & 1
        exponent = (packed_int >> 1).to(torch.float64)

        exponent_new = exponent * torch.log(old_base) / torch.log(new_base)
        new_packed_int = (exponent_new.round().to(torch.int64) << 1) | sign_bit
        new_tensor = new_packed_int.to(torch.float64)

        return new_tensor

    @staticmethod
    def setup_context(ctx, inputs, outputs):
        _, old_base, new_base = inputs
        ctx.save_for_backward(old_base, new_base)

    @staticmethod
    def backward(ctx, grad_output):
        old_base, new_base = ctx.saved_tensors

        packed_int = grad_output.to(torch.int64)
        sign_bit = packed_int & 1
        exponent = (packed_int >> 1).to(torch.float64)

        exponent_new = exponent * torch.log(new_base) / torch.log(old_base)
        new_packed_int = (exponent_new.round().to(torch.int64) << 1) | sign_bit
        new_tensor = new_packed_int.to(torch.float64)

        return new_tensor, None, None


def align_lnstensor_bases(
        *tensors: LNSTensor,
        base: torch.Tensor | None = None
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

        if torch.eq(tensor.base, new_base):
            aligned_tensors.append(tensor)
        else:
            aligned_tensor = LNSChangeBaseFunction.apply(tensor, tensor.base, new_base)
            aligned_tensors.append(tensor_module.lnstensor(aligned_tensor, from_lns=True, b=new_base))

    return tuple(aligned_tensors)


def format_lnstensor_operands(*operands: Any) -> Tuple[LNSTensor, ...]:
    """
    Converts a variable number of operands to LNSTensor objects, aligning
    all operands to the base of the first operand that is an LNSTensor.

    Parameters
    ----------
    operands : Any
        Variable number of operands, which can be LNSTensor objects or
        other array-like objects that can be converted to LNSTensor.

    Returns
    -------
    Tuple[LNSTensor, ...]
        A tuple of LNSTensor objects with their bases aligned to the base
        of the first LNSTensor operand. If no LNSTensor is found, all
        operands are converted to LNSTensors with the default base.
    """
    tensor_module = _get_tensor_module()
    
    base = None

    for operand in operands:
        if isinstance(operand, tensor_module.LNSTensor):
            base = operand.base
            break
    else:
        base = torch.tensor(xl.xlnsB, dtype=torch.float64)

    converted_operands = []
    for operand in operands:
        if isinstance(operand, tensor_module.LNSTensor):
            converted_operands.append(operand)
        else:
            converted_operands.append(tensor_module.lnstensor(operand, detach=False, b=base))

    return align_lnstensor_bases(*converted_operands, base=base)

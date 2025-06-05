from typing import Tuple, Any

import torch
import xlns as xl
from .tensor import LNSTensor, lnstensor

class LNSChangeBaseFunction(torch.autograd.Function):

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
    if base is None:
        new_base = torch.tensor(xl.xlnsB, dtype=torch.float64)
    else:
        new_base = base.detach()

    aligned_tensors = []
    for tensor in tensors:

        if torch.eq(tensor.base, new_base):
            aligned_tensors.append(tensor)
        else:
            aligned_tensor = LNSChangeBaseFunction.apply(tensor._lns, tensor.base, new_base)
            aligned_tensors.append(lnstensor(aligned_tensor, from_lns=True, b=new_base))

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
    base = None

    for operand in operands:
        if isinstance(operand, LNSTensor):
            base = operand.base
            break
    else:
        base = torch.tensor(xl.xlnsB, dtype=torch.float64)

    converted_operands = []
    for operand in operands:
        if isinstance(operand, LNSTensor):
            converted_operands.append(operand)
        else:
            converted_operands.append(lnstensor(operand, b=base))

    return align_lnstensor_bases(*converted_operands, base=base)
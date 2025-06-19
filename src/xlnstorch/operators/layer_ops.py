import torch
from .. import LNS_ZERO, LNSTensor, lnstensor, format_lnstensor_operands, implements, zeros_like
from . import (
    lns_mul
)

class LNSDropoutFunction(torch.autograd.Function):

    @staticmethod
    def forward(x, base, p=0.5):
        x_packed = x.to(torch.int64)

        mask = LNSTensor.get_internal_tensor(torch.bernoulli(torch.full_like(x, 1 - p)), base)
        result = lns_mul(x_packed, mask)

        return result.to(torch.float64)

    @staticmethod
    def setup_context(ctx, inputs, output):
        _, base, p = inputs
        ctx.save_for_backward(output, base)
        ctx.p = p

    @staticmethod
    def backward(ctx, grad_output):
        output, base = ctx.saved_tensors

        grad_x = torch.where(output == LNS_ZERO, LNS_ZERO, LNSTensor.get_internal_tensor(1 / (1 - ctx.p), base))
        grad_x = lns_mul(grad_output, grad_x)

        return grad_x, None, None

@implements(torch.nn.functional.dropout, LNSDropoutFunction.forward, "default", default=True)
def dropout(x, p=0.5, training=True, inplace=False):

    if not training or p == 0.0:
        return x

    if p < 0.0 or p >= 1.0:
        raise ValueError(f"Dropout probability p must be in the range [0, 1), but got {p}.")

    result = LNSDropoutFunction.apply(x._lns, x.base, p)

    if inplace:
        x._lns = result
        return x

    return lnstensor(result, from_lns=True, b=x.base)
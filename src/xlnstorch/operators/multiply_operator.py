import torch
import xlns as xl
from ..tensor import apply_lns_op, implements, lnstensor
from ..base import align_lnstensor_bases

class LNSMulFunction(torch.autograd.Function):
    """
    Multiplication becomes addition in the logarithmic domain.

    Gradients are computed as follows:
    d/dx(x * y) = y
    d/dy(x * y) = x
    """

    @staticmethod
    def forward(x, y):
        return x + y

    @staticmethod
    def setup_context(ctx, inputs, outputs):
        x, y = inputs
        ctx.save_for_backward(x, y)

    @staticmethod
    def backward(ctx, grad_output):
        x, y = ctx.saved_tensors

        grad_x = apply_lns_op(torch.mul, grad_output, y)
        grad_y = apply_lns_op(torch.mul, grad_output, x)

        return grad_x, grad_y

@implements(torch.mul, LNSMulFunction.forward, key='default', default=True)
def mul(x, y, *, out=None):

    x, y = align_lnstensor_bases(x, y, base=x.base)
    result = LNSMulFunction.apply(x._lns, y._lns)

    if out is not None:
        out._lns = result

    return lnstensor(result, from_lns=True, b=x.base)
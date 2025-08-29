import torch
from xlnstorch import LNS_ZERO, LNS_ONE, LNS_NEG_ONE, lnstensor, implements
from xlnstorch.autograd import LNSFunction
from . import lns_neg

def _neg(x):
    return x ^ 1

class LNSNegFunction(LNSFunction):
    """
    Negation becomes flipping the sign bit.

    Gradients are computed as follows:
    d/dx(-x) = -1
    """

    @staticmethod
    def forward(x):
        x = x.view(torch.int64)
        result = _neg(x)
        return result.view(torch.float64)

    @staticmethod
    def setup_context(ctx, inputs, output):
        pass # no context needed for this operation

    @staticmethod
    def backward(ctx, grad_output):
        return lns_neg(grad_output)

@implements(torch.neg, _neg, key="default", default=True)
def neg(x, *, out=None):

    result = LNSNegFunction.apply(x)

    if out is not None:
        return out._inplace_copy(result)

    return lnstensor(result, from_lns=True, b=x.base)

def _abs(x):
    abs_x = x & (~1)
    return torch.where(torch.eq(x | 1, LNS_ZERO), LNS_ZERO, abs_x)

class LNSAbsFunction(LNSFunction):
    """
    Absolute value becomes setting the sign bit off.

    Gradients are computed as follows:
    d/dx(|x|) = 1 if x > 0, -1 if x < 0 

    Note that PyTorch defines the gradient to be 0
    when x=0 despite it being undefined here.
    """

    @staticmethod
    def forward(x):
        x = x.view(torch.int64)
        result = _abs(x)
        return result.view(torch.float64)

    @staticmethod
    def setup_context(ctx, inputs, output):
        x, = inputs
        ctx.save_for_backward(x)

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        x = x.view(torch.int64)

        return torch.where(torch.eq(x & 1, 1), lns_neg(grad_output), grad_output)

@implements(torch.abs, _abs, "default", default=True)
def abs(x, *, out=None):

    result = LNSAbsFunction.apply(x)

    if out is not None:
        return out._inplace_copy(result)

    return lnstensor(result, from_lns=True, b=x.base)

class LNSPositiveFunction(LNSFunction):
    """
    This is implemented solely for completeness, this
    operation returns the input.

    Gradients are calculated as follows:
    d/dx(x) = 1
    """

    @staticmethod
    def forward(x):
        return x

    @staticmethod
    def setup_context(ctx, inputs, output):
        pass # no context needed for this operation

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

@implements(torch.positive, LNSPositiveFunction.forward, "default", default=True)
def positive(x):

    result = LNSPositiveFunction.apply(x)
    return lnstensor(result, from_lns=True, b=x.base)

def _sign(x):
    sign_x = x & 1

    return torch.where(
        torch.eq(x | 1, LNS_ZERO), LNS_ZERO,
        torch.where(sign_x == 1,
                    LNS_NEG_ONE, LNS_ONE))

class LNSSignFunction(LNSFunction):
    """
    Sign becomes checking the sign bit (rightmost bit).

    Gradients are computed as follows:
    d/dx(sign(x)) = 0
    """

    @staticmethod
    def forward(x, base):
        x = x.view(torch.int64)
        result = _sign(x)
        return result.view(torch.float64)

    @staticmethod
    def setup_context(ctx, inputs, output):
        pass # no context needed for this operation

    @staticmethod
    def backward(ctx, grad_output):
        return torch.full_like(grad_output, LNS_ZERO), None

@implements(torch.sign, _sign, "default", default=True)
def sign(x, *, out=None):

    result = LNSSignFunction.apply(x, x.base)

    if out is not None:
        return out._inplace_copy(result)

    return lnstensor(result, from_lns=True, b=x.base)

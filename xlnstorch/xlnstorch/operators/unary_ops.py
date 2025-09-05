import torch
from xlnstorch import LNS_ZERO, LNS_ONE, LNS_NEG_ONE, implements
from xlnstorch.autograd import LNSFunction

def _neg(ops, x):
    return x ^ 1

class LNSNegFunction(LNSFunction):
    """
    Negation becomes flipping the sign bit.

    Gradients are computed as follows:
    d/dx(-x) = -1
    """

    @staticmethod
    def forward(ops, x):
        return _neg(ops, x)

    @staticmethod
    def setup_context(ctx, ops, inputs, output):
        pass # no context needed for this operation

    @staticmethod
    def backward(ctx, ops, grad_output):
        grad_x = ops.neg(grad_output)
        return grad_x

@implements(torch.neg, _neg, key="default", default=True)
def neg(x, *, out=None):
    result = LNSNegFunction.apply(x)

    if out is not None:
        return out._inplace_copy(result)

    return result

def _abs(ops, x):
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
    def forward(ops, x):
        return _abs(ops, x)

    @staticmethod
    def setup_context(ctx, ops, inputs, output):
        x, = inputs
        ctx.save_for_backward(x)

    @staticmethod
    def backward(ctx, ops, grad_output):
        x, = ctx.saved_tensors

        grad_x = torch.where(torch.eq(x & 1, 1), ops.neg(grad_output), grad_output)

        return grad_x

@implements(torch.abs, _abs, "default", default=True)
def abs(x, *, out=None):
    result = LNSAbsFunction.apply(x)

    if out is not None:
        return out._inplace_copy(result)

    return result

def _positive(ops, x):
    return x

class LNSPositiveFunction(LNSFunction):
    """
    This is implemented solely for completeness, this
    operation returns the input.

    Gradients are calculated as follows:
    d/dx(x) = 1
    """

    @staticmethod
    def forward(ops, x):
        return x

    @staticmethod
    def setup_context(ctx, ops, inputs, output):
        pass # no context needed for this operation

    @staticmethod
    def backward(ctx, ops, grad_output):
        return grad_output

@implements(torch.positive, _positive, "default", default=True)
def positive(x):
    return LNSPositiveFunction.apply(x)

def _sign(ops, x):
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
    def forward(ops, x):
        return _sign(ops, x)

    @staticmethod
    def setup_context(ctx, ops, inputs, output):
        pass # no context needed for this operation

    @staticmethod
    def backward(ctx, ops, grad_output):
        grad_x = ops.zeros_like(grad_output)
        return grad_x

@implements(torch.sign, _sign, "default", default=True)
def sign(x, *, out=None):
    result = LNSSignFunction.apply(x)

    if out is not None:
        return out._inplace_copy(result)

    return result

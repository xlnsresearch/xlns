import torch
from .. import LNS_ZERO, LNSTensor, lnstensor, format_lnstensor_operands, implements
from . import (
    lns_mul,
    lns_add,
    lns_gt,
)

class LNSReLUFunction(torch.autograd.Function):
    """
    The ReLU activation function in LNS simply involves checking
    if the sign bit is set (i.e. if the value is negative).

    Gradients are computed as follows:
    d/dx(x) = 1 if x > 0 else 0
    """

    @staticmethod
    def forward(x, base):
        x_packed = x.to(torch.int64)

        result = torch.where(x_packed & 1 == 1, LNS_ZERO, x_packed)
        return result.to(torch.float64)

    @staticmethod
    def setup_context(ctx, inputs, output):
        _, base = inputs
        ctx.save_for_backward(output, base)

    @staticmethod
    def backward(ctx, grad_output):
        output, base = ctx.saved_tensors
        output_packed = output.to(torch.int64)

        grad_x = torch.where(output_packed | 1 == LNS_ZERO, LNS_ZERO, LNSTensor.get_internal_tensor(1.0, base))
        grad_x = lns_mul(grad_output, grad_x)

        return grad_x, None

@implements(torch.nn.functional.relu, LNSReLUFunction.forward, "default", default=True)
def relu(x, inplace=False):

    result = LNSReLUFunction.apply(x._lns, x.base)

    if inplace:
        x._lns = result._lns
        return x

    return lnstensor(result, from_lns=True, b=x.base)

@implements(torch.nn.functional.relu_, LNSReLUFunction.forward, "default", default=True)
def relu_(x):

    result = LNSReLUFunction.apply(x._lns, x.base)

    x._lns = result._lns
    return x

class LNSLeakyReLUFunction(torch.autograd.Function):
    """
    Again, the leaky ReLU activation function in LNS just involves
    checking the sign bit and applying a negative slope to the
    negative part of the input.

    Gradients are computed as follows:
    d/dx(x) = negative_slope if x < 0 else 1
    """

    @staticmethod
    def forward(x, negative_slope, base):
        x_packed = x.to(torch.int64)

        negative_part = lns_mul(x_packed, LNSTensor.get_internal_tensor(negative_slope, base))
        negative_result = torch.where(x_packed & 1 == 1, negative_part, LNS_ZERO)
        positive_result = torch.where(x_packed & 1 == 1, LNS_ZERO, x_packed)

        result = lns_add(negative_result, positive_result, base)
        return result.to(torch.float64)

    @staticmethod
    def setup_context(ctx, inputs, output):
        _, negative_slope, base = inputs
        ctx.save_for_backward(output, base)
        ctx.negative_slope = negative_slope

    @staticmethod
    def backward(ctx, grad_output):
        output, base = ctx.saved_tensors
        output_packed = output.to(torch.int64)

        grad_x = torch.where((output_packed | 1 == LNS_ZERO) | (output_packed & 1 == 1),
                             LNSTensor.get_internal_tensor(ctx.negative_slope, base),
                             LNSTensor.get_internal_tensor(1.0, base))
        grad_x = lns_mul(grad_output, grad_x)

        return grad_x, None, None

@implements(torch.nn.functional.leaky_relu, LNSLeakyReLUFunction.forward, "default", default=True)
def leaky_relu(x, negative_slope=0.01, inplace=False):

    result = LNSLeakyReLUFunction.apply(x._lns, negative_slope, x.base)

    if inplace:
        x._lns = result._lns
        return x

    return lnstensor(result, from_lns=True, b=x.base)

@implements(torch.nn.functional.leaky_relu_, LNSLeakyReLUFunction.forward, "default", default=True)
def leaky_relu_(x, negative_slope=0.01):

    result = LNSLeakyReLUFunction.apply(x._lns, negative_slope, x.base)

    x._lns = result._lns
    return x

class LNSThresholdFunction(torch.autograd.Function):

    @staticmethod
    def forward(x, threshold, value, base):
        x_packed = x.to(torch.int64)

        result = torch.where(lns_gt(x, LNSTensor.get_internal_tensor(threshold, base)),
                             x_packed, LNSTensor.get_internal_tensor(value, base))
        return result.to(torch.float64)

    @staticmethod
    def setup_context(ctx, inputs, output):
        _, _, value, base = inputs
        ctx.save_for_backward(output, base)
        ctx.value = value

    @staticmethod
    def backward(ctx, grad_output):
        output, base = ctx.saved_tensors
        output_packed = output.to(torch.int64)

        grad_x = torch.where(output_packed == LNSTensor.get_internal_tensor(ctx.value, base),
                             LNS_ZERO, LNSTensor.get_internal_tensor(1.0, base))
        grad_x = lns_mul(grad_output, grad_x)

        return grad_x, None, None, None

@implements(torch.nn.functional.threshold, LNSThresholdFunction.forward, "default", default=True)
def threshold(x, threshold, value, inplace=False):

    result = LNSThresholdFunction.apply(x._lns, threshold, value, x.base)

    if inplace:
        x._lns = result._lns
        return x

    return lnstensor(result, from_lns=True, b=x.base)

@implements(torch.nn.functional.threshold_, LNSThresholdFunction.forward, "default", default=True)
def threshold_(x, threshold, value):

    result = LNSThresholdFunction.apply(x._lns, threshold, value, x.base)

    x._lns = result._lns
    return x
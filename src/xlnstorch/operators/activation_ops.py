import torch
from .. import LNS_ZERO, LNSTensor, lnstensor, format_lnstensor_operands, implements
from . import (
    lns_mul,
    lns_add,
    lns_gt,
    lns_square,
    lns_sub,
    lns_exp,
    lns_sum,
    lns_div,
    lns_neg,
    lns_log,
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
        x._lns = result
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
        x._lns = result
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
        x._lns = result
        return x

    return lnstensor(result, from_lns=True, b=x.base)

@implements(torch.nn.functional.threshold_, LNSThresholdFunction.forward, "default", default=True)
def threshold_(x, threshold, value):

    result = LNSThresholdFunction.apply(x._lns, threshold, value, x.base)

    x._lns = result._lns
    return x

class LNSTanhFunction(torch.autograd.Function):
    """
    For now, we will implement the tanh function by converting
    the input back to its floating-point representation.

    Gradients are computed as follows:
    d/dx(tanh(x)) = 1 - tanh(x) ^ 2
    """

    @staticmethod
    def forward(x, base):
        x_fp = lnstensor(x, from_lns=True, b=base).value

        result = torch.tanh(x_fp)
        return lnstensor(result, b=base)._lns

    @staticmethod
    def setup_context(ctx, inputs, output):
        _, base = inputs
        ctx.save_for_backward(output, base)

    @staticmethod
    def backward(ctx, grad_output):
        output, base = ctx.saved_tensors
        output_packed = output.to(torch.int64)

        grad_x = lns_square(output_packed, base)
        grad_x = lns_sub(LNSTensor.get_internal_tensor(1.0, base), grad_x, base)
        grad_x = lns_mul(grad_output, grad_x)

        return grad_x, None

@implements(torch.tanh, LNSTanhFunction.forward, "default", default=True)
@implements(torch.nn.functional.tanh, LNSTanhFunction.forward, "default", default=True)
def tanh(x):
    result = LNSTanhFunction.apply(x._lns, x.base)
    return lnstensor(result, from_lns=True, b=x.base)

class LNSSigmoidFunction(torch.autograd.Function):
    """
    For now, we will implement the sigmoid function by
    converting the input back to its floating-point
    representation.

    Gradients are computed as follows:
    d/dx(sigmoid(x)) = sigmoid(x) * (1 - sigmoid(x))
    """

    @staticmethod
    def forward(x, base):
        x_fp = lnstensor(x, from_lns=True, b=base).value

        result = torch.sigmoid(x_fp)
        return lnstensor(result, b=base)._lns

    @staticmethod
    def setup_context(ctx, inputs, output):
        _, base = inputs
        ctx.save_for_backward(output, base)

    @staticmethod
    def backward(ctx, grad_output):
        output, base = ctx.saved_tensors
        output_packed = output.to(torch.int64)

        grad_x = lns_sub(LNSTensor.get_internal_tensor(1.0, base), output_packed, base)
        grad_x = lns_mul(output_packed, grad_x)
        grad_x = lns_mul(grad_output, grad_x)

        return grad_x, None

@implements(torch.sigmoid, LNSSigmoidFunction.forward, "default", default=True)
@implements(torch.nn.functional.sigmoid, LNSSigmoidFunction.forward, "default", default=True)
def sigmoid(x):
    result = LNSSigmoidFunction.apply(x._lns, x.base)
    return lnstensor(result, from_lns=True, b=x.base)

class LNSLogSigmoidFunction(torch.autograd.Function):
    """
    For now, we will implement the log sigmoid function by
    converting the input back to its floating-point
    representation.

    Gradients are computed as follows:
    d/dx(log_sigmoid(x)) =  e ^ (log_sigmoid(x) - x)
    """

    @staticmethod
    def forward(x, base):
        x_fp = lnstensor(x, from_lns=True, b=base).value

        result = torch.nn.functional.logsigmoid(x_fp)
        return lnstensor(result, b=base)._lns

    @staticmethod
    def setup_context(ctx, inputs, output):
        x, base = inputs
        ctx.save_for_backward(x, output, base)

    @staticmethod
    def backward(ctx, grad_output):
        x, output, base = ctx.saved_tensors
        output_packed = output.to(torch.int64)

        grad_x = lns_sub(output, x, base)
        grad_x = lns_exp(grad_x, base)
        grad_x = lns_mul(grad_output, grad_x)

        return grad_x, None

@implements(torch.nn.functional.logsigmoid, LNSLogSigmoidFunction.forward, "default", default=True)
def logsigmoid(x):
    result = LNSLogSigmoidFunction.apply(x._lns, x.base)
    return lnstensor(result, from_lns=True, b=x.base)

class LNSSoftminFunction(torch.autograd.Function):
    """
    The softmin function in LNS involves exponentiation,
    which currently requires converting to floating-point
    and back.

    Gradients are computed as follows:
    d/dx(softmin(x)) = -softmin(x) * (1 - softmin(x))
    """

    @staticmethod
    def forward(x, base, dim=None):
        x_packed = x.to(torch.int64)

        neg_x = lns_neg(x_packed)
        exp_x = lns_exp(neg_x, base)
        sum_exp_x = lns_sum(exp_x, base, dim=dim, keepdim=True)

        result = lns_div(exp_x, sum_exp_x, base)
        return result.to(torch.float64)

    @staticmethod
    def setup_context(ctx, inputs, output):
        _, base, dim = inputs
        ctx.save_for_backward(output, base)
        ctx.dim = dim

    @staticmethod
    def backward(ctx, grad_output):
        output, base = ctx.saved_tensors

        dot_product = lns_sum(lns_mul(grad_output, output), base, dim=ctx.dim, keepdim=True)
        grad_x = lns_mul(output, lns_sub(grad_output, dot_product, base))

        return grad_x, None, None

@implements(torch.nn.functional.softmin, LNSSoftminFunction.forward, "default", default=True)
def softmin(x, dim=None, _stacklevel=3, dtype=None):
    result = LNSSoftminFunction.apply(x._lns, x.base, dim)
    return lnstensor(result, from_lns=True, b=x.base)

class LNSSoftmaxFunction(torch.autograd.Function):
    """
    The softmax function in LNS involves exponentiation,
    which currently requires converting to floating-point
    and back.

    Gradients are computed as follows:
    d/dx(softmax(x)) = softmax(x) * (1 - softmax(x))
    """

    @staticmethod
    def forward(x, base, dim=None):
        x_packed = x.to(torch.int64)

        exp_x = lns_exp(x_packed, base)
        sum_exp_x = lns_sum(exp_x, base, dim=dim, keepdim=True)

        result = lns_div(exp_x, sum_exp_x, base)
        return result.to(torch.float64)

    @staticmethod
    def setup_context(ctx, inputs, output):
        _, base, dim = inputs
        ctx.save_for_backward(output, base)
        ctx.dim = dim

    @staticmethod
    def backward(ctx, grad_output):
        output, base = ctx.saved_tensors

        dot_product = lns_sum(lns_mul(grad_output, output), base, dim=ctx.dim, keepdim=True)
        grad_x = lns_mul(output, lns_sub(grad_output, dot_product, base))

        return grad_x, None, None

@implements(torch.nn.functional.softmax, LNSSoftmaxFunction.forward, "default", default=True)
def softmax(x, dim=None, _stacklevel=3, dtype=None):
    result = LNSSoftmaxFunction.apply(x._lns, x.base, dim)
    return lnstensor(result, from_lns=True, b=x.base)

class LNSLogSoftmaxFunction(torch.autograd.Function):
    """
    The log softmax function in LNS involves exponentiation,
    which currently requires converting to floating-point
    and back.

    Gradients are computed as follows:
    d/dx(log_softmax(x)) = 1 - softmax(x)
    """

    @staticmethod
    def forward(x, base, dim=None):
        x_packed = x.to(torch.int64)

        exp_x = lns_exp(x_packed, base)
        sum_exp_x = lns_sum(exp_x, base, dim=dim, keepdim=True)
        log_sum_exp_x = lns_log(sum_exp_x, base)

        result = lns_sub(x_packed, log_sum_exp_x, base)
        return result.to(torch.float64)

    @staticmethod
    def setup_context(ctx, inputs, output):
        _, base, dim = inputs
        ctx.save_for_backward(output, base)
        ctx.dim = dim

    @staticmethod
    def backward(ctx, grad_output):
        output, base = ctx.saved_tensors

        softmax = lns_exp(output, base)
        sum_grad = lns_sum(grad_output, base, dim=ctx.dim, keepdim=True)
        product = lns_mul(softmax, sum_grad)
        grad_x = lns_sub(grad_output, product, base)

        return grad_x, None, None

@implements(torch.nn.functional.log_softmax, LNSLogSoftmaxFunction.forward, "default", default=True)
def log_softmax(x, dim=None, _stacklevel=3, dtype=None):
    result = LNSLogSoftmaxFunction.apply(x._lns, x.base, dim)
    return lnstensor(result, from_lns=True, b=x.base)
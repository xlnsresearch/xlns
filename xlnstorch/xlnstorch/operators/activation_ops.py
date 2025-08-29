import torch
from xlnstorch import LNS_ZERO, LNS_ONE, LNSTensor, lnstensor, format_lnstensor_operands, implements, rand
from xlnstorch.autograd import LNSFunction
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
    lns_lt,
    lns_gt,
    lns_le,
    lns_ge,
    lns_sigmoid,
    lns_abs,
    lns_eq,
    lns_tanh,
    lns_max,
)

def _relu(x):
    result = torch.where(x & 1 == 1, LNS_ZERO, x)
    return result

class LNSReLUFunction(LNSFunction):
    """
    The ReLU activation function in LNS simply involves checking
    if the sign bit is set (i.e. if the value is negative).

    Gradients are computed as follows:
    d/dx(x) = 1 if x > 0 else 0
    """

    @staticmethod
    def forward(x):
        x = x.view(torch.int64)
        result = _relu(x)
        return result.view(torch.float64)

    @staticmethod
    def setup_context(ctx, inputs, output):
        ctx.save_for_backward(output)

    @staticmethod
    def backward(ctx, grad_output):
        output, = ctx.saved_tensors
        output, grad_output = output.view(torch.int64), grad_output.view(torch.int64)

        grad_x = torch.where(output | 1 == LNS_ZERO, LNS_ZERO, LNS_ONE)
        grad_x = lns_mul(grad_output, grad_x)

        return grad_x.view(torch.float64)

@implements(torch.nn.functional.relu, _relu, "default", default=True)
def relu(x, inplace=False):

    result = LNSReLUFunction.apply(x)

    if inplace:
        return x._inplace_copy(result)

    return lnstensor(result, from_lns=True, b=x.base)

@implements(torch.nn.functional.relu_, _relu, "default", default=True)
def relu_(x):

    result = LNSReLUFunction.apply(x)
    return x._inplace_copy(result)

def _leaky_relu(x, negative_slope):
    negative_part = lns_mul(x, negative_slope)
    result = torch.where(x & 1 == 1, negative_part, x)
    return result

class LNSLeakyReLUFunction(LNSFunction):
    """
    Again, the leaky ReLU activation function in LNS just involves
    checking the sign bit and applying a negative slope to the
    negative part of the input.

    Gradients are computed as follows:
    d/dx(x) = negative_slope if x < 0 else 1
    """

    @staticmethod
    def forward(x, negative_slope):
        x, negative_slope = x.view(torch.int64), negative_slope.view(torch.int64)
        result = _leaky_relu(x, negative_slope)
        return result.view(torch.float64)

    @staticmethod
    def setup_context(ctx, inputs, output):
        _, negative_slope = inputs
        ctx.save_for_backward(negative_slope, output)

    @staticmethod
    def backward(ctx, grad_output):
        negative_slope, output = ctx.saved_tensors
        negative_slope, output, grad_output = negative_slope.view(torch.int64), output.view(torch.int64), grad_output.view(torch.int64)

        grad_x = torch.where((output | 1 == LNS_ZERO) | (output & 1 == 1),
                             negative_slope, LNS_ONE)
        grad_x = lns_mul(grad_output, grad_x)

        return grad_x, None

@implements(torch.nn.functional.leaky_relu, _leaky_relu, "default", default=True)
def leaky_relu(x, negative_slope=0.01, inplace=False):

    x, negative_slope = format_lnstensor_operands(x, negative_slope)
    result = LNSLeakyReLUFunction.apply(x, negative_slope)

    if inplace:
        return x._inplace_copy(result)

    return lnstensor(result, from_lns=True, b=x.base)

@implements(torch.nn.functional.leaky_relu_, _leaky_relu, "default", default=True)
def leaky_relu_(x, negative_slope=0.01):

    x, negative_slope = format_lnstensor_operands(x, negative_slope)
    result = LNSLeakyReLUFunction.apply(x, negative_slope)

    return x._inplace_copy(result)

def _threshold(x, threshold, value):
    result = torch.where(lns_gt(x, threshold), x, value)
    return result

class LNSThresholdFunction(LNSFunction):

    @staticmethod
    def forward(x, threshold, value):
        x, threshold, value = x.view(torch.int64), threshold.view(torch.int64), value.view(torch.int64)
        result = _threshold(x, threshold, value)
        return result.view(torch.float64)

    @staticmethod
    def setup_context(ctx, inputs, output):
        _, _, value = inputs
        ctx.save_for_backward(value, output)

    @staticmethod
    def backward(ctx, grad_output):
        value, output = ctx.saved_tensors
        value, output, grad_output = value.view(torch.int64), output.view(torch.int64), grad_output.view(torch.int64)

        grad_x = torch.where(output == value, LNS_ZERO, LNS_ONE)
        grad_x = lns_mul(grad_output, grad_x)

        return grad_x.view(torch.float64), None, None

@implements(torch.nn.functional.threshold, _threshold, "default", default=True)
def threshold(x, threshold, value, inplace=False):

    x, threshold, value = format_lnstensor_operands(x, threshold, value)
    result = LNSThresholdFunction.apply(x, threshold, value)

    if inplace:
        x._inplace_copy(result)
        return x

    return lnstensor(result, from_lns=True, b=x.base)

@implements(torch.nn.functional.threshold_, _threshold, "default", default=True)
def threshold_(x, threshold, value):

    x, threshold, value = format_lnstensor_operands(x, threshold, value)
    result = LNSThresholdFunction.apply(x, threshold, value)

    x._inplace_copy(result)
    return x

def _tanh(x, base):
    x_fp = lnstensor(x, from_lns=True, b=base).value
    result = torch.tanh(x_fp)
    return lnstensor(result, b=base)._lns.view(torch.int64)

class LNSTanhFunction(LNSFunction):
    """
    For now, we will implement the tanh function by converting
    the input back to its floating-point representation.

    Gradients are computed as follows:
    d/dx(tanh(x)) = 1 - tanh(x) ^ 2
    """

    @staticmethod
    def forward(x, base):
        x = x.view(torch.int64)
        result = _tanh(x, base)
        return result.view(torch.float64)

    @staticmethod
    def setup_context(ctx, inputs, output):
        _, base = inputs
        ctx.save_for_backward(output, base)

    @staticmethod
    def backward(ctx, grad_output):
        output, base = ctx.saved_tensors
        output, grad_output = output.view(torch.int64), grad_output.view(torch.int64)

        grad_x = lns_square(output)
        grad_x = lns_sub(LNS_ONE, grad_x, base)
        grad_x = lns_mul(grad_output, grad_x)

        return grad_x.view(torch.float64), None

@implements(torch.tanh, _tanh, "default", default=True)
@implements(torch.nn.functional.tanh, _tanh, "default", default=True)
def tanh(x):
    result = LNSTanhFunction.apply(x, x.base)
    return lnstensor(result, from_lns=True, b=x.base)

def _sigmoid(x, base):
    x_fp = lnstensor(x, from_lns=True, b=base).value
    result = torch.sigmoid(x_fp)
    return lnstensor(result, b=base)._lns.view(torch.int64)

class LNSSigmoidFunction(LNSFunction):
    """
    For now, we will implement the sigmoid function by
    converting the input back to its floating-point
    representation.

    Gradients are computed as follows:
    d/dx(sigmoid(x)) = sigmoid(x) * (1 - sigmoid(x))
    """

    @staticmethod
    def forward(x, base):
        x = x.view(torch.int64)
        result = _sigmoid(x, base)
        return result.view(torch.float64)

    @staticmethod
    def setup_context(ctx, inputs, output):
        _, base = inputs
        ctx.save_for_backward(output, base)

    @staticmethod
    def backward(ctx, grad_output):
        output, base = ctx.saved_tensors
        output, grad_output = output.view(torch.int64), grad_output.view(torch.int64)

        grad_x = lns_sub(LNS_ONE, output, base)
        grad_x = lns_mul(output, grad_x)
        grad_x = lns_mul(grad_output, grad_x)

        return grad_x.view(torch.float64), None

@implements(torch.sigmoid, _sigmoid, "default", default=True)
@implements(torch.nn.functional.sigmoid, _sigmoid, "default", default=True)
def sigmoid(x):
    result = LNSSigmoidFunction.apply(x, x.base)
    return lnstensor(result, from_lns=True, b=x.base)

def _logsigmoid(x, base):
    x_fp = lnstensor(x, from_lns=True, b=base).value
    result = torch.nn.functional.logsigmoid(x_fp)
    return lnstensor(result, b=base)._lns.view(torch.int64)

class LNSLogSigmoidFunction(LNSFunction):
    """
    For now, we will implement the log sigmoid function by
    converting the input back to its floating-point
    representation.

    Gradients are computed as follows:
    d/dx(log_sigmoid(x)) =  e ^ (log_sigmoid(x) - x)
    """

    @staticmethod
    def forward(x, base):
        x = x.view(torch.int64)
        result = _logsigmoid(x, base)
        return result.view(torch.float64)

    @staticmethod
    def setup_context(ctx, inputs, output):
        x, base = inputs
        ctx.save_for_backward(x, output, base)

    @staticmethod
    def backward(ctx, grad_output):
        x, output, base = ctx.saved_tensors
        output, grad_output = output.view(torch.int64), grad_output.view(torch.int64)

        grad_x = lns_sub(output, x, base)
        grad_x = lns_exp(grad_x)
        grad_x = lns_mul(grad_output, grad_x)

        return grad_x.view(torch.float64), None

@implements(torch.nn.functional.logsigmoid, _logsigmoid, "default", default=True)
def logsigmoid(x):
    result = LNSLogSigmoidFunction.apply(x, x.base)
    return lnstensor(result, from_lns=True, b=x.base)

def _softmin(x, base, dim=None):
    neg_x = lns_neg(x)
    exp_x = lns_exp(neg_x, base)
    sum_exp_x = lns_sum(exp_x, base, dim=dim, keepdim=True)

    result = lns_div(exp_x, sum_exp_x)
    return result

class LNSSoftminFunction(LNSFunction):
    """
    The softmin function in LNS involves exponentiation,
    which currently requires converting to floating-point
    and back.

    Gradients are computed as follows:
    d/dx(softmin(x)) = -softmin(x) * (1 - softmin(x))
    """

    @staticmethod
    def forward(x, base, dim=None):
        x = x.view(torch.int64)
        result = _softmin(x, base, dim)
        return result.view(torch.float64)

    @staticmethod
    def setup_context(ctx, inputs, output):
        _, base, dim = inputs
        ctx.save_for_backward(output, base)
        ctx.dim = dim

    @staticmethod
    def backward(ctx, grad_output):
        output, base = ctx.saved_tensors
        output, grad_output = output.view(torch.int64), grad_output.view(torch.int64)

        dot_product = lns_sum(lns_mul(grad_output, output), base, dim=ctx.dim, keepdim=True)
        grad_x = lns_mul(output, lns_sub(grad_output, dot_product, base))

        return grad_x.view(torch.float64), None, None

@implements(torch.nn.functional.softmin, _softmin, "default", default=True)
def softmin(x, dim=None, _stacklevel=3, dtype=None):
    result = LNSSoftminFunction.apply(x, x.base, dim)
    return lnstensor(result, from_lns=True, b=x.base)

def _softmax(x, base, dim=None):
    exp_x = lns_exp(x, base)
    sum_exp_x = lns_sum(exp_x, base, dim=dim, keepdim=True)

    result = lns_div(exp_x, sum_exp_x)
    return result

class LNSSoftmaxFunction(LNSFunction):
    """
    The softmax function in LNS involves exponentiation,
    which currently requires converting to floating-point
    and back.

    Gradients are computed as follows:
    d/dx(softmax(x)) = softmax(x) * (1 - softmax(x))
    """

    @staticmethod
    def forward(x, base, dim=None):
        x = x.view(torch.int64)
        result = _softmax(x, base, dim)
        return result.view(torch.float64)

    @staticmethod
    def setup_context(ctx, inputs, output):
        _, base, dim = inputs
        ctx.save_for_backward(output, base)
        ctx.dim = dim

    @staticmethod
    def backward(ctx, grad_output):
        output, base = ctx.saved_tensors
        output, grad_output = output.view(torch.int64), grad_output.view(torch.int64)

        dot_product = lns_sum(lns_mul(grad_output, output), base, dim=ctx.dim, keepdim=True)
        grad_x = lns_mul(output, lns_sub(grad_output, dot_product, base))

        return grad_x.view(torch.float64), None, None

@implements(torch.nn.functional.softmax, _softmax, "default", default=True)
def softmax(x, dim=None, _stacklevel=3, dtype=None):
    result = LNSSoftmaxFunction.apply(x, x.base, dim)
    return lnstensor(result, from_lns=True, b=x.base)

def _log_softmax(x, base, dim=None):
    m = lns_max(x, dim=dim, keepdim=True)[0] # discard indices

    # subtract the max to prevent overflow (logsumexp trick)
    x_sub_m = lns_sub(x, m, base)
    exp_x_sub_m = lns_exp(x_sub_m, base)
    sum_exp_x_sub_m = lns_sum(exp_x_sub_m, base, dim=dim, keepdim=True)
    log_sum_exp_x_sub_m = lns_log(sum_exp_x_sub_m, base)

    result = lns_sub(x_sub_m, log_sum_exp_x_sub_m, base)
    return result

class LNSLogSoftmaxFunction(LNSFunction):
    """
    The log softmax function in LNS involves exponentiation,
    which currently requires converting to floating-point
    and back.

    Gradients are computed as follows:
    d/dx(log_softmax(x)) = 1 - softmax(x)
    """

    @staticmethod
    def forward(x, base, dim=None):
        x = x.view(torch.int64)
        result = _log_softmax(x, base, dim)
        return result.view(torch.float64)

    @staticmethod
    def setup_context(ctx, inputs, output):
        _, base, dim = inputs
        ctx.save_for_backward(output, base)
        ctx.dim = dim

    @staticmethod
    def backward(ctx, grad_output):
        output, base = ctx.saved_tensors
        output, grad_output = output.view(torch.int64), grad_output.view(torch.int64)

        softmax = lns_exp(output, base)
        sum_grad = lns_sum(grad_output, base, dim=ctx.dim, keepdim=True)
        product = lns_mul(softmax, sum_grad)
        grad_x = lns_sub(grad_output, product, base)

        return grad_x.view(torch.float64), None, None

@implements(torch.nn.functional.log_softmax, _log_softmax, "default", default=True)
def log_softmax(x, dim=None, _stacklevel=3, dtype=None):
    result = LNSLogSoftmaxFunction.apply(x, x.base, dim)
    return lnstensor(result, from_lns=True, b=x.base)

def _hardtanh(x, min_val, max_val):
    result = torch.where(lns_lt(x, min_val), min_val, x)
    result = torch.where(lns_gt(result, max_val), max_val, result)
    return result

class LNSHardtanhFunction(LNSFunction):
    """
    The hardtanh function in LNS is implemented by checking
    the input value and clamping it to the given range.

    Gradients are computed as follows:
    d/dx(hardtanh(x)) = 1 if min_val < x < max_val else 0
    """

    @staticmethod
    def forward(x, min_val, max_val):
        x, min_val, max_val = x.view(torch.int64), min_val.view(torch.int64), max_val.view(torch.int64)
        result = _hardtanh(x, min_val, max_val)
        return result.view(torch.float64)

    @staticmethod
    def setup_context(ctx, inputs, output):
        _, min_val, max_val = inputs
        ctx.save_for_backward(min_val, max_val, output)

    @staticmethod
    def backward(ctx, grad_output):
        min_val, max_val, output = ctx.saved_tensors
        min_val, max_val = min_val.view(torch.int64), max_val.view(torch.int64)
        output, grad_output = output.view(torch.int64), grad_output.view(torch.int64)

        grad_x = torch.where(lns_le(output, min_val) | lns_ge(output, max_val), LNS_ZERO, LNS_ONE)
        grad_x = lns_mul(grad_output, grad_x)

        return grad_x.view(torch.float64), None, None, None

@implements(torch.nn.functional.hardtanh, _hardtanh, "default", default=True)
def hardtanh(x, min_val=-1.0, max_val=1.0, inplace=False):

    x, min_val, max_val = format_lnstensor_operands(x, min_val, max_val)
    result = LNSHardtanhFunction.apply(x, min_val, max_val)

    if inplace:
        return x._inplace_copy(result)

    return lnstensor(result, from_lns=True, b=x.base)

@implements(torch.nn.functional.hardtanh_, _hardtanh, "default", default=True)
def hardtanh_(x, min_val=-1.0, max_val=1.0):

    x, min_val, max_val = format_lnstensor_operands(x, min_val, max_val)
    result = LNSHardtanhFunction.apply(x, min_val, max_val)

    return x._inplace_copy(result)

def _hardswish(x, base):
    three = LNSTensor.get_internal_tensor(3.0, base)
    swish = lns_div(lns_mul(x, lns_add(x, three, base)),
                    LNSTensor.get_internal_tensor(6, base), base)

    result = torch.where(lns_le(x, LNSTensor.get_internal_tensor(-3, base)), LNS_ZERO,
                            torch.where(lns_ge(x, three), x, swish))

    return result

class LNSHardswishFunction(LNSFunction):
    """
    The hardswish function in LNS is implemented by checking
    the input value and applying the hardswish formula.

    Gradients are computed as follows:
    d/dx(hardswish(x)) = (2x + 3) / 6 if -3 < x < 3 else 1 if x >= 3 else 0
    """

    @staticmethod
    def forward(x, base):
        x = x.view(torch.int64)
        result = _hardswish(x, base)
        return result.view(torch.float64)

    @staticmethod
    def setup_context(ctx, inputs, output):
        x, base = inputs
        ctx.save_for_backward(x, base)

    @staticmethod
    def backward(ctx, grad_output):
        x, base = ctx.saved_tensors
        x, grad_output = x.view(torch.int64), grad_output.view(torch.int64)

        grad_swish = lns_div(lns_add(lns_mul(x, LNSTensor.get_internal_tensor(2, base), base),
                                     LNSTensor.get_internal_tensor(3, base), base),
                                     LNSTensor.get_internal_tensor(6, base), base)

        grad_x = torch.where(lns_le(x, LNSTensor.get_internal_tensor(-3, base)), LNS_ZERO,
                             torch.where(lns_ge(x, LNSTensor.get_internal_tensor(3, base)),
                                         LNS_ONE, grad_swish))
        grad_x = lns_mul(grad_output, grad_x)

        return grad_x.view(torch.float64), None

@implements(torch.nn.functional.hardswish, _hardswish, "default", default=True)
def hardswish(x, inplace=False):

    result = LNSHardswishFunction.apply(x, x.base)

    if inplace:
        return x._inplace_copy(result)

    return lnstensor(result, from_lns=True, b=x.base)

def _relu6(x, base):
    six = LNSTensor.get_internal_tensor(6, base)
    result = torch.where(lns_lt(x, LNS_ZERO), LNS_ZERO,
                            torch.where(lns_gt(x, six), six, x))
    return result

class LNSReLU6Function(LNSFunction):
    """
    The ReLU6 function in LNS is implemented by checking
    the input value and clamping it to the range [0, 6].

    Gradients are computed as follows:
    d/dx(ReLU6(x)) = 1 if 0 < x < 6 else 0
    """

    @staticmethod
    def forward(x, base):
        x = x.view(torch.int64)
        result = _relu6(x, base)
        return result.view(torch.float64)

    @staticmethod
    def setup_context(ctx, inputs, output):
        _, base = inputs
        ctx.save_for_backward(output, base)

    @staticmethod
    def backward(ctx, grad_output):
        output, base = ctx.saved_tensors
        output, grad_output = output.view(torch.int64), grad_output.view(torch.int64)

        grad_x = torch.where(lns_le(output, LNS_ZERO) | lns_ge(output, LNSTensor.get_internal_tensor(6, base)),
                             LNS_ZERO, LNS_ONE)
        grad_x = lns_mul(grad_output, grad_x)

        return grad_x, None

@implements(torch.nn.functional.relu6, _relu6, "default", default=True)
def relu6(x, inplace=False):

    result = LNSReLU6Function.apply(x, x.base)

    if inplace:
        return x._inplace_copy(result)

    return lnstensor(result, from_lns=True, b=x.base)

def _elu(x, alpha, base):
    negative_part = lns_mul(alpha, lns_sub(lns_exp(x, base), LNS_ONE, base))
    result = torch.where(lns_gt(x, LNS_ZERO), x, negative_part)
    return result

class LNSELUFunction(LNSFunction):
    """
    The ELU function in LNS is implemented by checking
    the input value and applying the ELU formula.

    Gradients are computed as follows:
    d/dx(ELU(x)) = 1 if x > 0 else alpha * exp(x) if x <= 0
    """

    @staticmethod
    def forward(x, alpha, base):
        x, alpha = x.view(torch.int64), alpha.view(torch.int64)
        result = _elu(x, alpha, base)
        return result.view(torch.float64)

    @staticmethod
    def setup_context(ctx, inputs, output):
        x, alpha, base = inputs
        ctx.save_for_backward(x, alpha, base)

    @staticmethod
    def backward(ctx, grad_output):
        x, alpha, base = ctx.saved_tensors
        x, alpha = x.view(torch.int64), alpha.view(torch.int64)

        grad_x = torch.where(lns_gt(x, LNS_ZERO), LNS_ONE, lns_mul(alpha, lns_exp(x, base)))
        grad_x = lns_mul(grad_output, grad_x)

        return grad_x.view(torch.float64), None, None

@implements(torch.nn.functional.elu, _elu, "default", default=True)
def elu(x, alpha=1.0, inplace=False):

    x, alpha = format_lnstensor_operands(x, alpha)

    result = LNSELUFunction.apply(x, alpha, x.base)

    if inplace:
        return x._inplace_copy(result)

    return lnstensor(result, from_lns=True, b=x.base)

@implements(torch.nn.functional.elu_, _elu, "default", default=True)
def elu_(x, alpha=1.0):

    x, alpha = format_lnstensor_operands(x, alpha)
    result = LNSELUFunction.apply(x, alpha, x.base)

    return x._inplace_copy(result)

def _selu(x, base):
    scale = LNSTensor.get_internal_tensor(1.6732632423543772848170429916717, base)
    alpha = LNSTensor.get_internal_tensor(1.0507009873554804934193349852946, base)

    negative_part = lns_mul(scale, lns_mul(alpha, lns_sub(lns_exp(x, base), LNS_ONE, base)))
    result = torch.where(lns_gt(x, LNS_ZERO), lns_mul(scale, x, base), negative_part)

    return result

class LNSSELUFunction(LNSFunction):
    """
    The SELU function in LNS is implemented by checking
    the input value and applying the SELU formula.

    Gradients are computed as follows:
    d/dx(SELU(x)) = scale if x > 0 else scale * alpha * exp(x) if x <= 0
    """

    @staticmethod
    def forward(x, base):
        x = x.view(torch.int64)
        result = _selu(x, base)
        return result.view(torch.float64)

    @staticmethod
    def setup_context(ctx, inputs, output):
        x, base = inputs
        ctx.save_for_backward(x, base)

    @staticmethod
    def backward(ctx, grad_output):
        x, base = ctx.saved_tensors
        x = x.view(torch.int64)

        scale = LNSTensor.get_internal_tensor(1.6732632423543772848170429916717, base)
        alpha = LNSTensor.get_internal_tensor(1.0507009873554804934193349852946, base)

        grad_x = torch.where(lns_gt(x, LNS_ZERO),
                             lns_mul(scale, LNS_ONE),
                             lns_mul(scale, lns_mul(alpha, lns_exp(x, base))))
        grad_x = lns_mul(grad_output, grad_x)

        return grad_x.view(torch.float64), None, None

@implements(torch.nn.functional.selu, _selu, "default", default=True)
def selu(x, inplace=False):

    result = LNSSELUFunction.apply(x, x.base)

    if inplace:
        return x._inplace_copy(result)

    return lnstensor(result, from_lns=True, b=x.base)

def _celu(x, alpha, base):
    negative_part = lns_mul(alpha, lns_sub(lns_exp(lns_div(x, alpha), base), LNS_ONE, base))
    result = torch.where(lns_gt(x, LNS_ZERO), x, negative_part)
    return result

class LNSCELUFunction(LNSFunction):
    """
    The CELU function in LNS is implemented by checking
    the input value and applying the CELU formula.

    Gradients are computed as follows:
    d/dx(CELU(x)) = 1 if x > 0 else exp(x / alpha) if x <= 0
    """

    @staticmethod
    def forward(x, alpha, base):
        x, alpha = x.view(torch.int64), alpha.view(torch.int64)
        result = _celu(x, alpha, base)
        return result.view(torch.float64)

    @staticmethod
    def setup_context(ctx, inputs, output):
        x, alpha, base = inputs
        ctx.save_for_backward(x, alpha, base)

    @staticmethod
    def backward(ctx, grad_output):
        x, alpha, base = ctx.saved_tensors
        x, alpha = x.view(torch.int64), alpha.view(torch.int64)

        grad_x = torch.where(lns_gt(x, LNS_ZERO), LNS_ONE, lns_exp(lns_div(x, alpha), base))
        grad_x = lns_mul(grad_output, grad_x)

        return grad_x.view(torch.float64), None, None

@implements(torch.nn.functional.celu, _celu, "default", default=True)
def celu(x, alpha=1.0, inplace=False):

    x, alpha = format_lnstensor_operands(x, alpha)
    result = LNSCELUFunction.apply(x, alpha, x.base)

    if inplace:
        return x._inplace_copy(result)

    return lnstensor(result, from_lns=True, b=x.base)

def _prelu(x, a):
    negative_part = lns_mul(x, a)
    result = torch.where(lns_gt(x, LNS_ZERO), x, negative_part)
    return result

class LNSPReLUFunction(LNSFunction):
    """
    The PReLU function in LNS is implemented by checking
    the input value and applying the PReLU formula.

    Gradients are computed as follows:
    d/dx(PReLU(x, a)) = a if x < 0 else 1
    d/da(PReLU(x, a)) = x if x < 0 else 0
    """

    @staticmethod
    def forward(x, a, base):
        x, a = x.view(torch.int64), a.view(torch.int64)
        result = _prelu(x, a)
        return result.view(torch.float64)

    @staticmethod
    def setup_context(ctx, inputs, output):
        x, a, base = inputs
        ctx.save_for_backward(x, a, base)

    @staticmethod
    def backward(ctx, grad_output):
        x, a, base = ctx.saved_tensors
        x, a = x.view(torch.int64), a.view(torch.int64)

        negative_mask = lns_le(x, LNS_ZERO)

        grad_x = torch.where(negative_mask, a, LNS_ONE)
        grad_x = lns_mul(grad_output, grad_x)

        grad_a = torch.where(negative_mask, x, LNS_ZERO)
        grad_a = lns_mul(grad_output, grad_a)

        while grad_a.dim() > a.dim():
            grad_a = lns_sum(grad_a, base, dim=0)

        for i, dim in enumerate(a.shape):
            if dim == 1:
                grad_a = lns_sum(grad_a, base, dim=i, keepdim=True)

        grad_a = lns_add(grad_a, LNS_ONE, base)

        return grad_x.view(torch.float64), grad_a.view(torch.float64), None

@implements(torch.nn.functional.prelu, _prelu, "default", default=True)
def prelu(x, a, inplace=False):

    x, a = format_lnstensor_operands(x, a)
    result = LNSPReLUFunction.apply(x, a, x.base)

    if inplace:
        return x._inplace_copy(result)

    return lnstensor(result, from_lns=True, b=x.base)

def _rrelu(x, a):
    negative_part = lns_mul(x, a)
    result = torch.where(lns_gt(x, LNS_ZERO), x, negative_part)

    return result

class LNSRReLUFunction(LNSFunction):
    """
    The RReLU function in LNS is implemented by checking
    the input value and applying the RReLU formula. This
    function is identical to LeakyReLU, but the slope `a`
    is randomly sampled from a uniform distribution between
    `lower` and `upper` during the forward pass.

    Gradients are computed as follows:
    d/dx(RReLU(x)) = a if x < 0 else 1
    """

    @staticmethod
    def forward(x, a):
        x, a = x.view(torch.int64), a.view(torch.int64)
        result = _rrelu(x, a)
        return result.view(torch.float64)

    @staticmethod
    def setup_context(ctx, inputs, output):
        x, a = inputs
        ctx.save_for_backward(x, a)

    @staticmethod
    def backward(ctx, grad_output):
        x, a = ctx.saved_tensors
        x, a = x.view(torch.int64), a.view(torch.int64)

        negative_mask = lns_lt(x, LNS_ZERO)

        grad_x = torch.where(negative_mask, a, LNS_ONE)
        grad_x = lns_mul(grad_output, grad_x)

        return grad_x.view(torch.float64), None

@implements(torch.nn.functional.rrelu, _rrelu, "default", default=True)
def rrelu(x, lower=1/8, upper=1/3, training=False, inplace=False):

    if training:
        a = rand(*x.shape, b=x.base) * (upper - lower) + lower
    else:
        a = lnstensor((lower + upper) / 2, b=x.base)

    result = LNSRReLUFunction.apply(x, a)

    if inplace:
        return x._inplace_copy(result)

    return lnstensor(result, from_lns=True, b=x.base)

@implements(torch.nn.functional.rrelu_, LNSRReLUFunction.forward, "default", default=True)
def rrelu_(x, lower=1/8, upper=1/3, training=False):

    if training:
        a = rand(x.shape, b=x.base) * (upper - lower) + lower
    else:
        a = lnstensor((lower + upper) / 2, b=x.base)

    result = LNSRReLUFunction.apply(x, a, x.base)
    return x._inplace_copy(result)

def _glu(x, base, dim=-1):
    half_size = x.size(dim) // 2

    a = x.narrow(dim, 0, half_size)
    b = x.narrow(dim, half_size, half_size)

    sigmoid_b = lns_sigmoid(b, base)
    result = lns_mul(a, sigmoid_b)

    return result

class LNSGLUFunction(LNSFunction):
    """
    The GLU function in LNS is implemented by splitting the input
    into two halves and applying the sigmoid activation to the
    second half, then multiplying it with the first half.

    Gradients are computed as follows:
    d/da(GLU(a, b)) = sigmoid(b)
    d/db(GLU(a, b)) = a * sigmoid(b) * (1 - sigmoid(b))
    """

    @staticmethod
    def forward(x, base, dim=-1):
        x = x.view(torch.int64)
        result = _glu(x, base, dim)
        return result.view(torch.float64)

    @staticmethod
    def setup_context(ctx, inputs, output):
        x, base, dim = inputs
        ctx.save_for_backward(x, base)
        ctx.dim = dim

    @staticmethod
    def backward(ctx, grad_output):
        x, base = ctx.saved_tensors
        x, grad_output = x.view(torch.int64), grad_output.view(torch.int64)

        half_size = x.size(ctx.dim) // 2
        a = x.narrow(ctx.dim, 0, half_size)
        b = x.narrow(ctx.dim, half_size, half_size)

        sigmoid_b = lns_sigmoid(b, base)
        grad_a = lns_mul(grad_output, sigmoid_b)
        grad_b = lns_sub(LNS_ONE, sigmoid_b, base)
        grad_b = lns_mul(grad_output, lns_mul(a, lns_mul(sigmoid_b, grad_b)))

        grad_x = torch.cat([grad_a, grad_b], dim=ctx.dim)

        return grad_x.view(torch.float64), None, None

@implements(torch.nn.functional.glu, _glu, "default", default=True)
def glu(x, dim=-1):
    result = LNSGLUFunction.apply(x, x.base, dim)
    return lnstensor(result, from_lns=True, b=x.base)

def _hardshrink(x, lambd):
    result = torch.where(lns_le(lns_abs(x), lambd), LNS_ZERO, x)
    return result

class LNSHardshrinkFunction(LNSFunction):
    """
    The hardshrink function in LNS is implemented by checking
    the input value and applying the hardshrink formula.

    Gradients are computed as follows:
    d/dx(hardshrink(x)) = 0 if |x| < lambd else 1
    """

    @staticmethod
    def forward(x, lambd):
        x, lambd = x.view(torch.int64), lambd.view(torch.int64)
        result = _hardshrink(x, lambd)
        return result.view(torch.float64)

    @staticmethod
    def setup_context(ctx, inputs, output):
        ctx.save_for_backward(output)

    @staticmethod
    def backward(ctx, grad_output):
        output, = ctx.saved_tensors
        output, grad_output = output.view(torch.int64), grad_output.view(torch.int64)

        grad_x = torch.where(lns_eq(output, LNS_ZERO), LNS_ZERO, LNS_ONE)
        grad_x = lns_mul(grad_output, grad_x)

        return grad_x.view(torch.float64), None, None

@implements(torch.nn.functional.hardshrink, _hardshrink, "default", default=True)
def hardshrink(x, lambd=0.5):

    x, lambd = format_lnstensor_operands(x, lambd)
    result = LNSHardshrinkFunction.apply(x, lambd)

    return lnstensor(result, from_lns=True, b=x.base)

def _tanhshrink(x, base):
    tanh_x = lns_tanh(x, base)
    result = lns_sub(x, tanh_x, base)
    return result

class LNSTanhshrinkFunction(LNSFunction):
    """
    The tanhshrink function in LNS is implemented by applying
    the tanh function and subtracting it from the input.

    Gradients are computed as follows:
    d/dx(tanhshrink(x)) = 1 - tanh(x) ^ 2
    """

    @staticmethod
    def forward(x, base):
        x = x.view(torch.int64)
        result = _tanhshrink(x, base)
        return result.view(torch.float64)

    @staticmethod
    def setup_context(ctx, inputs, output):
        x, base = inputs
        ctx.save_for_backward(x, base)

    @staticmethod
    def backward(ctx, grad_output):
        x, base = ctx.saved_tensors
        x, grad_output = x.view(torch.int64), grad_output.view(torch.int64)

        tanh_x = lns_tanh(x, base)
        grad_x = lns_mul(tanh_x, tanh_x)
        grad_x = lns_mul(grad_output, grad_x)

        return grad_x, None

@implements(torch.nn.functional.tanhshrink, _tanhshrink, "default", default=True)
def tanhshrink(x):
    result = LNSTanhshrinkFunction.apply(x, x.base)
    return lnstensor(result, from_lns=True, b=x.base)

def _softsign(x, base):
    abs_x = lns_abs(x)
    denominator = lns_add(abs_x, LNS_ONE, base)
    result = lns_div(x, denominator, base)

    return result

class LNSSoftsignFunction(LNSFunction):
    """
    The softsign function in LNS is implemented by dividing
    the input by the sum of its absolute value and 1.

    Gradients are computed as follows:
    d/dx(softsign(x)) = 1 / (|x| + 1) ^ 2
    """

    @staticmethod
    def forward(x, base):
        x = x.view(torch.int64)
        result = _softsign(x, base)
        return result.view(torch.float64)

    @staticmethod
    def setup_context(ctx, inputs, output):
        x, _ = inputs
        ctx.save_for_backward(x, output)

    @staticmethod
    def backward(ctx, grad_output):
        x, output = ctx.saved_tensors
        x, output, grad_output = x.view(torch.int64), output.view(torch.int64), grad_output.view(torch.int64)

        denominator = lns_div(output, x)
        grad_x = lns_mul(grad_output, lns_mul(denominator, denominator))

        return grad_x.view(torch.float64), None

@implements(torch.nn.functional.softsign, _softsign, "default", default=True)
def softsign(x):
    result = LNSSoftsignFunction.apply(x, x.base)
    return lnstensor(result, from_lns=True, b=x.base)

def _softplus(x, beta, threshold, base):
    threshold_mask = lns_gt(lns_mul(x, beta), threshold)
    result = torch.where(threshold_mask, x, lns_log(lns_add(lns_exp(x, base), LNS_ONE, base), base))

    return result

class LNSSoftplusFunction(LNSFunction):
    """
    The softplus function in LNS is implemented by applying
    the softplus formula.

    Gradients are computed as follows:
    d/dx(softplus(x)) = sigmoid(x) if x <= threshold / beta else 1
    """

    @staticmethod
    def forward(x, beta, threshold, base):
        x, beta, threshold = x.view(torch.int64), beta.view(torch.int64), threshold.view(torch.int64)
        result = _softplus(x, beta, threshold, base)
        return result.view(torch.float64)

    @staticmethod
    def setup_context(ctx, inputs, output):
        x, beta, threshold, base = inputs
        ctx.save_for_backward(x, beta, threshold, base)

    @staticmethod
    def backward(ctx, grad_output):
        x, beta, threshold, base = ctx.saved_tensors
        x, beta, threshold = x.view(torch.int64), beta.view(torch.int64), threshold.view(torch.int64)

        threshold_mask = lns_gt(lns_mul(x, beta), threshold)

        grad_x = lns_sigmoid(x, base)
        grad_x = lns_mul(grad_output, grad_x)
        grad_x = torch.where(threshold_mask, LNS_ONE, grad_x)

        return grad_x.view(torch.float64), None, None, None

@implements(torch.nn.functional.softplus, _softplus, "default", default=True)
def softplus(x, beta=1.0, threshold=20.0):

    x, beta, threshold = format_lnstensor_operands(x, beta, threshold)
    result = LNSSoftplusFunction.apply(x, beta, threshold, x.base)

    return lnstensor(result, from_lns=True, b=x.base)

def _softshrink(x, lambd, base):
    result = torch.where(lns_gt(lns_abs(x), lambd),
                         torch.where(lns_gt(x, LNS_ZERO), lns_sub(x, lambd, base),
                                        lns_add(x, lambd, base)), LNS_ZERO)
    return result

class LNSSoftshrinkFunction(LNSFunction):
    """
    The softshrink function in LNS is implemented by checking
    the input value and applying the softshrink formula.

    Gradients are computed as follows:
    d/dx(softshrink(x)) = 1 if |x| > lambd else 0
    """

    @staticmethod
    def forward(x, lambd, base):
        x, lambd = x.view(torch.int64), lambd.view(torch.int64)
        result = _softshrink(x, lambd, base)
        return result.view(torch.float64)

    @staticmethod
    def setup_context(ctx, inputs, output):
        ctx.save_for_backward(output)

    @staticmethod
    def backward(ctx, grad_output):
        output, = ctx.saved_tensors
        output, grad_output = output.view(torch.int64), grad_output.view(torch.int64)

        grad_x = torch.where(lns_eq(output, LNS_ZERO), LNS_ZERO, LNS_ONE)
        grad_x = lns_mul(grad_output, grad_x)

        return grad_x.view(torch.float64), None, None

@implements(torch.nn.functional.softshrink, _softshrink, "default", default=True)
def softshrink(x, lambd=0.5):

    x, lambd = format_lnstensor_operands(x, lambd)
    result = LNSSoftshrinkFunction.apply(x, lambd, x.base)

    return lnstensor(result, from_lns=True, b=x.base)

def _hardsigmoid(x, base):
    three = LNSTensor.get_internal_tensor(3, base)
    result = torch.where(lns_lt(x, lns_neg(three)), LNS_ZERO,
                         torch.where(lns_gt(x, three), LNS_ONE,
                                     lns_div(lns_add(x, three, base), LNSTensor.get_internal_tensor(6, base))))
    return result

class LNSHardsigmoidFunction(LNSFunction):
    """
    The hard sigmoid function in LNS is implemented by checking
    the input value and applying the hard sigmoid formula.

    Gradients are computed as follows:
    d/dx(hard_sigmoid(x)) = 1 if -3 < x < 3 else 0
    """

    @staticmethod
    def forward(x, base):
        x = x.view(torch.int64)
        result = _hardsigmoid(x, base)
        return result.view(torch.float64)

    @staticmethod
    def setup_context(ctx, inputs, output):
        x, base = inputs
        ctx.save_for_backward(x, base)

    @staticmethod
    def backward(ctx, grad_output):
        x, base = ctx.saved_tensors
        x, grad_output = x.view(torch.int64), grad_output.view(torch.int64)

        grad_x = torch.where(lns_gt(lns_abs(x), LNSTensor.get_internal_tensor(3, base)),
                             LNS_ZERO, LNSTensor.get_internal_tensor(1/6, base))
        grad_x = lns_mul(grad_output, grad_x)

        return grad_x.view(torch.float64), None

@implements(torch.nn.functional.hardsigmoid, _hardsigmoid, "default", default=True)
def hardsigmoid(x, inplace=False):

    result = LNSHardsigmoidFunction.apply(x, x.base)

    if inplace:
        return x._inplace_copy(result)

    return lnstensor(result, from_lns=True, b=x.base)

def _silu(x, base):
    sigmoid_x = lns_sigmoid(x, base)
    result = lns_mul(x, sigmoid_x)
    return result

class LNSSiLUFunction(LNSFunction):
    """
    The SiLU (Sigmoid Linear Unit) function in LNS is implemented
    by multiplying the input by the sigmoid of the input.

    Gradients are computed as follows:
    d/dx(silu(x)) = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
    """

    @staticmethod
    def forward(x, base):
        x = x.view(torch.int64)
        result = _silu(x, base)
        return result.view(torch.float64)

    @staticmethod
    def setup_context(ctx, inputs, output):
        x, base = inputs
        ctx.save_for_backward(x, base)

    @staticmethod
    def backward(ctx, grad_output):
        x, base = ctx.saved_tensors
        x, grad_output = x.view(torch.int64), grad_output.view(torch.int64)

        sigmoid_x = lns_sigmoid(x, base)
        one_minus_sigmoid_x = lns_sub(LNS_ONE, sigmoid_x, base)
        grad_x = lns_mul(x, lns_mul(sigmoid_x, one_minus_sigmoid_x))
        grad_x = lns_add(sigmoid_x, grad_x, base)
        grad_x = lns_mul(grad_output, grad_x)

        return grad_x.view(torch.float64), None

@implements(torch.nn.functional.silu, _silu, "default", default=True)
def silu(x, inplace=False):

    result = LNSSiLUFunction.apply(x, x.base)

    if inplace:
        return x._inplace_copy(result)

    return lnstensor(result, from_lns=True, b=x.base)
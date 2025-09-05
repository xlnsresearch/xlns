import torch
from xlnstorch import LNS_ZERO, LNS_ONE, lnstensor, format_lnstensor_operands, implements, rand
from xlnstorch.autograd import LNSFunction

def _relu(ops, x):
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
    def forward(ops, x):
        return _relu(ops, x)

    @staticmethod
    def setup_context(ctx, ops, inputs, output):
        ctx.save_for_backward(output)

    @staticmethod
    def backward(ctx, ops, grad_output):
        output, = ctx.saved_tensors

        grad_x = torch.where(output | 1 == LNS_ZERO, LNS_ZERO, LNS_ONE)
        grad_x = ops.mul(grad_output, grad_x)

        return grad_x

@implements(torch.nn.functional.relu, _relu, "default", default=True)
def relu(x, inplace=False):
    result = LNSReLUFunction.apply(x)

    if inplace:
        return x._inplace_copy(result)

    return result

@implements(torch.nn.functional.relu_, _relu, "default", default=True)
def relu_(x):
    result = LNSReLUFunction.apply(x)
    return x._inplace_copy(result)

def _leaky_relu(ops, x, negative_slope):
    negative_part = ops.mul(x, negative_slope)
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
    def forward(ops, x, negative_slope):
        return _leaky_relu(ops, x, negative_slope)

    @staticmethod
    def setup_context(ctx, ops, inputs, output):
        _, negative_slope = inputs
        ctx.save_for_backward(negative_slope, output)

    @staticmethod
    def backward(ctx, ops, grad_output):
        negative_slope, output = ctx.saved_tensors

        grad_x = torch.where((output | 1 == LNS_ZERO) | (output & 1 == 1),
                             negative_slope, LNS_ONE)
        grad_x = ops.mul(grad_output, grad_x)

        return grad_x, None

@implements(torch.nn.functional.leaky_relu, _leaky_relu, "default", default=True)
def leaky_relu(x, negative_slope=0.01, inplace=False):
    x, negative_slope = format_lnstensor_operands(x, negative_slope)
    result = LNSLeakyReLUFunction.apply(x, negative_slope)

    if inplace:
        return x._inplace_copy(result)

    return result

@implements(torch.nn.functional.leaky_relu_, _leaky_relu, "default", default=True)
def leaky_relu_(x, negative_slope=0.01):
    x, negative_slope = format_lnstensor_operands(x, negative_slope)
    result = LNSLeakyReLUFunction.apply(x, negative_slope)

    return x._inplace_copy(result)

def _threshold(ops, x, threshold, value):
    result = torch.where(ops.gt(x, threshold), x, value)
    return result

class LNSThresholdFunction(LNSFunction):

    @staticmethod
    def forward(ops, x, threshold, value):
        return _threshold(ops, x, threshold, value)

    @staticmethod
    def setup_context(ctx, ops, inputs, output):
        _, _, value = inputs
        ctx.save_for_backward(value, output)

    @staticmethod
    def backward(ctx, ops, grad_output):
        value, output = ctx.saved_tensors

        grad_x = torch.where(output == value, LNS_ZERO, LNS_ONE)
        grad_x = ops.mul(grad_output, grad_x)

        return grad_x, None, None

@implements(torch.nn.functional.threshold, _threshold, "default", default=True)
def threshold(x, threshold, value, inplace=False):
    x, threshold, value = format_lnstensor_operands(x, threshold, value)
    result = LNSThresholdFunction.apply(x, threshold, value)

    if inplace:
        x._inplace_copy(result)
        return x

    return result

@implements(torch.nn.functional.threshold_, _threshold, "default", default=True)
def threshold_(x, threshold, value):
    x, threshold, value = format_lnstensor_operands(x, threshold, value)
    result = LNSThresholdFunction.apply(x, threshold, value)

    x._inplace_copy(result)
    return x

def _tanh(ops, x):
    x_fp = ops.from_lns(x)
    result = torch.tanh(x_fp)
    return ops.to_lns(result)

class LNSTanhFunction(LNSFunction):
    """
    For now, we will implement the tanh function by converting
    the input back to its floating-point representation.

    Gradients are computed as follows:
    d/dx(tanh(x)) = 1 - tanh(x) ^ 2
    """

    @staticmethod
    def forward(ops, x):
        return _tanh(ops, x)

    @staticmethod
    def setup_context(ctx, ops, inputs, output):
        ctx.save_for_backward(output)

    @staticmethod
    def backward(ctx, ops, grad_output):
        output, = ctx.saved_tensors

        grad_x = ops.square(output)
        grad_x = ops.sub(LNS_ONE, grad_x)
        grad_x = ops.mul(grad_output, grad_x)

        return grad_x

@implements(torch.tanh, _tanh, "default", default=True)
@implements(torch.nn.functional.tanh, _tanh, "default", default=True)
def tanh(x):
    return LNSTanhFunction.apply(x)

def _sigmoid(ops, x):
    x_fp = ops.from_lns(x)
    result = torch.sigmoid(x_fp)
    return ops.to_lns(result)

class LNSSigmoidFunction(LNSFunction):
    """
    For now, we will implement the sigmoid function by
    converting the input back to its floating-point
    representation.

    Gradients are computed as follows:
    d/dx(sigmoid(x)) = sigmoid(x) * (1 - sigmoid(x))
    """

    @staticmethod
    def forward(ops, x):
        return _sigmoid(ops, x)

    @staticmethod
    def setup_context(ctx, ops, inputs, output):
        ctx.save_for_backward(output)

    @staticmethod
    def backward(ctx, ops, grad_output):
        output, = ctx.saved_tensors

        grad_x = ops.sub(LNS_ONE, output)
        grad_x = ops.mul(output, grad_x)
        grad_x = ops.mul(grad_output, grad_x)

        return grad_x

@implements(torch.sigmoid, _sigmoid, "default", default=True)
@implements(torch.nn.functional.sigmoid, _sigmoid, "default", default=True)
def sigmoid(x):
    return LNSSigmoidFunction.apply(x)

def _logsigmoid(ops, x):
    x_fp = ops.from_lns(x)
    result = torch.nn.functional.logsigmoid(x_fp)
    return ops.to_lns(result)

class LNSLogSigmoidFunction(LNSFunction):
    """
    For now, we will implement the log sigmoid function by
    converting the input back to its floating-point
    representation.

    Gradients are computed as follows:
    d/dx(log_sigmoid(x)) =  e ^ (log_sigmoid(x) - x)
    """

    @staticmethod
    def forward(ops, x):
        return _logsigmoid(ops, x)

    @staticmethod
    def setup_context(ctx, ops, inputs, output):
        x, = inputs
        ctx.save_for_backward(x, output)

    @staticmethod
    def backward(ctx, ops, grad_output):
        x, output = ctx.saved_tensors

        grad_x = ops.sub(output, x)
        grad_x = ops.exp(grad_x)
        grad_x = ops.mul(grad_output, grad_x)

        return grad_x

@implements(torch.nn.functional.logsigmoid, _logsigmoid, "default", default=True)
def logsigmoid(x):
    return LNSLogSigmoidFunction.apply(x)

def _softmin(ops, x, dim=None):
    neg_x = ops.neg(x)
    exp_x = ops.exp(neg_x)
    sum_exp_x = ops.sum(exp_x, dim=dim, keepdim=True)

    result = ops.div(exp_x, sum_exp_x)
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
    def forward(ops, x, dim=None):
        return _softmin(ops, x, dim)

    @staticmethod
    def setup_context(ctx, ops, inputs, output):
        _, dim = inputs
        ctx.save_for_backward(output)
        ctx.dim = dim

    @staticmethod
    def backward(ctx, ops, grad_output):
        output, = ctx.saved_tensors

        dot_product = ops.sum(ops.mul(grad_output, output), dim=ctx.dim, keepdim=True)
        grad_x = ops.mul(output, ops.sub(grad_output, dot_product))

        return grad_x, None

@implements(torch.nn.functional.softmin, _softmin, "default", default=True)
def softmin(x, dim=None, _stacklevel=3, dtype=None):
    return LNSSoftminFunction.apply(x, dim)

def _softmax(ops, x, dim=None):
    exp_x = ops.exp(x)
    sum_exp_x = ops.sum(exp_x, dim=dim, keepdim=True)

    result = ops.div(exp_x, sum_exp_x)
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
    def forward(ops, x, dim=None):
        return _softmax(ops, x, dim)

    @staticmethod
    def setup_context(ctx, ops, inputs, output):
        _, dim = inputs
        ctx.save_for_backward(output)
        ctx.dim = dim

    @staticmethod
    def backward(ctx, ops, grad_output):
        output, = ctx.saved_tensors

        dot_product = ops.sum(ops.mul(grad_output, output), dim=ctx.dim, keepdim=True)
        grad_x = ops.mul(output, ops.sub(grad_output, dot_product))

        return grad_x, None

@implements(torch.nn.functional.softmax, _softmax, "default", default=True)
def softmax(x, dim=None, _stacklevel=3, dtype=None):
    return LNSSoftmaxFunction.apply(x, dim)

def _log_softmax(ops, x, dim=None):
    if dim is None:
        m = ops.max(x)
    else:
        m = ops.max(x, dim=dim, keepdim=True)[0] # discard indices

    # subtract the max to prevent overflow (logsumexp trick)
    x_sub_m = ops.sub(x, m)
    exp_x_sub_m = ops.exp(x_sub_m)
    sum_exp_x_sub_m = ops.sum(exp_x_sub_m, dim=dim, keepdim=True)
    log_sum_exp_x_sub_m = ops.log(sum_exp_x_sub_m)

    result = ops.sub(x_sub_m, log_sum_exp_x_sub_m)
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
    def forward(ops, x, dim=None):
        return _log_softmax(ops, x, dim)

    @staticmethod
    def setup_context(ctx, ops, inputs, output):
        _, dim = inputs
        ctx.save_for_backward(output)
        ctx.dim = dim

    @staticmethod
    def backward(ctx, ops, grad_output):
        output, = ctx.saved_tensors

        softmax = ops.exp(output)
        sum_grad = ops.sum(grad_output, dim=ctx.dim, keepdim=True)
        product = ops.mul(softmax, sum_grad)
        grad_x = ops.sub(grad_output, product)

        return grad_x, None

@implements(torch.nn.functional.log_softmax, _log_softmax, "default", default=True)
def log_softmax(x, dim=None, _stacklevel=3, dtype=None):
    return LNSLogSoftmaxFunction.apply(x, dim)

def _hardtanh(ops, x, min_val, max_val):
    result = torch.where(ops.lt(x, min_val), min_val, x)
    result = torch.where(ops.gt(result, max_val), max_val, result)
    return result

class LNSHardtanhFunction(LNSFunction):
    """
    The hardtanh function in LNS is implemented by checking
    the input value and clamping it to the given range.

    Gradients are computed as follows:
    d/dx(hardtanh(x)) = 1 if min_val < x < max_val else 0
    """

    @staticmethod
    def forward(ops, x, min_val, max_val):
        return _hardtanh(ops, x, min_val, max_val)

    @staticmethod
    def setup_context(ctx, ops, inputs, output):
        _, min_val, max_val = inputs
        ctx.save_for_backward(min_val, max_val, output)

    @staticmethod
    def backward(ctx, ops, grad_output):
        min_val, max_val, output = ctx.saved_tensors

        grad_x = torch.where(ops.le(output, min_val) | ops.ge(output, max_val), LNS_ZERO, LNS_ONE)
        grad_x = ops.mul(grad_output, grad_x)

        return grad_x, None, None

@implements(torch.nn.functional.hardtanh, _hardtanh, "default", default=True)
def hardtanh(x, min_val=-1.0, max_val=1.0, inplace=False):
    x, min_val, max_val = format_lnstensor_operands(x, min_val, max_val)
    result = LNSHardtanhFunction.apply(x, min_val, max_val)

    if inplace:
        return x._inplace_copy(result)

    return result

@implements(torch.nn.functional.hardtanh_, _hardtanh, "default", default=True)
def hardtanh_(x, min_val=-1.0, max_val=1.0):
    x, min_val, max_val = format_lnstensor_operands(x, min_val, max_val)
    result = LNSHardtanhFunction.apply(x, min_val, max_val)
    return x._inplace_copy(result)

def _hardswish(ops, x):
    three = ops.to_lns(3.0)
    swish = ops.div(ops.mul(x, ops.add(x, three)), ops.to_lns(6.0))

    result = torch.where(ops.le(x, ops.neg(three)), LNS_ZERO,
                            torch.where(ops.ge(x, three), x, swish))

    return result

class LNSHardswishFunction(LNSFunction):
    """
    The hardswish function in LNS is implemented by checking
    the input value and applying the hardswish formula.

    Gradients are computed as follows:
    d/dx(hardswish(x)) = (2x + 3) / 6 if -3 < x < 3 else 1 if x >= 3 else 0
    """

    @staticmethod
    def forward(ops, x):
        return _hardswish(ops, x)

    @staticmethod
    def setup_context(ctx, ops, inputs, output):
        x, = inputs
        ctx.save_for_backward(x)

    @staticmethod
    def backward(ctx, ops, grad_output):
        x, = ctx.saved_tensors

        three = ops.to_lns(3.0)
        grad_swish = ops.div(ops.add(ops.mul(x, ops.to_lns(2.0)),
                                     three),
                                     ops.to_lns(6.0))

        grad_x = torch.where(ops.le(x, ops.neg(three)), LNS_ZERO,
                             torch.where(ops.ge(x, three),
                                         LNS_ONE, grad_swish))
        grad_x = ops.mul(grad_output, grad_x)

        return grad_x

@implements(torch.nn.functional.hardswish, _hardswish, "default", default=True)
def hardswish(x, inplace=False):
    result = LNSHardswishFunction.apply(x)

    if inplace:
        return x._inplace_copy(result)

    return result

def _relu6(ops, x):
    six = ops.to_lns(6.0)
    result = torch.where(ops.lt(x, LNS_ZERO), LNS_ZERO,
                            torch.where(ops.gt(x, six), six, x))
    return result

class LNSReLU6Function(LNSFunction):
    """
    The ReLU6 function in LNS is implemented by checking
    the input value and clamping it to the range [0, 6].

    Gradients are computed as follows:
    d/dx(ReLU6(x)) = 1 if 0 < x < 6 else 0
    """

    @staticmethod
    def forward(ops, x):
        return _relu6(ops, x)

    @staticmethod
    def setup_context(ctx, ops, inputs, output):
        ctx.save_for_backward(output)

    @staticmethod
    def backward(ctx, ops, grad_output):
        output, = ctx.saved_tensors

        grad_x = torch.where(ops.le(output, LNS_ZERO) | ops.ge(output, ops.to_lns(6.0)),
                             LNS_ZERO, LNS_ONE)
        grad_x = ops.mul(grad_output, grad_x)

        return grad_x

@implements(torch.nn.functional.relu6, _relu6, "default", default=True)
def relu6(x, inplace=False):
    result = LNSReLU6Function.apply(x)

    if inplace:
        return x._inplace_copy(result)

    return result

def _elu(ops, x, alpha):
    negative_part = ops.mul(alpha, ops.sub(ops.exp(x), LNS_ONE))
    result = torch.where(ops.gt(x, LNS_ZERO), x, negative_part)
    return result

class LNSELUFunction(LNSFunction):
    """
    The ELU function in LNS is implemented by checking
    the input value and applying the ELU formula.

    Gradients are computed as follows:
    d/dx(ELU(x)) = 1 if x > 0 else alpha * exp(x) if x <= 0
    """

    @staticmethod
    def forward(ops, x, alpha):
        return _elu(ops, x, alpha)

    @staticmethod
    def setup_context(ctx, ops, inputs, output):
        x, alpha = inputs
        ctx.save_for_backward(x, alpha)

    @staticmethod
    def backward(ctx, ops, grad_output):
        x, alpha = ctx.saved_tensors

        grad_x = torch.where(ops.gt(x, LNS_ZERO), LNS_ONE, ops.mul(alpha, ops.exp(x)))
        grad_x = ops.mul(grad_output, grad_x)

        return grad_x, None

@implements(torch.nn.functional.elu, _elu, "default", default=True)
def elu(x, alpha=1.0, inplace=False):
    x, alpha = format_lnstensor_operands(x, alpha)
    result = LNSELUFunction.apply(x, alpha)

    if inplace:
        return x._inplace_copy(result)

    return result

@implements(torch.nn.functional.elu_, _elu, "default", default=True)
def elu_(ops, x, alpha=1.0):
    x, alpha = format_lnstensor_operands(x, alpha)
    result = LNSELUFunction.apply(x, alpha)
    return x._inplace_copy(result)

def _selu(ops, x):
    scale = ops.to_lns(1.6732632423543772848170429916717)
    alpha = ops.to_lns(1.0507009873554804934193349852946)

    negative_part = ops.mul(scale, ops.mul(alpha, ops.sub(ops.exp(x), LNS_ONE)))
    result = torch.where(ops.gt(x, LNS_ZERO), ops.mul(scale, x), negative_part)

    return result

class LNSSELUFunction(LNSFunction):
    """
    The SELU function in LNS is implemented by checking
    the input value and applying the SELU formula.

    Gradients are computed as follows:
    d/dx(SELU(x)) = scale if x > 0 else scale * alpha * exp(x) if x <= 0
    """

    @staticmethod
    def forward(ops, x):
        return _selu(ops, x)

    @staticmethod
    def setup_context(ctx, ops, inputs, output):
        x, = inputs
        ctx.save_for_backward(x)

    @staticmethod
    def backward(ctx, ops, grad_output):
        x, = ctx.saved_tensors

        scale = ops.to_lns(1.6732632423543772848170429916717)
        alpha = ops.to_lns(1.0507009873554804934193349852946)

        grad_x = torch.where(ops.gt(x, LNS_ZERO),
                             ops.mul(scale, LNS_ONE),
                             ops.mul(scale, ops.mul(alpha, ops.exp(x))))
        grad_x = ops.mul(grad_output, grad_x)

        return grad_x

@implements(torch.nn.functional.selu, _selu, "default", default=True)
def selu(x, inplace=False):
    result = LNSSELUFunction.apply(x)

    if inplace:
        return x._inplace_copy(result)

    return result

def _celu(ops, x, alpha):
    negative_part = ops.mul(alpha, ops.sub(ops.exp(ops.div(x, alpha)), LNS_ONE))
    result = torch.where(ops.gt(x, LNS_ZERO), x, negative_part)
    return result

class LNSCELUFunction(LNSFunction):
    """
    The CELU function in LNS is implemented by checking
    the input value and applying the CELU formula.

    Gradients are computed as follows:
    d/dx(CELU(x)) = 1 if x > 0 else exp(x / alpha) if x <= 0
    """

    @staticmethod
    def forward(ops, x, alpha):
        return _celu(ops, x, alpha)

    @staticmethod
    def setup_context(ctx, ops, inputs, output):
        x, alpha = inputs
        ctx.save_for_backward(x, alpha)

    @staticmethod
    def backward(ctx, ops, grad_output):
        x, alpha = ctx.saved_tensors

        grad_x = torch.where(ops.gt(x, LNS_ZERO), LNS_ONE, ops.exp(ops.div(x, alpha)))
        grad_x = ops.mul(grad_output, grad_x)

        return grad_x, None

@implements(torch.nn.functional.celu, _celu, "default", default=True)
def celu(x, alpha=1.0, inplace=False):
    x, alpha = format_lnstensor_operands(x, alpha)
    result = LNSCELUFunction.apply(x, alpha)

    if inplace:
        return x._inplace_copy(result)

    return result

def _prelu(ops, x, a):
    negative_part = ops.mul(x, a)
    result = torch.where(ops.gt(x, LNS_ZERO), x, negative_part)
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
    def forward(ops, x, a):
        return _prelu(ops, x, a)

    @staticmethod
    def setup_context(ctx, ops, inputs, output):
        x, a = inputs
        ctx.save_for_backward(x, a)

    @staticmethod
    def backward(ctx, ops, grad_output):
        x, a = ctx.saved_tensors

        negative_mask = ops.le(x, LNS_ZERO)

        grad_x = torch.where(negative_mask, a, LNS_ONE)
        grad_x = ops.mul(grad_output, grad_x)

        grad_a = torch.where(negative_mask, x, LNS_ZERO)
        grad_a = ops.mul(grad_output, grad_a)

        while grad_a.dim() > a.dim():
            grad_a = ops.sum(grad_a, dim=0)

        for i, dim in enumerate(a.shape):
            if dim == 1:
                grad_a = ops.sum(grad_a, dim=i, keepdim=True)

        grad_a = ops.add(grad_a, LNS_ONE)

        return grad_x, grad_a, None

@implements(torch.nn.functional.prelu, _prelu, "default", default=True)
def prelu(x, a, inplace=False):
    x, a = format_lnstensor_operands(x, a)
    result = LNSPReLUFunction.apply(x, a)

    if inplace:
        return x._inplace_copy(result)

    return result

def _rrelu(ops, x, a):
    negative_part = ops.mul(x, a)
    result = torch.where(ops.gt(x, LNS_ZERO), x, negative_part)

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
    def forward(ops, x, a):
        return _rrelu(ops, x, a)

    @staticmethod
    def setup_context(ctx, ops, inputs, output):
        x, a = inputs
        ctx.save_for_backward(x, a)

    @staticmethod
    def backward(ctx, ops, grad_output):
        x, a = ctx.saved_tensors

        negative_mask = ops.lt(x, LNS_ZERO)

        grad_x = torch.where(negative_mask, a, LNS_ONE)
        grad_x = ops.mul(grad_output, grad_x)

        return grad_x, None

@implements(torch.nn.functional.rrelu, _rrelu, "default", default=True)
def rrelu(x, lower=1/8, upper=1/3, training=False, inplace=False):
    if training:
        a = rand(*x.shape, b=x.base) * (upper - lower) + lower
    else:
        a = lnstensor((lower + upper) / 2, b=x.base)

    result = LNSRReLUFunction.apply(x, a)

    if inplace:
        return x._inplace_copy(result)

    return result

@implements(torch.nn.functional.rrelu_, LNSRReLUFunction.forward, "default", default=True)
def rrelu_(x, lower=1/8, upper=1/3, training=False):
    if training:
        a = rand(x.shape, b=x.base) * (upper - lower) + lower
    else:
        a = lnstensor((lower + upper) / 2, b=x.base)

    result = LNSRReLUFunction.apply(x, a)
    return x._inplace_copy(result)

def _glu(ops, x, dim=-1):
    half_size = x.size(dim) // 2

    a = x.narrow(dim, 0, half_size)
    b = x.narrow(dim, half_size, half_size)

    sigmoid_b = ops.sigmoid(b)
    result = ops.mul(a, sigmoid_b)

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
    def forward(ops, x, dim=-1):
        return _glu(ops, x, dim)

    @staticmethod
    def setup_context(ctx, ops, inputs, output):
        x, dim = inputs
        ctx.save_for_backward(x)
        ctx.dim = dim

    @staticmethod
    def backward(ctx, ops, grad_output):
        x, = ctx.saved_tensors

        half_size = x.size(ctx.dim) // 2
        a = x.narrow(ctx.dim, 0, half_size)
        b = x.narrow(ctx.dim, half_size, half_size)

        sigmoid_b = ops.sigmoid(b)
        grad_a = ops.mul(grad_output, sigmoid_b)
        grad_b = ops.sub(LNS_ONE, sigmoid_b)
        grad_b = ops.mul(grad_output, ops.mul(a, ops.mul(sigmoid_b, grad_b)))

        grad_x = torch.cat([grad_a, grad_b], dim=ctx.dim)

        return grad_x, None

@implements(torch.nn.functional.glu, _glu, "default", default=True)
def glu(x, dim=-1):
    return LNSGLUFunction.apply(x, dim)

def _hardshrink(ops, x, lambd):
    result = torch.where(ops.le(ops.abs(x), lambd), LNS_ZERO, x)
    return result

class LNSHardshrinkFunction(LNSFunction):
    """
    The hardshrink function in LNS is implemented by checking
    the input value and applying the hardshrink formula.

    Gradients are computed as follows:
    d/dx(hardshrink(x)) = 0 if |x| < lambd else 1
    """

    @staticmethod
    def forward(ops, x, lambd):
        return _hardshrink(ops, x, lambd)

    @staticmethod
    def setup_context(ctx, ops, inputs, output):
        ctx.save_for_backward(output)

    @staticmethod
    def backward(ctx, ops, grad_output):
        output, = ctx.saved_tensors

        grad_x = torch.where(ops.eq(output, LNS_ZERO), LNS_ZERO, LNS_ONE)
        grad_x = ops.mul(grad_output, grad_x)

        return grad_x, None

@implements(torch.nn.functional.hardshrink, _hardshrink, "default", default=True)
def hardshrink(x, lambd=0.5):
    x, lambd = format_lnstensor_operands(x, lambd)
    return LNSHardshrinkFunction.apply(x, lambd)

def _tanhshrink(ops, x):
    tanh_x = ops.tanh(x)
    result = ops.sub(x, tanh_x)
    return result

class LNSTanhshrinkFunction(LNSFunction):
    """
    The tanhshrink function in LNS is implemented by applying
    the tanh function and subtracting it from the input.

    Gradients are computed as follows:
    d/dx(tanhshrink(x)) = 1 - tanh(x) ^ 2
    """

    @staticmethod
    def forward(ops, x):
        return _tanhshrink(ops, x)

    @staticmethod
    def setup_context(ctx, ops, inputs, output):
        x, = inputs
        ctx.save_for_backward(x)

    @staticmethod
    def backward(ctx, ops, grad_output):
        x, = ctx.saved_tensors

        tanh_x = ops.tanh(x)
        grad_x = ops.mul(tanh_x, tanh_x)
        grad_x = ops.mul(grad_output, grad_x)

        return grad_x

@implements(torch.nn.functional.tanhshrink, _tanhshrink, "default", default=True)
def tanhshrink(x):
    return LNSTanhshrinkFunction.apply(x)

def _softsign(ops, x):
    abs_x = ops.abs(x)
    denominator = ops.add(abs_x, LNS_ONE)
    result = ops.div(x, denominator)

    return result

class LNSSoftsignFunction(LNSFunction):
    """
    The softsign function in LNS is implemented by dividing
    the input by the sum of its absolute value and 1.

    Gradients are computed as follows:
    d/dx(softsign(x)) = 1 / (|x| + 1) ^ 2
    """

    @staticmethod
    def forward(ops, x):
        return _softsign(ops, x)

    @staticmethod
    def setup_context(ctx, inputs, output):
        x, = inputs
        ctx.save_for_backward(x, output)

    @staticmethod
    def backward(ctx, ops, grad_output):
        x, output = ctx.saved_tensors

        denominator = ops.div(output, x)
        grad_x = ops.mul(grad_output, ops.mul(denominator, denominator))

        return grad_x

@implements(torch.nn.functional.softsign, _softsign, "default", default=True)
def softsign(x):
    return LNSSoftsignFunction.apply(x)

def _softplus(ops, x, beta, threshold):
    threshold_mask = ops.gt(ops.mul(x, beta), threshold)
    result = torch.where(threshold_mask, x, ops.log(ops.add(ops.exp(x), LNS_ONE)))

    return result

class LNSSoftplusFunction(LNSFunction):
    """
    The softplus function in LNS is implemented by applying
    the softplus formula.

    Gradients are computed as follows:
    d/dx(softplus(x)) = sigmoid(x) if x <= threshold / beta else 1
    """

    @staticmethod
    def forward(ops, x, beta, threshold):
        return _softplus(ops, x, beta, threshold)

    @staticmethod
    def setup_context(ctx, ops, inputs, output):
        x, beta, threshold = inputs
        ctx.save_for_backward(x, beta, threshold)

    @staticmethod
    def backward(ctx, ops, grad_output):
        x, beta, threshold = ctx.saved_tensors

        threshold_mask = ops.gt(ops.mul(x, beta), threshold)

        grad_x = ops.sigmoid(x)
        grad_x = ops.mul(grad_output, grad_x)
        grad_x = torch.where(threshold_mask, LNS_ONE, grad_x)

        return grad_x, None, None

@implements(torch.nn.functional.softplus, _softplus, "default", default=True)
def softplus(x, beta=1.0, threshold=20.0):
    x, beta, threshold = format_lnstensor_operands(x, beta, threshold)
    return LNSSoftplusFunction.apply(x, beta, threshold)

def _softshrink(ops, x, lambd):
    result = torch.where(ops.gt(ops.abs(x), lambd),
                         torch.where(ops.gt(x, LNS_ZERO), ops.sub(x, lambd),
                                        ops.add(x, lambd)), LNS_ZERO)
    return result

class LNSSoftshrinkFunction(LNSFunction):
    """
    The softshrink function in LNS is implemented by checking
    the input value and applying the softshrink formula.

    Gradients are computed as follows:
    d/dx(softshrink(x)) = 1 if |x| > lambd else 0
    """

    @staticmethod
    def forward(ops, x, lambd):
        return _softshrink(ops, x, lambd)

    @staticmethod
    def setup_context(ctx, ops, inputs, output):
        ctx.save_for_backward(output)

    @staticmethod
    def backward(ctx, ops, grad_output):
        output, = ctx.saved_tensors

        grad_x = torch.where(ops.eq(output, LNS_ZERO), LNS_ZERO, LNS_ONE)
        grad_x = ops.mul(grad_output, grad_x)

        return grad_x, None

@implements(torch.nn.functional.softshrink, _softshrink, "default", default=True)
def softshrink(x, lambd=0.5):
    x, lambd = format_lnstensor_operands(x, lambd)
    return LNSSoftshrinkFunction.apply(x, lambd)

def _hardsigmoid(ops, x):
    three = ops.to_lns(3.0)
    result = torch.where(ops.lt(x, ops.neg(three)), LNS_ZERO,
                         torch.where(ops.gt(x, three), LNS_ONE,
                                     ops.div(ops.add(x, three), ops.to_lns(6.0))))
    return result

class LNSHardsigmoidFunction(LNSFunction):
    """
    The hard sigmoid function in LNS is implemented by checking
    the input value and applying the hard sigmoid formula.

    Gradients are computed as follows:
    d/dx(hard_sigmoid(x)) = 1 if -3 < x < 3 else 0
    """

    @staticmethod
    def forward(ops, x):
        return _hardsigmoid(ops, x)

    @staticmethod
    def setup_context(ctx, ops, inputs, output):
        x, = inputs
        ctx.save_for_backward(x)

    @staticmethod
    def backward(ctx, ops, grad_output):
        x, = ctx.saved_tensors

        grad_x = torch.where(ops.gt(ops.abs(x), ops.to_lns(3.0)),
                             LNS_ZERO, ops.to_lns(1.0 / 6.0))
        grad_x = ops.mul(grad_output, grad_x)

        return grad_x

@implements(torch.nn.functional.hardsigmoid, _hardsigmoid, "default", default=True)
def hardsigmoid(x, inplace=False):
    result = LNSHardsigmoidFunction.apply(x)

    if inplace:
        return x._inplace_copy(result)

    return result

def _silu(ops, x):
    sigmoid_x = ops.sigmoid(x)
    result = ops.mul(x, sigmoid_x)
    return result

class LNSSiLUFunction(LNSFunction):
    """
    The SiLU (Sigmoid Linear Unit) function in LNS is implemented
    by multiplying the input by the sigmoid of the input.

    Gradients are computed as follows:
    d/dx(silu(x)) = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
    """

    @staticmethod
    def forward(ops, x):
        return _silu(ops, x)

    @staticmethod
    def setup_context(ctx, ops, inputs, output):
        x, = inputs
        ctx.save_for_backward(x)

    @staticmethod
    def backward(ctx, ops, grad_output):
        x, = ctx.saved_tensors

        sigmoid_x = ops.sigmoid(x)
        one_minus_sigmoid_x = ops.sub(LNS_ONE, sigmoid_x)
        grad_x = ops.mul(x, ops.mul(sigmoid_x, one_minus_sigmoid_x))
        grad_x = ops.add(sigmoid_x, grad_x)
        grad_x = ops.mul(grad_output, grad_x)

        return grad_x

@implements(torch.nn.functional.silu, _silu, "default", default=True)
def silu(x, inplace=False):
    result = LNSSiLUFunction.apply(x)

    if inplace:
        return x._inplace_copy(result)

    return result
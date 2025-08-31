import torch
from xlnstorch import LNS_ZERO, CSRC_AVAILABLE, lnstensor, format_lnstensor_operands, implements, implements_sbdb, sbdb
from xlnstorch.autograd import LNSFunction

@implements_sbdb('ideal', default=True)
def sbdb_ideal(z, s, base):
    """
    Ideal implementation of the sbdb function that directly computes:
    log_(base)(1 - 2 * s + base ^ z)
    """
    power_term = torch.pow(base, z)
    magnitude = torch.abs(1.0 - 2.0 * s + power_term)

    log_term = torch.log(magnitude) / torch.log(base)
    result = torch.round(log_term).to(torch.int64) << 1

    return result

def _add(ops, x, y):
    max_operand = torch.max(x, y)

    abs_diff = torch.abs((x >> 1) - (y >> 1))
    sign_diff = (x ^ y) & 1

    result = max_operand + sbdb(-abs_diff, sign_diff, ops.base)
    return torch.where(
        torch.eq(x | 1, LNS_ZERO), y, torch.where(
            torch.eq(y | 1, LNS_ZERO), x, torch.where(
                x ^ 1 == y, LNS_ZERO, result)))

class LNSAddFunction(LNSFunction):
    """
    Addition is far more challenging in the logarithmic domain.
    We can implement different approximate methods for the sum
    and difference functions (Gaussian logarithms). See

    https://en.wikipedia.org/wiki/Logarithmic_number_system
    https://en.wikipedia.org/wiki/Gaussian_logarithm

    For two internal representations x and y, their addition can
    be computed as follows:
    max(x, y) + sbdb(-|(x >> 1) - (y >> 1)|, (x ^ y) & 1)

    Gradients are computed as follows:
    d/dx(x + y) = 1
    d/dy(x + y) = 1
    """

    @staticmethod
    def forward(ops, x, y):
        x, y = x.view(torch.int64), y.view(torch.int64)
        result = _add(ops, x, y)
        return result.view(torch.float64)

    @staticmethod
    def setup_context(ctx, ops, inputs, output):
        x, y = inputs
        ctx.save_for_backward(x, y)

    @staticmethod
    def backward(ctx, ops, grad_output):
        x, y = ctx.saved_tensors
        x, y, grad_output = x.view(torch.int64), y.view(torch.int64), grad_output.view(torch.int64)

        grad_x = ops.sum_to_size(grad_output, x.shape)
        grad_y = ops.sum_to_size(grad_output, y.shape)

        return grad_x.view(torch.float64), grad_y.view(torch.float64)

@implements(torch.add, _add, key='default', default=not CSRC_AVAILABLE)
def add(x, y, *, alpha=1, out=None):

    x, y = format_lnstensor_operands(x, y)

    if alpha != 1:
        y = torch.mul(y, alpha)

    result = LNSAddFunction.apply(x, y)

    if out is not None:
        return out._inplace_copy(result)

    return lnstensor(result, from_lns=True, b=x.base)

def _sub(ops, x, y):
    neg_y = ops.neg(y)
    return ops.add(x, neg_y)

class LNSSubFunction(LNSFunction):
    """
    See LNSAddFunction for details on the internal computations.

    Gradients are computed as follows:
    d/dx(x - y) = 1
    d/dy(x - y) = -1
    """

    @staticmethod
    def forward(ops, x, y):
        x, y = x.view(torch.int64), y.view(torch.int64)
        result = _sub(ops, x, y)
        return result.view(torch.float64)

    @staticmethod
    def setup_context(ctx, ops, inputs, output):
        x, y = inputs
        ctx.save_for_backward(x, y)

    @staticmethod
    def backward(ctx, ops, grad_output):
        x, y = ctx.saved_tensors
        x, y, grad_output = x.view(torch.int64), y.view(torch.int64), grad_output.view(torch.int64)

        grad_y = ops.neg(grad_output)

        grad_x = ops.sum_to_size(grad_output, x.shape)
        grad_y = ops.sum_to_size(grad_y, y.shape)

        return grad_x.view(torch.float64), grad_y.view(torch.float64)

@implements(torch.sub, _sub, key="default", default=True)
def sub(x, y, *, alpha=1, out=None):

    x, y = format_lnstensor_operands(x, y)

    if alpha != 1:
        y = torch.mul(y, alpha)

    result = LNSSubFunction.apply(x, y)

    if out is not None:
        return out._inplace_copy(result)

    return lnstensor(result, from_lns=True, b=x.base)

def _sum(ops, x, dim=None, keepdim=False):
    if dim is None:
        flat = x.reshape(-1)
        out = flat[0]

        for i in range(1, flat.numel()):
            out = ops.add(out, flat[i])

        if keepdim:
            out = out.reshape([1] * x.dim())

        return out

    red_dims = (dim,) if isinstance(dim, int) else tuple(dim)
    red_dims = tuple(sorted(d % x.dim() for d in red_dims))

    permute_order = [d for d in range(x.dim()) if d not in red_dims] + list(red_dims)
    transposed = x.permute(*permute_order)

    outer_shape = transposed.shape[:-len(red_dims)]
    transposed = transposed.reshape(*outer_shape, -1)

    out = transposed[..., 0]
    for i in range(1, transposed.shape[-1]):
        out = ops.add(out, transposed[..., i])

    if keepdim:
        for d in red_dims:
            out = out.unsqueeze(d)

    return out

class LNSSumFunction(LNSFunction):
    """
    We use the addition operation to compute the sum.

    Gradients are computed as follows:
    d/dx(sum(x)) = 1
    """

    @staticmethod
    def forward(ops, x, dim=None, keepdim=False):
        x = x.view(torch.int64)
        result = _sum(ops, x, dim, keepdim)
        return result.view(torch.float64)

    @staticmethod
    def setup_context(ctx, ops, inputs, output):
        x, dim, keepdim = inputs
        ctx.save_for_backward(x)
        ctx.dim = dim
        ctx.keepdim = keepdim

    @staticmethod
    def backward(ctx, ops, grad_output):
        x, = ctx.saved_tensors
        x, grad_output = x.view(torch.int64), grad_output.view(torch.int64)

        grad_x = grad_output
        if ctx.dim is None:
            grad_x = grad_x.expand(x.shape)

        else:
            red_dims = (ctx.dim,) if isinstance(ctx.dim, int) else tuple(ctx.dim)
            red_dims = tuple(d % x.dim() for d in red_dims)

            if not ctx.keepdim:
                for d in sorted(red_dims):
                    grad_x = grad_x.unsqueeze(d)

            grad_x = grad_x.expand(x.shape)

        return grad_x.view(torch.float64), None, None

@implements(torch.sum, _sum, "default", default=not CSRC_AVAILABLE)
def sum(x, dim=None, keepdim=False, *, out=None):

    result = LNSSumFunction.apply(x, dim, keepdim)

    if out is not None:
        return out._inplace_copy(result)

    return lnstensor(result, from_lns=True, b=x.base)
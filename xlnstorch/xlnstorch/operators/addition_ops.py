import torch
from xlnstorch import LNS_ZERO, LNS_ONE, CSRC_AVAILABLE, lnstensor, format_lnstensor_operands, implements, implements_sbdb, sbdb
from xlnstorch.autograd import LNSFunction
from . import (
    lns_add,
    lns_neg,
    lns_sum_to_size,
)

@implements_sbdb('ideal', default=True)
def sbdb_ideal(z, s, base):
    """
    Ideal implementation of the sbdb function that directly computes:
    log_(base)(1 - 2 * s + base ^ z)
    """
    power_term = torch.pow(base, z)
    magnitude = torch.abs(1.0 - 2.0 * s + power_term)

    log_term = torch.log(magnitude) / torch.log(base)
    result = torch.round(log_term) * 2

    return result.to(torch.float64)

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
    def forward(x, y, base):

        x_packed, y_packed = x.to(torch.int64), y.to(torch.int64)
        max_operand = torch.max(x_packed, y_packed)

        abs_diff = torch.abs((x_packed >> 1) - (y_packed >> 1))
        sign_diff = (x_packed ^ y_packed) & 1

        result = max_operand + sbdb(-abs_diff, sign_diff, base)
        return torch.where(
            torch.eq(x_packed | 1, LNS_ZERO), y, torch.where(
                torch.eq(y_packed | 1, LNS_ZERO), x, torch.where(
                    x_packed ^ 1 == y_packed, LNS_ZERO, result.to(torch.float64))))

    @staticmethod
    def setup_context(ctx, inputs, output):
        x, y, base = inputs
        ctx.save_for_backward(x, y, base)

    @staticmethod
    def backward(ctx, grad_output):
        x, y, base = ctx.saved_tensors

        grad_x = lns_sum_to_size(grad_output, base, x.shape)
        grad_y = lns_sum_to_size(grad_output, base, y.shape)

        return grad_x, grad_y, None

@implements(torch.add, LNSAddFunction.forward, key='default', default=not CSRC_AVAILABLE)
def add(x, y, *, alpha=1, out=None):

    x, y = format_lnstensor_operands(x, y)

    if alpha != 1:
        y = torch.mul(y, alpha)

    result = LNSAddFunction.apply(x, y, x.base)

    if out is not None:
        return out._inplace_copy(result)

    return lnstensor(result, from_lns=True, b=x.base)

class LNSSubFunction(LNSFunction):
    """
    See LNSAddFunction for details on the internal computations.

    Gradients are computed as follows:
    d/dx(x - y) = 1
    d/dy(x - y) = -1
    """

    @staticmethod
    def forward(x, y, base):
        neg_y = lns_neg(y)
        return lns_add(x, neg_y, base)

    @staticmethod
    def setup_context(ctx, inputs, output):
        x, y, base = inputs
        ctx.save_for_backward(x, y, base)

    @staticmethod
    def backward(ctx, grad_output):
        x, y, base = ctx.saved_tensors
        grad_y = lns_neg(grad_output)

        grad_x = lns_sum_to_size(grad_output, base, x.shape)
        grad_y = lns_sum_to_size(grad_y, base, y.shape)

        return grad_x, grad_y, None

@implements(torch.sub, LNSSubFunction.forward, key="default", default=True)
def sub(x, y, *, alpha=1, out=None):

    x, y = format_lnstensor_operands(x, y)

    if alpha != 1:
        y = torch.mul(y, alpha)

    result = LNSSubFunction.apply(x, y, x.base)

    if out is not None:
        return out._inplace_copy(result)

    return lnstensor(result, from_lns=True, b=x.base)

class LNSSumFunction(LNSFunction):
    """
    We use the addition operation to compute the sum.

    Gradients are computed as follows:
    d/dx(sum(x)) = 1
    """

    @staticmethod
    def forward(x, base, dim=None, keepdim=False):
        if dim is None:
            flat = x.reshape(-1)
            out = flat[0]
            for i in range(1, flat.numel()):
                out = lns_add(out, flat[i], base)
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
            out = lns_add(out, transposed[..., i], base)

        if keepdim:
            for d in red_dims:
                out = out.unsqueeze(d)

        return out

    @staticmethod
    def setup_context(ctx, inputs, output):
        x, _, dim, keepdim = inputs
        ctx.save_for_backward(x)
        ctx.dim = dim
        ctx.keepdim = keepdim

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors

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

        return grad_x, None, None, None

@implements(torch.sum, LNSSumFunction.forward, "default", default=not CSRC_AVAILABLE)
def sum(x, dim=None, keepdim=False, *, out=None):

    result = LNSSumFunction.apply(x, x.base, dim, keepdim)

    if out is not None:
        return out._inplace_copy(result)

    return lnstensor(result, from_lns=True, b=x.base)
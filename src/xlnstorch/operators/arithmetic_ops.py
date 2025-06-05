import torch
import xlns as xl
from ..tensor import apply_lns_op, implements, lnstensor, LNSTensor
from ..base import format_lnstensor_operands

class LNSMulFunction(torch.autograd.Function):
    """
    Multiplication becomes addition in the logarithmic domain.

    Gradients are computed as follows:
    d/dx(x * y) = y
    d/dy(x * y) = x
    """

    @staticmethod
    def forward(x, y):

        x_packed, y_packed = x.to(torch.int64), y.to(torch.int64)
        x_sign, y_sign = x_packed & 1, y_packed & 1
        x_log, y_log = x_packed >> 1, y_packed >> 1

        result_log = x_log + y_log
        result_sign = x_sign ^ y_sign
        result = (result_log << 1) | result_sign

        return result.to(torch.float64)

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

    x, y = format_lnstensor_operands(x, y)
    result = LNSMulFunction.apply(x._lns, y._lns)

    if out is not None:
        out._lns = result

    return lnstensor(result, from_lns=True, b=x.base)

class LNSSquareFunction(torch.autograd.Function):
    """
    Squaring becomes doubling in the logarithmic domain.
    
    Gradients are computed as follows:
    d/dx(x ^ 2) = 2 * x
    """

    @staticmethod
    def forward(x, base):
        return apply_lns_op(torch.mul, x, x)

    @staticmethod
    def setup_context(ctx, inputs, outputs):
        x, base = inputs
        ctx.save_for_backward(x, base)

    @staticmethod
    def backward(ctx, grad_output):
        x, base = ctx.saved_tensors

        grad_x = apply_lns_op(torch.mul, x, lnstensor(2.0, b=base)._lns)
        grad_x = apply_lns_op(torch.mul, grad_output, grad_x)

        return grad_x, None

@implements(torch.square, LNSSquareFunction.forward, key='default', default=True)
def square(x, *, out=None):

    result = LNSSquareFunction.apply(x._lns, x.base)

    if out is not None:
        out._lns = result

    return lnstensor(result, from_lns=True, b=x.base)

class LNSPowFunction(torch.autograd.Function):
    """
    Exponentiation becomes multiplication in the logarithmic domain.

    Gradients are computed as follows:
    d/dx(x ^ n) = n * x ^ (n - 1)
    """

    @staticmethod
    def forward(x, n, base):
        x_packed = x.to(torch.int64)

        if torch.is_floating_point(n):
            result = ((x_packed & (-2)) * n) & (-2)

        else:
            if n & 1 == 0:
                result = ((x_packed & (-2)) * n) & (-2)
            else:
                result = (((x_packed & (-2)) * n) & (-2)) | (x_packed & 1)

        return result.to(torch.float64)

    @staticmethod
    def setup_context(ctx, inputs, outputs):
        x, n, base = inputs
        ctx.save_for_backward(x, n, base)

    @staticmethod
    def backward(ctx, grad_output):
        pass # Placeholder until addition logic is implemented

@implements(torch.pow, LNSPowFunction.forward, key='default', default=True)
def pow(x, n, *, out=None):
    # todo: implement support for LNSTensor exponentiation
    result = LNSPowFunction.apply(x._lns, n, x.base)

    if out is not None:
        out._lns = result

    return lnstensor(result, from_lns=True, b=x.base)

class LNSDivFunction(torch.autograd.Function):
    """
    Division becomes subtraction in the logarithmic domain.

    Gradients are computed as follows:
    d/dx(x / y) = 1 / y
    d/dy(x / y) = -x / (y^2)
    """

    @staticmethod
    def forward(x, y, base):

        x_packed, y_packed = x.to(torch.int64), y.to(torch.int64)

        x_sign, y_sign = x_packed & 1, y_packed & 1
        x_log, y_log = x_packed >> 1, y_packed >> 1

        result_log = x_log - y_log
        result_sign = x_sign ^ y_sign
        result = (result_log << 1) | result_sign

        return result.to(torch.float64)

    @staticmethod
    def setup_context(ctx, inputs, outputs):
        x, y, base = inputs
        ctx.save_for_backward(x, y, base)

    @staticmethod
    def backward(ctx, grad_output):
        x, y, base = ctx.saved_tensors

        grad_x = apply_lns_op(torch.div, grad_output, y, base)
        grad_y = apply_lns_op(torch.square, y, base)
        grad_y = apply_lns_op(torch.div, x, grad_y, base)
        grad_y = apply_lns_op(torch.mul, grad_y, lnstensor(-1.0, b=base)._lns)
        grad_y = apply_lns_op(torch.mul, grad_output, grad_y)

        return grad_x, grad_y, None

@implements(torch.div, LNSDivFunction.forward, key='default', default=True)
def div(x, y, *, out=None):

    x, y = format_lnstensor_operands(x, y)
    result = LNSDivFunction.apply(x._lns, y._lns, x.base)

    if out is not None:
        out._lns = result
        out.base = x.base

    return lnstensor(result, from_lns=True, b=x.base)
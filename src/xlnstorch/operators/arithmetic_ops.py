import torch
from .. import LNS_ZERO, lnstensor, format_lnstensor_operands, implements
from . import (
    lns_add,
    lns_neg,
    lns_mul,
    lns_div,
    lns_square,
    lns_reciprocal,
)

# SBDB_FUNCS is a dictionary that contains different implementations
# of the sbdb (Gaussian logarithm) function. Each implementation is
# registered with a unique key.
SBDB_FUNCS = {}
DEFAULT_SBDB_FUNC = ""

def implement_sbdb(key, default=False):
    """
    A decorator to register a custom sbdb implementation. This will
    be used to compute/approximate the Gaussian logarithms for the
    addition and subtraction operations in the logarithmic domain. See

    https://en.wikipedia.org/wiki/Logarithmic_number_system
    https://en.wikipedia.org/wiki/Gaussian_logarithm

    Parameters
    ----------
    key : str
        The key to register the sbdb function under. This should be
        unique across all sbdb implementations.
    default : bool, optional
        If True, this sbdb function will be set as the default sbdb
        implementation. If multiple sbdb functions are registered
        with `default=True`, the last one registered will be used as
        the default. Defaults to False.

    Raises
    ------
    ValueError
        If an sbdb function with the given key is already registered.
    """
    def decorator(func):
        function_key = key or func.__name__

        if function_key in SBDB_FUNCS:
            raise ValueError(f"sbdb function with key '{function_key}' is already implemented.")
        SBDB_FUNCS[function_key] = func

        if default:
            global DEFAULT_SBDB_FUNC
            DEFAULT_SBDB_FUNC = function_key

        return func
    return decorator

def sbdb(z, s, base):
    """
    Computes the Gaussian logarithm for the given inputs z and s.

    Parameters
    ----------
    z : torch.Tensor
        The negation of the absolute difference between the two operands
        in the logarithmic domain.
    s : torch.Tensor
        The sign difference between the two operands in the logarithmic
        domain.
    base : torch.Tensor
        The base of the operands. Required for certain sbdb implementations.

    Returns
    -------
    torch.Tensor
        The result of the Gaussian logarithm computation.

    Raises
    ------
    ValueError
        If no default sbdb function is implemented.
    """
    if DEFAULT_SBDB_FUNC not in SBDB_FUNCS:
        raise ValueError(f"No default sbdb function implemented.")

    return SBDB_FUNCS[DEFAULT_SBDB_FUNC](z, s, base)

@implement_sbdb('ideal', default=True)
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

class LNSAddFunction(torch.autograd.Function):
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
        return result.to(torch.float64)

    @staticmethod
    def setup_context(ctx, inputs, output):
        pass # no context needed for this operation

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, grad_output, None

@implements(torch.add, LNSAddFunction.forward, key='default', default=True)
def add(x, y, *, alpha=1, out=None):

    x, y = format_lnstensor_operands(x, y)

    if alpha != 1:
        y = torch.mul(y, alpha)
    result = LNSAddFunction.apply(x._lns, y._lns, x.base)

    if out is not None:
        out._lns = result

    return lnstensor(result, from_lns=True, b=x.base)

class LNSSubFunction(torch.autograd.Function):
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
        pass # no context needed for this operation

    @staticmethod
    def backward(ctx, grad_output):
        grad_y = lns_neg(grad_output)
        return grad_output, grad_y, None

@implements(torch.sub, LNSSubFunction.forward, key="default", default=True)
def sub(x, y, *, alpha=1, out=None):

    x, y = format_lnstensor_operands(x, y)
    if alpha != 1:
        y = torch.mul(y, alpha)
    result = LNSSubFunction.apply(x._lns, y._lns, x.base)

    if out is not None:
        out._lns = result

    return lnstensor(result, from_lns=True, b=x.base)

class LNSNegFunction(torch.autograd.Function):
    """
    Negation becomes flipping the sign bit.

    Gradients are computed as follows:
    d/dx(-x) = -1
    """

    @staticmethod
    def forward(x):

        x_packed = x.to(torch.int64)
        neg_x_packed = x_packed ^ 1

        return neg_x_packed.to(torch.float64)
    
    @staticmethod
    def setup_context(ctx, inputs, output):
        pass # no context needed for this operation

    @staticmethod
    def backward(ctx, grad_output):
        return lns_neg(grad_output)

@implements(torch.neg, LNSNegFunction.forward, key="default", default=True)
def neg(x, *, out=None):

    result = LNSNegFunction.apply(x._lns)

    if out is not None:
        out._lns = result

    return lnstensor(result, from_lns=True, b=x.base)

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
    def setup_context(ctx, inputs, output):
        x, y = inputs
        ctx.save_for_backward(x, y)

    @staticmethod
    def backward(ctx, grad_output):
        x, y = ctx.saved_tensors

        grad_x = lns_mul(grad_output, y)
        grad_y = lns_mul(grad_output, x)

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
        return lns_mul(x, x)

    @staticmethod
    def setup_context(ctx, inputs, output):
        x, base = inputs
        ctx.save_for_backward(x, base)

    @staticmethod
    def backward(ctx, grad_output):
        x, base = ctx.saved_tensors

        grad_x = lns_mul(x, lnstensor(2.0, b=base)._lns)
        grad_x = lns_mul(grad_output, grad_x)

        return grad_x, None

@implements(torch.square, LNSSquareFunction.forward, key='default', default=True)
def square(x, *, out=None):

    result = LNSSquareFunction.apply(x._lns, x.base)

    if out is not None:
        out._lns = result

    return lnstensor(result, from_lns=True, b=x.base)

class LNSSqrtFunction(torch.autograd.Function):
    """
    Square rooting becomes halving in the logarithmic domain.

    Gradients are computed as follows:
    d/dx(sqrt(x)) = 1 / (2 * sqrt(x))
    """

    @staticmethod
    def forward(x, base):
        x_packed = x.to(torch.int64)
        result = ((x_packed & (-2)) / 2).to(torch.int64) & (-2)

        return result.to(torch.float64)

    @staticmethod
    def setup_context(ctx, inputs, output):
        _, base = inputs
        sqrt_x = output
        ctx.save_for_backward(sqrt_x, base)

    @staticmethod
    def backward(ctx, grad_output):
        sqrt_x, base = ctx.saved_tensors

        grad_x = lns_mul(sqrt_x, lnstensor(2.0, b=base)._lns)
        grad_x = lns_div(grad_output, grad_x, base)

        return grad_x, None

@implements(torch.sqrt, LNSSqrtFunction.forward, key='default', default=True)
def sqrt(x, *, out=None):

    result = LNSSqrtFunction.apply(x._lns, x.base)

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
            result = ((x_packed & (-2)) * n).to(torch.int64) & (-2)

        else:
            if n & 1 == 0:
                result = ((x_packed & (-2)) * n) & (-2)
            else:
                result = (((x_packed & (-2)) * n) & (-2)) | (x_packed & 1)

        return result.to(torch.float64)

    @staticmethod
    def setup_context(ctx, inputs, output):
        x, n, base = inputs
        ctx.save_for_backward(x, n, base)

    @staticmethod
    def backward(ctx, grad_output):
        pass # Placeholder until addition logic is implemented

@implements(torch.pow, LNSPowFunction.forward, key='default', default=True)
def pow(x, n, *, out=None):
    # todo: implement support for LNSTensor exponentiation
    result = LNSPowFunction.apply(x._lns, torch.tensor(n), x.base)

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
    def setup_context(ctx, inputs, output):
        x, y, base = inputs
        ctx.save_for_backward(x, y, base)

    @staticmethod
    def backward(ctx, grad_output):
        x, y, base = ctx.saved_tensors

        grad_x = lns_div(grad_output, y, base)
        grad_y = lns_square(y, base)
        grad_y = lns_div(x, grad_y, base)
        grad_y = lns_mul(grad_y, lnstensor(-1.0, b=base)._lns)
        grad_y = lns_mul(grad_output, grad_y)

        return grad_x, grad_y, None

@implements(torch.div, LNSDivFunction.forward, key='default', default=True)
def div(x, y, *, out=None):

    x, y = format_lnstensor_operands(x, y)
    result = LNSDivFunction.apply(x._lns, y._lns, x.base)

    if out is not None:
        out._lns = result
        out.base = x.base

    return lnstensor(result, from_lns=True, b=x.base)

class LNSReciprocalFunction(torch.autograd.Function):
    """
    See LNSDivFunction for details on the internal computation.

    Gradients are calculated as follows:
    d/dx(1 / x) = -1 / (x ^ 2)
    """

    @staticmethod
    def forward(x, base):
        return lns_div(lnstensor(1.0, b=base)._lns, x, base)

    @staticmethod
    def setup_context(ctx, inputs, output):
        x, base = inputs
        ctx.save_for_backward(x, base)

    @staticmethod
    def backward(ctx, grad_output):
        x, base = ctx.saved_tensors

        grad_x = lns_square(x, base)
        grad_x = lns_reciprocal(grad_x, base)
        grad_x = lns_mul(grad_x, lnstensor(-1.0, b=base)._lns)
        grad_x = lns_mul(grad_output, grad_x)

        return grad_x, None

@implements(torch.reciprocal, LNSReciprocalFunction.forward, key='default', default=True)
def reciprocal(x, *, out=None):

    result = LNSReciprocalFunction.apply(x._lns, x.base)

    if out is not None:
        out._lns = result

    return lnstensor(result, from_lns=True, b=x.base)

class LNSAbsFunction(torch.autograd.Function):
    """
    Absolute value becomes setting the sign bit off.

    Gradients are computed as follows:
    d/dx(|x|) = 1 if x > 0, -1 if x < 0 

    Note that PyTorch defines the gradient to be 0
    when x=0 despite it being undefined here.
    """

    @staticmethod
    def forward(x):
        x_packed = x.to(torch.int64)
        x_packed_abs = x_packed & (~1)

        return x_packed_abs.to(torch.float64)

    @staticmethod
    def setup_context(ctx, inputs, output):
        x, = inputs
        ctx.save_for_backward(x)

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        x_packed = x.to(torch.int64)
        x_packed_sign = x_packed & 1

        return torch.where(x_packed_sign == 1, lns_neg(grad_output), grad_output)

@implements(torch.abs, LNSAbsFunction.apply, "default", default=True)
def abs(x, *, out=None):

    result = LNSAbsFunction.apply(x._lns)

    if out is not None:
        out._lns = result

    return lnstensor(result, from_lns=True, b=x.base)

class LNSPositiveFunction(torch.autograd.Function):
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

@implements(torch.positive, LNSPositiveFunction.apply, "default", default=True)
def positive(x):

    result = LNSPositiveFunction.apply(x._lns)
    return lnstensor(result, from_lns=True, b=x.base)

class LNSSignFunction(torch.autograd.Function):
    """
    Sign becomes checking the sign bit (rightmost bit).

    Gradients are computed as follows:
    d/dx(sign(x)) = 0
    """

    @staticmethod
    def forward(x, base):
        x_packed = x.to(torch.int64)
        x_packed_sign = x_packed & 1

        # to check for zero case (return 0 lnstensor)
        return torch.where(x_packed_sign == 1, lnstensor(-1.0, b=base)._lns, lnstensor(1.0, b=base)._lns)

    @staticmethod
    def setup_context(ctx, inputs, output):
        pass # no context needed for this operation

    @staticmethod
    def backward(ctx, grad_output):
        pass # todo when zero is implemented

@implements(torch.sign, LNSSignFunction.forward, "default", default=True)
def sign(x, *, out=None):

    result = LNSSignFunction.apply(x._lns, x.base)

    if out is not None:
        out._lns = result

    return lnstensor(result, from_lns=True, b=x.base)
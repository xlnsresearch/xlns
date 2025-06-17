import torch
from .. import LNS_ZERO, LNSTensor, lnstensor, format_lnstensor_operands, implements, full_like
from . import (
    lns_add,
    lns_neg,
    lns_mul,
    lns_div,
    lns_square,
    lns_reciprocal,
    lns_pow,
    lns_matmul,
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
        return torch.where(
            torch.eq(x_packed | 1, LNS_ZERO), y, torch.where(
                torch.eq(y_packed | 1, LNS_ZERO), x, result.to(torch.float64)))

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
        result = (x_packed + y_packed - (y_packed & 1)) ^ (y_packed & 1)

        return torch.where(torch.eq(x_packed | 1, LNS_ZERO) | torch.eq(y_packed | 1, LNS_ZERO),
                             LNS_ZERO, result.to(torch.float64))

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

        grad_x = lns_mul(x, LNSTensor.get_internal_tensor(2.0, base))
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

        return torch.where(torch.eq(x_packed | 1, LNS_ZERO), LNS_ZERO, result.to(torch.float64))

    @staticmethod
    def setup_context(ctx, inputs, output):
        _, base = inputs
        sqrt_x = output
        ctx.save_for_backward(sqrt_x, base)

    @staticmethod
    def backward(ctx, grad_output):
        sqrt_x, base = ctx.saved_tensors

        grad_x = lns_mul(sqrt_x, LNSTensor.get_internal_tensor(2.0, base))
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
    Raising to a power becomes multiplication in the logarithmic domain.
    This function relies on the fact that the exponent is a floating
    point or integer which allows us to compute the power directly.

    Gradients are computed as follows:
    d/dx(x ^ n) = n * x ^ (n - 1)
    """

    @staticmethod
    def forward(x, n, base):
        x_packed = x.to(torch.int64)

        if torch.is_floating_point(n):
            result = ((x_packed & (-2)) * n).to(torch.int64) & (-2)

        else:
            abs_result = ((x_packed & (-2)) * n) & (-2)
            result = torch.where(n & 1 == 0, abs_result, abs_result | (x_packed & 1))

        return result.to(torch.float64)

    @staticmethod
    def setup_context(ctx, inputs, output):
        x, n, base = inputs
        ctx.save_for_backward(x, n, base)

    @staticmethod
    def backward(ctx, grad_output):
        x, n, base = ctx.saved_tensors

        grad_x = lns_pow(x, n - 1, base)
        grad_x = lns_mul(grad_x, LNSTensor.get_internal_tensor(n, base))
        grad_x = lns_mul(grad_output, grad_x)

        return grad_x, None, None

@implements(torch.pow, LNSPowFunction.forward, key='default', default=True)
def pow(x, n, *, out=None):

    if isinstance(x, LNSTensor) and not isinstance(n, LNSTensor):
        if not isinstance(n, torch.Tensor):
            dtype = torch.int64 if (isinstance(n, int) or isinstance(n, float) and n.is_integer()) else torch.float64
            n = torch.tensor(n, dtype=dtype)
        x._lns, n = torch.broadcast_tensors(x._lns, n)
        result = LNSPowFunction.apply(x._lns, n, x.base)

    else:
        x, n = format_lnstensor_operands(x, n)
        x._lns, n._lns = torch.broadcast_tensors(x._lns, n._lns)
        result = LNSPowFunction.apply(x._lns, n.value, x.base)

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
        result = (x_packed - y_packed + (y_packed & 1)) ^ (y_packed & 1)

        return torch.where(torch.eq(x_packed | 1, LNS_ZERO), LNS_ZERO, result.to(torch.float64))

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
        grad_y = lns_mul(grad_y, LNSTensor.get_internal_tensor(-1.0, base))
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
        return lns_div(LNSTensor.get_internal_tensor(1.0, base), x, base)

    @staticmethod
    def setup_context(ctx, inputs, output):
        x, base = inputs
        ctx.save_for_backward(x, base)

    @staticmethod
    def backward(ctx, grad_output):
        x, base = ctx.saved_tensors

        grad_x = lns_square(x, base)
        grad_x = lns_reciprocal(grad_x, base)
        grad_x = lns_mul(grad_x, LNSTensor.get_internal_tensor(-1.0, base))
        grad_x = lns_mul(grad_output, grad_x)

        return grad_x, None

@implements(torch.reciprocal, LNSReciprocalFunction.forward, key='default', default=True)
def reciprocal(x, *, out=None):

    result = LNSReciprocalFunction.apply(x._lns, x.base)

    if out is not None:
        out._lns = result

    return lnstensor(result, from_lns=True, b=x.base)

class LNSExpFunction(torch.autograd.Function):
    """
    Exponentiation in the logarithmic domain requires us to
    convert the input to its floating point representation
    and then compute raising to the power of e.

    Gradients are computed as follows:
    d/dx(e ^ x) = e ^ x
    """

    @staticmethod
    def forward(x, base):
        e = full_like(x, torch.exp(torch.tensor(1.0)), b=base)
        return lns_pow(e._lns, lnstensor(x, from_lns=True, b=base).value, base)

    @staticmethod
    def setup_context(ctx, inputs, output):
        ctx.save_for_backward(output)

    @staticmethod
    def backward(ctx, grad_output):
        exp_x, = ctx.saved_tensors
        return lns_mul(grad_output, exp_x), None

@implements(torch.exp, LNSExpFunction.forward, key='default', default=True)
def exp(x, *, out=None):

    result = LNSExpFunction.apply(x._lns, x.base)

    if out is not None:
        out._lns = result

    return lnstensor(result, from_lns=True, b=x.base)

class LNSLogFunction(torch.autograd.Function):
    """
    Taking the logarithm in the logarithmic domain requires us to
    convert the input to its floating point representation and then
    compute the logarithm with respect to the base.

    Gradients are computed as follows:
    d/dx(log(x)) = 1 / x
    """

    @staticmethod
    def forward(x, base):
        x_log = torch.log(lnstensor(x, from_lns=True, b=base).value)
        return lnstensor(x_log, b=base)._lns

    @staticmethod
    def setup_context(ctx, inputs, output):
        x, base = inputs
        ctx.save_for_backward(x, base)

    @staticmethod
    def backward(ctx, grad_output):
        x, base = ctx.saved_tensors

        grad_x = lns_div(grad_output, x, base)
        return grad_x, None

@implements(torch.log, LNSLogFunction.forward, key='default', default=True)
def log(x, *, out=None):

    result = LNSLogFunction.apply(x._lns, x.base)

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

        return torch.where(torch.eq(x_packed | 1, LNS_ZERO), LNS_ZERO, x_packed_abs.to(torch.float64))

    @staticmethod
    def setup_context(ctx, inputs, output):
        x, = inputs
        ctx.save_for_backward(x)

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        x_packed = x.to(torch.int64)
        x_packed_sign = x_packed & 1

        return torch.where(torch.eq(x_packed_sign, 1), lns_neg(grad_output), grad_output)

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

        return torch.where(
            torch.eq(x_packed | 1, LNS_ZERO), LNS_ZERO,
            torch.where(x_packed_sign == 1,
                        LNSTensor.get_internal_tensor(-1.0, base), LNSTensor.get_internal_tensor(1.0, base)))

    @staticmethod
    def setup_context(ctx, inputs, output):
        pass # no context needed for this operation

    @staticmethod
    def backward(ctx, grad_output):
        return LNS_ZERO, None

@implements(torch.sign, LNSSignFunction.forward, "default", default=True)
def sign(x, *, out=None):

    result = LNSSignFunction.apply(x._lns, x.base)

    if out is not None:
        out._lns = result

    return lnstensor(result, from_lns=True, b=x.base)

class LNSSumFunction(torch.autograd.Function):
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
        x, base, _, _ = inputs
        ctx.save_for_backward(x)
        ctx.base = base

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        return torch.full_like(x, LNSTensor.get_internal_tensor(1.0, ctx.base).item()), None, None, None

@implements(torch.sum, LNSSumFunction.forward, "default", default=True)
def sum(x, dim=None, keepdim=False, *, out=None):

    result = LNSSumFunction.apply(x._lns, x.base, dim, keepdim)

    if out is not None:
        out._lns = result

    return lnstensor(result, from_lns=True, b=x.base)

class LNSMatmulFunction(torch.autograd.Function):
    """
    Matrix multiplication uses the lns addition and
    multiplication functions to compute the result.

    Gradients are computed as follows:
    d/dA(A @ B) = B^T
    d/dB(A @ B) = A^T
    """

    @staticmethod
    def forward(A, B, base):
        M, K = A.shape[-2:]
        N = B.shape[-1]

        result = torch.full((*A.shape[:-2], M, N), fill_value=LNS_ZERO,
                            dtype=torch.float64, device=A.device)

        for k in range(K):
            term = lns_mul(A[..., :, k].unsqueeze(-1), B[..., k, :].unsqueeze(-2))
            result = lns_add(result, term, base)

        return result

    @staticmethod
    def setup_context(ctx, inputs, output):
        A, B, base = inputs
        ctx.save_for_backward(A, B, base)

    @staticmethod
    def backward(ctx, grad_output):
        A, B, base = ctx.saved_tensors

        grad_A = lns_matmul(grad_output, B.transpose(-1, -2), base)
        grad_B = lns_matmul(A.transpose(-1, -2), grad_output, base)

        return grad_A, grad_B, None

@implements(torch.matmul, LNSMatmulFunction.forward, "default", default=True)
def matmul(A, B, *, out=None):

    A, B = format_lnstensor_operands(A, B)
    result = LNSMatmulFunction.apply(A._lns, B._lns, A.base)

    if out is not None:
        out._lns = result

    return lnstensor(result, from_lns=True, b=A.base)

class LNSTransposeFunction(torch.autograd.Function):
    """
    Transpose operation simply rearranges the dimensions
    of the input tensor. It doesn't change the underlying
    representations, so the forward pass isn't special.

    Gradients are computed as follows:
    d/dx(A.T) = 1
    """

    @staticmethod
    def forward(A, dim0, dim1):
        return torch.transpose(A, dim0, dim1)

    @staticmethod
    def setup_context(ctx, inputs, output):
        A, _, _ = inputs
        ctx.save_for_backward(A)

    @staticmethod
    def backward(ctx, grad_output):
        A, = ctx.saved_tensors
        return torch.full_like(A, LNSTensor.get_internal_tensor(1.0, ctx.base).item()), None, None

@implements(torch.transpose, LNSTransposeFunction.forward, "default", default=True)
def transpose(A, dim0, dim1):

    result = LNSTransposeFunction.apply(A._lns, dim0, dim1)
    return lnstensor(result, from_lns=True, b=A.base)
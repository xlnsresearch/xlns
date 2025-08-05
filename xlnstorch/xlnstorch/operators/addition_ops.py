import torch
import contextlib
from typing import Generator, Callable
from xlnstorch import LNS_ZERO, CSRC_AVAILABLE, lnstensor, format_lnstensor_operands, implements
from xlnstorch.autograd import LNSFunction
from xlnstorch.tensor_utils import get_precision_from_base
from . import (
    lns_add,
    lns_neg,
)

# SBDB_FUNCS is a dictionary that contains different implementations
# of the sbdb (Gaussian logarithm) function. Each implementation is
# registered with a unique key.
SBDB_FUNCS = {}
DEFAULT_SBDB_FUNC = ""

def set_default_sbdb_implementation(impl_key: str) -> None:
    """
    Set the default implementation for the sbdb function.

    Parameters
    ----------
    impl_key : str
        The key identifying the implementation to be set as default.

    Raises
    ------
    ValueError
        If the specified implementation key is not registered for the sbdb function.
    """
    if impl_key not in SBDB_FUNCS:
        raise ValueError(f"Implementation '{impl_key}' is not registered for the sbdb function.")

    global DEFAULT_SBDB_FUNC
    DEFAULT_SBDB_FUNC = impl_key

    if CSRC_AVAILABLE:
        import xlnstorch.csrc
        xlnstorch.csrc.set_default_sbdb_implementation(impl_key)

@contextlib.contextmanager
def override_sbdb_implementation(impl_key: str) -> Generator[None, None, None]:
    """
    Temporarily override the default sbdb implementation within a context. This
    allows for testing or using a different implementation without permanently
    changing the default.

    Parameters
    ----------
    impl_key : str
        The key identifying the new implementation to use as default.

    Yields
    ------
    None
        The function yields control back to the context block.
    """
    global DEFAULT_SBDB_FUNC
    original_default = DEFAULT_SBDB_FUNC
    set_default_sbdb_implementation(impl_key)

    try:
        yield
    finally:
        DEFAULT_SBDB_FUNC = original_default

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

def register_xlnsconf_implementation(xlns_function: Callable, impl_key: str) -> None:
    """
    """
    if impl_key in SBDB_FUNCS:
        raise ValueError(f"Implementation '{impl_key}' is already registered for the sbdb function.")

    def wrapper_sbdb(z, s, base):
        precision = get_precision_from_base(base)
        z_np = z.numpy()
        s_np = s.numpy()

        xlns_result = xlns_function(z_np, s_np, B=base.item(), F=precision)
        return torch.tensor(xlns_result, dtype=torch.int64)

    SBDB_FUNCS[impl_key] = wrapper_sbdb

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
        pass # no context needed for this operation

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, grad_output, None

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

    result = LNSSubFunction.apply(x, y, x.base)

    if out is not None:
        return out._inplace_copy(result)

    return lnstensor(result, from_lns=True, b=x.base)

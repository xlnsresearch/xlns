import contextlib
from typing import Generator, Callable
import torch
import xlnstorch
import xlnstorch.csrc
from xlnstorch.tensor_utils import get_precision_from_base

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

    if xlnstorch.CSRC_AVAILABLE:
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

def implements_sbdb(key, default=False):
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
    Registers an implementation of the sbdb function using a ufunc from the xlnsconf
    packed. This allows for experimentation with implementations from xlnsconf that
    haven't been ported to xlnstorch yet.

    Parameters
    ----------
    xlns_function : Callable
        The xlnsconf ufunc that implements the sbdb function.
    impl_key : str
        The key to register the sbdb function under. This should be unique
        across all sbdb implementations.

    Raises
    ------
    ValueError
        If an sbdb function with the given key is already registered.
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
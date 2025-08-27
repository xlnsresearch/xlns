import functools
import contextlib
from dataclasses import dataclass
from typing import Callable, Generator, Tuple, Optional

@dataclass
class Implementation:
    """
    A dataclass to represent an implementation of a torch function in the LNS context.

    Attributes
    ----------
    func : Callable
        The function that implements the LNS operation.
    lns_op : Callable
        The internal computation function for the LNS operation. This takes
        internal representations as arguments rather than LNSTensor objects.
    """
    func: Callable
    lns_op: Callable

# _HANDLED_FUNCTIONS is a dictionary that maps torch functions to their
# corresponding implementations for LNSTensor. Each key is a torch function
# and the value is a dictionary mapping implementation keys to an Implementation
# object, which contains the function that implements the LNS operation, the
# internal computation function done on the internal torch tensor representations.
_HANDLED_FUNCTIONS = {}
# _DEFAULT_IMPLEMENTATIONS is a dictionary that maps torch functions to their
# default implementation keys. This is used to determine which implementation
# to use by default when a torch function is called on an LNSTensor.
_DEFAULT_IMPLEMENTATIONS = {}

def implements(
        torch_function: Callable,
        lns_operation: Callable,
        key: Optional[str] = None,
        default: bool = False,
    ) -> Callable:
    """
    A decorator to register a custom implementation for a given torch function.

    This allows functions to be mapped to specific handlers in the LNS context
    and optionally set as the default implementation for that function.

    Parameters
    ----------
    torch_function : Callable
        The torch function to be overriden.
    lns_operation : Callable
        The function that implements the given LNS operation to tensors. This
        should be the computation that is performed on the internal torch tensor
        representations for the LNSTensor objects.
    key : str, optional
        A unique key to identify the implementation. If not provided, the 
        function's name will be used by default.
    default : bool, optional
        If True, this implementation will be set as the default for the
        specified torch function. Defaults to False.

    Returns
    -------
    Callable
        The decorator that registers the function as an implementation for
        the specific torch function.
    """
    def decorator(func):
        function_key = key or func.__name__
        functools.update_wrapper(func, torch_function)

        if torch_function not in _HANDLED_FUNCTIONS:
            _HANDLED_FUNCTIONS[torch_function] = {}

        implementation = Implementation(
            func=func,
            lns_op=lns_operation,
        )
        _HANDLED_FUNCTIONS[torch_function][function_key] = implementation

        if default:
            _DEFAULT_IMPLEMENTATIONS[torch_function] = function_key

        return func
    return decorator

def get_implementation(torch_function: Callable, impl_key: str) -> Implementation:
    """
    Get the implementation tuple for a given torch function and implementation key.

    Parameters
    ----------
    torch_function : Callable
        The torch function for which to get the implementation.
    impl_key : str
        The key identifying the specific implementation.

    Returns
    -------
    Implementation
        An implementation object containing the function that implements the LNS operation,
        and the internal computation function.

    Raises
    ------
    ValueError
        If no implementations are registered for the given torch function.
        If the specified implementation key is not registered for the torch function.
    """
    if torch_function not in _HANDLED_FUNCTIONS:
        raise ValueError(f"No implementations registered for {torch_function.__name__}.")

    if impl_key not in _HANDLED_FUNCTIONS[torch_function]:
        raise ValueError(f"Implementation '{impl_key}' is not registered for {torch_function.__name__}.")

    return _HANDLED_FUNCTIONS[torch_function][impl_key]

def set_default_implementation(torch_function: Callable, impl_key: str) -> None:
    """
    Set the default implementation for a given torch function.

    Parameters
    ----------
    torch_function : Callable
        The torch function for which to set the default implementation.
    impl_key : str
        The key identifying the implementation to be set as default.

    Raises
    ------
    ValueError
        If no implementations are registered for the given torch function.
        If the specified implementation key is not registered for the torch function.
    """
    if torch_function not in _HANDLED_FUNCTIONS:
        raise ValueError(f"No implementations registered for {torch_function.__name__}.")

    if impl_key not in _HANDLED_FUNCTIONS[torch_function]:
        raise ValueError(f"Implementation '{impl_key}' is not registered for {torch_function.__name__}.")

    _DEFAULT_IMPLEMENTATIONS[torch_function] = impl_key

def get_default_implementation_key(torch_function: Callable) -> str:
    """
    Get the default implementation key for a given torch function.

    Parameters
    ----------
    torch_function : Callable
        The torch function for which to get the default implementation.

    Returns
    -------
    str
        The key identifying the default implementation.

    Raises
    ------
    ValueError
        If no implementations are registered for the given torch function.
        If no default implementation is set for the torch function.
    """
    if torch_function not in _HANDLED_FUNCTIONS:
        raise ValueError(f"No implementations registered for {torch_function.__name__}.")

    if torch_function not in _DEFAULT_IMPLEMENTATIONS:
        raise ValueError(f"No default implementation set for {torch_function.__name__}.")

    return _DEFAULT_IMPLEMENTATIONS[torch_function]

@contextlib.contextmanager
def override_implementation(torch_function: Callable, impl_key: str) -> Generator[None, None, None]:
    """
    Temporarily override the default implementation for a torch function within a context.
    This allows for testing or using a different implementation without permanently changing
    the default.

    Parameters
    ----------
    torch_function : Callable
        The torch function for which the implementation is to be temporarily overridden.
    impl_key : str
        The key identifying the new implementation to use as default.

    Yields
    ------
    None
        The function yields control back to the context block.

    Examples
    --------
    >>> with override_implementation(torch.add, 'custom_add_impl'):
    >>>     # Inside this block, torch.add will use 'custom_add_impl'
    >>>     pass
    """
    original_default = _DEFAULT_IMPLEMENTATIONS.get(torch_function)
    set_default_implementation(torch_function, impl_key)

    try:
        yield
    finally:
        set_default_implementation(torch_function, original_default)

def apply_lns_op(torch_function: Callable, *args, **kwargs):
    """
    Performs the computation for the default LNS implementation of a given
    torch function in the logarithmic domain. This function is used to
    apply the LNS internal operation defined in ``HANDLED_FUNCTIONS``.

    Parameters
    ----------
    torch_function : Callable
        The torch function for which the LNS operation is to be applied.
    *args : Any
        Positional arguments to be passed to the LNS operation.
    **kwargs : Any
        Keyword arguments to be passed to the LNS operation.

    Returns
    -------
    Any
        The result of the LNS operation applied to the provided arguments.

    Raises
    ------
    ValueError
        If no implementations are registered for the given torch function.
        If no default implementation is set for the torch function.
        If the implementation key is not registered for the torch function.
    """
    impl_key = get_default_implementation_key(torch_function)
    impl = get_implementation(torch_function, impl_key)

    return impl.lns_op(*args, **kwargs) # internal computation function
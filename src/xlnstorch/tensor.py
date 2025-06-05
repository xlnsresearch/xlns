from __future__ import annotations
from typing import Any, Union, Callable, Generator
import functools
import contextlib

import numpy as np
import torch
from torch import Tensor
import xlns as xl

# HANDLED_FUNCTIONS is a dictionary that maps torch functions to their
# corresponding implementations for LNSTensor. Each key is a torch function
# and the value is a dictionary mapping implementation keys to a tuple of
# the LNSTensor implementation and its internal computation function.
HANDLED_FUNCTIONS = {}
# DEFAULT_IMPLEMENTATIONS is a dictionary that maps torch functions to their
# default implementation keys. This is used to determine which implementation
# to use by default when a torch function is called on an LNSTensor.
DEFAULT_IMPLEMENTATIONS = {}

def implements(
        torch_function: Callable,
        lns_operation: Callable,
        key: str | None = None,
        default: bool = False
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

        if torch_function not in HANDLED_FUNCTIONS:
            HANDLED_FUNCTIONS[torch_function] = {}
        HANDLED_FUNCTIONS[torch_function][function_key] = (func, lns_operation)

        if default:
            DEFAULT_IMPLEMENTATIONS[torch_function] = function_key

        return func
    return decorator

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
    if torch_function not in HANDLED_FUNCTIONS:
        raise ValueError("No implementations registered for the given torch function.")

    if impl_key not in HANDLED_FUNCTIONS[torch_function]:
        raise ValueError(f"Implementation '{impl_key}' is not registered for {torch_function}.")

    DEFAULT_IMPLEMENTATIONS[torch_function] = impl_key

@contextlib.contextmanager
def override_impl(torch_function: Callable, impl_key: str) -> Generator[None, None, None]:
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
    >>> with override_impl(torch.add, 'custom_add_impl'):
    >>>     # Inside this block, torch.add will use 'custom_add_impl'
    >>>     pass
    """
    original_default = DEFAULT_IMPLEMENTATIONS.get(torch_function)
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
    if torch_function not in HANDLED_FUNCTIONS:
        raise ValueError("No implementations registered for the given torch function.")

    impl_key = DEFAULT_IMPLEMENTATIONS.get(torch_function)
    if impl_key is None:
        raise ValueError(f"No default implementation set for {torch_function}.")

    if impl_key not in HANDLED_FUNCTIONS[torch_function]:
        raise ValueError(f"Implementation key '{impl_key}' is not registered for {torch_function}.")

    return HANDLED_FUNCTIONS[torch_function][impl_key][1](*args, **kwargs)

class LNSTensor:
    r"""
    A logarithmic number system (LNS) wrapper for PyTorch tensors. 

    Internally each real number :math:`x` is stored as one packed integer

    .. math::

        \begin{align*}
            \text{sign_bit} &= \begin{cases}
                0 & x \ge 0 \\
                1 & x <  0
            \end{cases} \\
            \text{exponent} &= \mathrm{round} \left(
                \frac{\ln(|x|)}{\ln(\text{base})}
            \right) \\
            \text{packed} &= (\text{exponent} \ll 1) \mathbin{|} \text{sign_bit}
        \end{align*}

    The packed integers live in :attr:`_lns` (stored as a ``torch.float64``
    Tensor so that gradients can be retained for autograd).

    Parameters
    ----------
    data : torch.Tensor
        *Real-valued* tensor to encode **or** a pre-packed LNS tensor when
        ``from_lns`` is ``True``. Must have dtype ``float64``.
    base : torch.Tensor
        Scalar tensor that holds the logarithm base.
        Must be positive and **not** equal to 1.
        Shape must be ``()`` and dtype must be ``float64``.
    from_lns : bool, optional
        If ``True`` interpret *data* as already packed.
        Defaults to ``False``.

    Notes
    -----
    The use of the ``xlnstorch.Tensor`` constructor is discouraged. Use
    ``xlnstorch.lnstensor()`` instead.
    """

    def __init__(
            self,
            data: Tensor,
            base: Tensor,
            from_lns: bool = False
            ):

        if not torch.is_tensor(data) or data.dtype != torch.float64:
            raise TypeError("`data` must be a torch.Tensor with dtype=float64.")

        if (
            not torch.is_tensor(base)
            or base.dtype != torch.float64
            or base.shape != torch.Size([])
        ):
            raise TypeError("`base` must be a scalar torch.Tensor with dtype=float64.")

        if base <= 0 or torch.eq(base, 1):
            raise ValueError("`base` must be positive and not equal to 1.")

        self.base: Tensor = base.clone()

        if from_lns:
            packed = data.clone()
        else:
            with torch.no_grad():
                log_base = torch.log(self.base)
                log_data = torch.log(torch.abs(data)) / log_base
                exponent = log_data.round().to(torch.int64)

                sign_bit = (data < 0).to(torch.int64)
                packed_int = (exponent << 1) | sign_bit
                packed = packed_int.to(torch.float64)

        self._lns: Tensor = packed
        self._lns.requires_grad_(True)

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        """
        Overrides default torch function behavior for LNSTensor objects.
        This allows us to define custom LNS operators. See

        https://docs.pytorch.org/docs/stable/notes/extending.html#extending-torch-python-api

        for more details on how this works.
        """
        if kwargs is None:
            kwargs = {}

        if func not in HANDLED_FUNCTIONS or not all(issubclass(t, LNSTensor) for t in types):
            return NotImplemented

        chosen_impl = DEFAULT_IMPLEMENTATIONS.get(func)
        if chosen_impl is None:
            raise RuntimeError(f"No default implementation has been set for {func}.")

        if chosen_impl not in HANDLED_FUNCTIONS[func]:
            raise ValueError(f"Implementation key '{chosen_impl}' is not registered for {func}.")

        return HANDLED_FUNCTIONS[func][chosen_impl][0](*args, **kwargs)

    def backward(self, gradient=None, retain_graph=None, create_graph=False, inputs=None):
        """
        Computes the gradients of the current LNSTensor with respect to the graph leaves.
        This method is analogous to the standard PyTorch `backward` method, but works with
        LNSTensor objects. See

        https://docs.pytorch.org/docs/stable/generated/torch.Tensor.backward.html

        for more details on the parameters. Note that the `gradient` and `inputs` parameters
        here are LNSTensor objects, not regular PyTorch tensors.

        Parameters
        ----------
        gradient : LNSTensor, optional
            Gradient of the function being differentiated w.r.t. `self`. This argument should
            be omitted if `self` is a scalar. In this case, the gradient is set to 1.
        retain_graph : bool, optional
            If True, the graph used to compute the gradient will be retained, allowing for 
            further backward passes, by default None.
        create_graph : bool, optional
            If True, the graph of the derivative will be constructed, allowing to compute higher 
            order derivative products, by default False.
        inputs : sequence of LNSTensor, optional
            Inputs with respect to which the gradient will be accumulated into their `.grad` 
            attributes, all other tensors will be ignored. If not provided, the gradient is 
            accumulated into all the leaf Tensors that were used to compute the tensors.
        """
        if gradient is None:
            tensor_gradient = torch.zeros_like(self._lns, dtype=torch.float64)
        else:
            tensor_gradient = gradient.lns

        if inputs is None:
            tensor_inputs = None
        else:
            tensor_inputs = [inp.lns for inp in inputs]

        self._lns.backward(
            tensor_gradient,
            retain_graph=retain_graph,
            create_graph=create_graph,
            inputs=tensor_inputs
        )

    @property
    def lns(self) -> Tensor:
        """
        The packed representation that **does** carry gradients.

        Returns
        -------
        torch.Tensor
            Tensor of dtype ``float64`` holding the packed integers.
        """
        return self._lns

    @property
    def value(self) -> Tensor:
        """
        Decode the packed integers back to ordinary floating-point numbers.

        Returns
        -------
        torch.Tensor
            Real-valued tensor (dtype ``float64``).
        """
        packed_int = self._lns.to(torch.int64)

        exponent = (packed_int >> 1).to(torch.float64)
        sign = torch.where((packed_int & 1).bool(), -1.0, 1.0)

        return sign * torch.pow(self.base, exponent)

    @property
    def grad(self) -> LNSTensor:
        """
        The gradient of the LNSTensor, if it has been computed.
        
        Returns
        -------
        LNSTensor or None
            The gradient as an LNSTensor computed by PyTorch's autograd.
        """
        if self._lns.grad is None:
            return None
        
        return lnstensor(self._lns.grad, from_lns=True, b=self.base)

    def __repr__(self) -> str:
        return f"LNSTensor(value={self.value}, base={self.base.item()})"

def lnstensor(
        data: Any,
        from_lns: bool = False,
        f: int | None = None,
        b: Union[float, int, Tensor, None] = None
        ) -> LNSTensor:
    r"""
    Constructs an :class:`LNSTensor` from some array-like *data*.

    The function accepts ordinary numeric data (tensors, NumPy arrays,
    scalars) **and** every non-redundant *xlns* type.  Redundant formats
    (``xlnsr`` and ``xlnsnpr``) are **not** supported.

    The LNSTensor ``base`` is chosen in the following order:

    1. If ``f`` is given, ``base`` = 2.0 ^ (2 ^ -f).
    2. Else if ``b is given``, use ``b`` (float *or* scalar tensor).
    3. Else, default to ``xlns.xlnsB`` (global constant).

    Parameters
    ----------
    data : LNSTensor, torch.Tensor, numpy.ndarray, numbers, xlns types
        - A real-valued tensor/array/scalar to *encode* **or**
        - A pre-packed representation (when ``from_lns`` is ``True``) **or**
        - An existing :class:`LNSTensor` (which will be copied or converted base).
    from_lns : bool, optional
        If ``True``, treat *data* as already packed. Defaults to ``False``.
    f : int, optional
        The number of fractional exponent bits. mutually exclusive with ``b``.
    b : float, int, torch.Tensor, optional
        The explicit logarithm base; mutually exclusive with ``f``.

    Returns
    -------
    LNSTensor
        The constructed LNSTensor.

    Raises
    ------
    ValueError
        If both ``f`` and ``b`` are provided, or if neither can be resolved
        to a valid base.
    TypeError
        If *data* is of an unsupported type (i.e. not array-like).
    """
    # 1. Determine the logarithm base
    if f is not None and b is not None:
        raise ValueError("Cannot specify both `f` and `b`.")
    if f is not None:
        base_val: float = 2.0 ** (2 ** (-f))
    elif b is not None:
        base_val = float(b) if isinstance(b, Tensor) else b
    else:
        base_val = xl.xlnsB

    if base_val < 0 or base_val == 1:
        raise ValueError("base must be positive and not equal to 1")
    base_tensor: Tensor = torch.tensor(base_val, dtype=torch.float64)

    # 2. Convert data to a float64 tensor
    # Some branchs switch from_lns to True if the data is already packed.

    # xlnstorch.LNSTensor
    if isinstance(data, LNSTensor):
        input_data = data.lns.clone()
        from_lns = True

        if not torch.equal(data.base, base_tensor):
            with torch.no_grad():
                packed_int = input_data.to(torch.int64)
                sign_bit = packed_int & 1
                exponent = (packed_int >> 1).to(torch.float64)

                exponent_new = exponent * torch.log(data.base) / torch.log(base_tensor)
                new_packed_int = (exponent_new.round().to(torch.int64) << 1) | sign_bit
                input_data = new_packed_int.to(torch.float64)

    # torch.Tensor
    elif isinstance(data, torch.Tensor):
        input_data = data.to(torch.float64)

    # numpy.ndarray
    elif isinstance(data, np.ndarray):
        input_data = torch.from_numpy(data).to(torch.float64)

    # xlns scalar objects
    elif isinstance(data, (xl.xlns, xl.xlnsud, xl.xlnsv, xl.xlnsb)):
        if isinstance(data, (xl.xlns, xl.xlnsud)):
            log_part = data.x * np.log(xl.xlnsB) / np.log(base_val)
        elif isinstance(data, (xl.xlnsb, xl.xlnsv)):
            log_part = data.x * np.log(data.B) / np.log(base_val)
        else:
            log_part = data.x

        packed_int = (int(round(log_part)) << 1) | data.s
        input_data = torch.tensor(packed_int, dtype=torch.float64)
        from_lns = True

    # xlns numpy-like arrays
    elif isinstance(data, (xl.xlnsnp, xl.xlnsnpv, xl.xlnsnpb)):
        data_x = data.nd >> 1
        data_s = data.nd & 1

        if isinstance(data, (xl.xlnsnp, xl.xlnsnpv)) and base_val != xl.xlnsB:
            log_part = data_x * np.log(xl.xlnsB) / np.log(base_val)
        elif isinstance(data, xl.xlnsnpb) and base_val != data.B:
            log_part = data_x * np.log(data.B) / np.log(base_val)
        else:
            log_part = data_x

        packed_int = (np.int64(np.round(log_part)) << 1) | data_s
        input_data = torch.tensor(packed_int, dtype=torch.float64)
        from_lns = True

    # Everything else (scalars, lists, tuples, etc.)
    else:
        try:
            input_data = torch.tensor(data, dtype=torch.float64)
        except Exception as e:
            raise TypeError(f"Unsupported data type for LNSTensor: {type(data).__name__}") from e

    return LNSTensor(input_data, base_tensor, from_lns=from_lns)
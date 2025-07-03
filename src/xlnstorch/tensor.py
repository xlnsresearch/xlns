from __future__ import annotations
from typing import Any, Union, List

import math
import numpy as np
import torch
from torch import Tensor
import xlns as xl
from . import LNS_ZERO, get_default_implementation_key, get_implementation

_xlns_types = (xl.xlns, xl.xlnsud, xl.xlnsv, xl.xlnsb, xl.xlnsnp, xl.xlnsnpv, xl.xlnsnpb)

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
    requires_grad : bool, optional
        If ``True``, the LNSTensor will track gradients.

    Notes
    -----
    The use of the ``xlnstorch.Tensor`` constructor is discouraged. Use
    ``xlnstorch.lnstensor()`` instead.
    """

    def __init__(
            self,
            data: Tensor,
            base: Tensor,
            from_lns: bool = False,
            requires_grad: bool = False,
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
            packed = data.to(torch.float64)
        else:
            with torch.no_grad():
                log_base = torch.log(self.base)
                log_data = torch.log(torch.abs(data)) / log_base
                exponent = log_data.round().to(torch.int64)

                sign_bit = (data < 0).to(torch.int64)
                packed_int = (exponent << 1) | sign_bit
                packed = packed_int.to(torch.float64)
                packed = torch.where(torch.eq(data, 0), LNS_ZERO, packed)

        self._lns: Tensor = packed
        self._lns.requires_grad_(requires_grad)

        if requires_grad and self._lns.is_leaf and not hasattr(self._lns, "_incoming_grads"):
            self.register_grad_hooks()

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

        if not all(issubclass(t, LNSTensor) or issubclass(t, Tensor) for t in types):
            return NotImplemented

        impl_key = get_default_implementation_key(func)
        impl = get_implementation(func, impl_key)

        return impl[0](*args, **kwargs) # LNSTensor custom operator

    def register_grad_hooks(self):

        self._lns._incoming_grads = []

        def _hook(grad):
            self._lns._incoming_grads.append(grad.clone())
            return grad

        def _accum_hook(param):
            accum_grad = lnstensor(0, from_lns=False, b=self.base)
            for grad in self._lns._incoming_grads:
                if grad is not None:
                    accum_grad += lnstensor(grad, from_lns=True, b=self.base)
            param.grad = accum_grad._lns

        self._hook_handle = self._lns.register_hook(_hook)
        self._accum_hook_handle = self._lns.register_post_accumulate_grad_hook(_accum_hook)    

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
            # if self._lns.numel() != 1:
            #     raise RuntimeError("grad can be implicitly created only for scalar outputs")
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

    @staticmethod
    def get_internal_tensor(fp_value: Any, base: Tensor) -> Tensor:
        """
        Converts an array-like floating point value to an LNSTensor and
        returns the internal packed representation for a given base.

        Parameters
        ----------
        fp_value : Any
            Array-like floating point value to convert to an LNSTensor.
            For possible types see the ``data`` parameter of :func:`lnstensor`
        base : torch.Tensor
            Scalar tensor that holds the desired logarithm base.

        Returns
        -------
        torch.Tensor
            The internal packed representation of the LNSTensor for a
            requested floating point value and base.
        """
        return lnstensor(fp_value, b=base)._lns

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

        return torch.where(torch.eq(packed_int | 1, LNS_ZERO), 0.0, sign * torch.pow(self.base, exponent))

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

    @property
    def shape(self) -> torch.Size:
        """
        Returns the shape of the LNSTensor.

        Returns
        -------
        torch.Size
            The shape of the LNSTensor.
        """
        return self._lns.shape

    @property
    def ndim(self) -> int:
        """
        Alias for :meth:`dim`.
        """
        return self.dim()

    @property
    def requires_grad(self) -> bool:
        """
        Returns whether the LNSTensor requires gradients.

        Returns
        -------
        bool
            True if the LNSTensor requires gradients, False otherwise.
        """
        return self._lns.requires_grad

    def view(self, *shape: int) -> LNSTensor:
        """
        Returns a new tensor with the same data as this LNSTensor
        but with a different shape.
        """
        return lnstensor(self._lns.view(*shape), from_lns=True, b=self.base)

    def item(self) -> float:
        """
        Returns the value of the LNSTensor as a Python number. This method
        is only valid for LNSTensors that contain a single element.
        """
        return self.value.item()

    def size(self, dim: int | None = None) -> torch.Size | int:
        """
        Returns the size of the LNSTensor along a specified dimension or all dimensions.

        Parameters
        ----------
        dim : int, optional
            If specified, returns the size of the given dimension; otherwise,
        returns the size of all dimensions.

        Returns
        -------
        torch.Size or int
            The size of the LNSTensor. The returned value is a ``torch.Size`` if
            `dim` is ``None``, otherwise it returns an integer representing the
            size of the specified dimension.
        """
        return self._lns.size(dim=dim)

    def numel(self) -> int:
        """
        Returns the total number of elements in the LNSTensor.

        Returns
        -------
        int
            The total number of elements in the LNSTensor.
        """
        return self._lns.numel()

    def dim(self) -> int:
        """
        Returns the number of dimensions of the LNSTensor.

        Returns
        -------
        int
            The number of dimensions of the LNSTensor.
        """
        return self._lns.dim()

    def broadcast_to(self, shape) -> LNSTensor:
        """
        Broadcasts ``self`` to the shape ``shape``. Analogous to

        https://docs.pytorch.org/docs/stable/generated/torch.broadcast_to.html

        Parameters
        ----------
        shape : list, tuple, torch.Size
            The new shape to broadcast to.

        Returns
        -------
        An ```LNSTensor`` object broadcasted to the new shape ``shape``.
        """
        return lnstensor(self._lns.broadcast_to(shape), from_lns=True, b=self.base)

    def clone(self, *, memory_format=torch.preserve_format) -> LNSTensor:
        """
        Returns a copy of the LNSTensor with the same data and base.
        """
        cloned_lns = self._lns.clone(memory_format=memory_format)
        return lnstensor(cloned_lns, from_lns=True, b=self.base)

    def squeeze(self, dim: Union[int, List[int]] | None = None) -> LNSTensor:
        """
        Returns a new LNSTensor with all specified dimensions of size
        1 removed. If no dimensions are specified, all dimensions of
        size 1 are removed.

        Parameters
        ----------
        dim : int, list of int, optional
            The dimension(s) to remove. If ``None``, all dimensions of size 1
            are removed. Defaults to ``None``.

        Returns
        -------
        LNSTensor
            A new LNSTensor with the specified dimensions removed.
        """
        return lnstensor(self._lns.squeeze(dim), from_lns=True, b=self.base)

    def unsqueeze(self, dim: int) -> LNSTensor:
        """
        Returns a new LNSTensor with a dimension of size one inserted at the specified position.

        Parameters
        ----------
        dim : int
            The dimension index at which to insert the new dimension.

        Returns
        -------
        LNSTensor
            A new LNSTensor with the specified dimension added.
        """
        return lnstensor(self._lns.unsqueeze(dim), from_lns=True, b=self.base)

    def detach(self) -> LNSTensor:
        """
        Returns a new LNSTensor that is detached from the current computation graph.
        The returned tensor will **not** require gradients.

        Returns
        -------
        LNSTensor
            A new LNSTensor that is detached from the current computation graph.
        """
        return lnstensor(self._lns.detach(), from_lns=True, b=self.base)

    def requires_grad_(self, requires_grad: bool = True) -> LNSTensor:
        """
        Sets the requires_grad flag for the LNSTensor in-place.

        Parameters
        ----------
        requires_grad : bool, optional
            If ``True``, the LNSTensor will track gradients. Defaults to ``True``.

        Returns
        -------
        LNSTensor
            The LNSTensor with the updated requires_grad flag.
        """
        self._lns.requires_grad_(requires_grad)
        if requires_grad:
            self.register_grad_hooks()
        return self

    def __repr__(self) -> str:
        return f"LNSTensor(value={self.value}, base={self.base.item()})"

    def __add__(self, other):
        if isinstance(other, _xlns_types):
            other = lnstensor(other, b=self.base)
        return torch.add(self, other)

    def __sub__(self, other):
        if isinstance(other, _xlns_types):
            other = lnstensor(other, b=self.base)
        return torch.sub(self, other)

    def __mul__(self, other):
        if isinstance(other, _xlns_types):
            other = lnstensor(other, b=self.base)
        return torch.mul(self, other)

    def __truediv__(self, other):
        if isinstance(other, _xlns_types):
            other = lnstensor(other, b=self.base)
        return torch.div(self, other)

    def __pow__(self, other):
        return torch.pow(self, other) # not implemented LNSTensor powers for now

    def __neg__(self):
        return torch.neg(self)

    def __pos__(self):
        return torch.pos(self)

    def __eq__(self, other):
        if isinstance(other, _xlns_types):
            other = lnstensor(other, b=self.base)
        return torch.eq(self, other)

    def __ne__(self, other):
        if isinstance(other, _xlns_types):
            other = lnstensor(other, b=self.base)
        return torch.ne(self, other)

    def __ge__(self, other):
        if isinstance(other, _xlns_types):
            other = lnstensor(other, b=self.base)
        return torch.ge(self, other)

    def __gt__(self, other):
        if isinstance(other, _xlns_types):
            other = lnstensor(other, b=self.base)
        return torch.gt(self, other)

    def __le__(self, other):
        if isinstance(other, _xlns_types):
            other = lnstensor(other, b=self.base)
        return torch.le(self, other)

    def __lt__(self, other):
        if isinstance(other, _xlns_types):
            other = lnstensor(other, b=self.base)
        return torch.lt(self, other)

    def add(self, other, *, alpha=1):
        return torch.add(self, other, alpha=alpha)

    def add_(self, other, *, alpha=1):
        self._lns = torch.add(self, other, alpha=alpha)._lns
        return self

    def sub(self, other, *, alpha=1):
        return torch.sub(self, other, alpha=1)

    def sub_(self, other, *, alpha=1):
        self._lns = torch.sub(self, other, alpha=alpha)._lns
        return self

    def mul(self, other):
        return torch.mul(self, other)

    def mul_(self, other):
        self._lns = torch.mul(self, other)._lns
        return self

    def div(self, other):
        return torch.div(self, other)

    def div_(self, other):
        self._lns = torch.div(self, other)._lns
        return self

    def pow(self, other):
        return torch.pow(self, other)

    def pow_(self, other):
        self._lns = torch.pow(self, other)._lns
        return self

    def neg(self):
        return torch.neg(self)

    def neg_(self):
        self._lns = torch.neg(self)._lns
        return self

    def abs(self):
        return torch.abs(self)

    def abs_(self):
        self._lns = torch.abs(self)._lns
        return self

    def sqrt(self):
        return torch.sqrt(self)

    def sqrt_(self):
        self._lns = torch.sqrt(self)._lns
        return self

    def square(self):
        return torch.square(self)

    def square_(self):
        self._lns = torch.square(self)._lns
        return self

    def reciprocal(self):
        return torch.reciprocal(self)

    def reciprocal_(self):
        self._lns = torch.reciprocal(self)._lns
        return self

    def sign(self):
        return torch.sign(self)

    def sign_(self):
        self._lns = torch.sign(self)._lns
        return self

    def positive(self):
        return torch.pos(self)

    def sum(self, dim=None, keepdim=False):
        return torch.sum(self, dim=dim, keepdim=keepdim)

    def transpose(self, dim0, dim1):
        return torch.transpose(self, dim0, dim1)

    def equal(self, other):
        return torch.equal(self, other)

    def eq(self, other):
        return torch.eq(self, other)

    def ne(self, other):
        return torch.ne(self, other)

    def ge(self, other):
        return torch.ge(self, other)

    def gt(self, other):
        return torch.gt(self, other)

    def le(self, other):
        return torch.le(self, other)

    def lt(self, other):
        return torch.lt(self, other)

    def isclose(self, other, rtol=1e-05, atol=1e-08):
        return torch.isclose(self, other, rtol=rtol, atol=atol)

    def allclose(self, other, rtol=1e-05, atol=1e-08):
        return torch.allclose(self, other, rtol=rtol, atol=atol)

    def any(self, dim=None, keepdim=False):
        return torch.any(self, dim=dim, keepdim=keepdim)

    def all(self, dim=None, keepdim=False):
        return torch.all(self, dim=dim, keepdim=keepdim)

    def sort(self, dim=-1, descending=False, stable=False):
        return torch.sort(self, dim=dim, descending=descending, stable=stable)

    def argsort(self, dim=-1, descending=False, stable=False):
        return torch.argsort(self, dim=dim, descending=descending, stable=stable)

    def kthvalue(self, k, dim=None, keepdim=False):
        return torch.kthvalue(self, k, dim=dim, keepdim=keepdim)

    def maximum(self, other):
        return torch.maximum(self, other)

    def minimum(self, other):
        return torch.minimum(self, other)

    def tanh(self):
        return torch.tanh(self)

    def sigmoid(self):
        return torch.sigmoid(self)

def lnstensor(
        data: Any,
        from_lns: bool = False,
        requires_grad: bool = False,
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
    requires_grad : bool, optional
        If ``True``, the LNSTensor will track gradients. Defaults to ``False``.
        If a pre-packed LNSTensor or a torch.Tensor is provided, this parameter
        is ignored.
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
        input_data = data.lns
        from_lns = True
        requires_grad = data.lns.requires_grad

        if not torch.eq(data.base, base_tensor):
            with torch.no_grad():
                packed_int = input_data.to(torch.int64)
                sign_bit = packed_int & 1
                exponent = (packed_int >> 1).to(torch.float64)

                exponent_new = exponent * torch.log(data.base) / torch.log(base_tensor)
                new_packed_int = (exponent_new.round().to(torch.int64) << 1) | sign_bit
                input_data = new_packed_int.to(torch.float64)
                input_data = torch.where(torch.eq(data, LNS_ZERO), LNS_ZERO, input_data)

    # torch.Tensor
    elif isinstance(data, torch.Tensor):
        input_data = data.to(torch.float64)
        requires_grad = input_data.requires_grad

    # numpy.ndarray
    elif isinstance(data, np.ndarray):
        input_data = torch.from_numpy(data).to(torch.float64)

    # xlns scalar objects
    elif isinstance(data, (xl.xlns, xl.xlnsud, xl.xlnsv, xl.xlnsb)):
        if data.x == -math.inf:
            input_data = LNS_ZERO

        else:
            if isinstance(data, (xl.xlns, xl.xlnsud)) and not base_val == xl.xlnsB:
                log_part = data.x * np.log(xl.xlnsB) / np.log(base_val)
            elif isinstance(data, (xl.xlnsb, xl.xlnsv)) and not base_val == data.B:
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
        input_data = torch.tensor(
            np.where(data.nd == -2**53, LNS_ZERO, packed_int),
            dtype=torch.float64
        )
        from_lns = True

    # Everything else (scalars, lists, tuples, etc.)
    else:
        try:
            input_data = torch.tensor(data, dtype=torch.float64)
        except Exception as e:
            raise TypeError(f"Unsupported data type for LNSTensor: {type(data).__name__}") from e

    return LNSTensor(input_data, base_tensor, from_lns=from_lns, requires_grad=requires_grad)

def zeros(
        *size,
        out=None,
        layout=torch.strided,
        device=None,
        requires_grad=False,
        f=None,
        b=None
        ) -> LNSTensor:
    """
    Returns an LNSTensor filled with zeros, with the specified shape and
    properties. See `torch.zeros` for more details on the parameters.
    """
    result = lnstensor(
        torch.zeros(*size, dtype=torch.float64, layout=layout,
                    device=device, requires_grad=requires_grad),
        from_lns=False, f=f, b=b
    )

    if out is not None:
        out.copy_(result)

    return result

def zeros_like(
        input,
        *,
        layout=None,
        device=None,
        requires_grad=False,
        memory_format=torch.preserve_format,
        f=None,
        b=None,
        ) -> LNSTensor:
    """
    Returns an LNSTensor filled with zeros, with the same shape and
    properties as the input tensor. See `torch.zeros_like` for more
    details on the parameters.
    """
    if isinstance(input, LNSTensor):
        if f is None and b is None:
            b = input.base
        input = input._lns

    return lnstensor(
        torch.zeros_like(input, dtype=torch.float64, device=device, layout=layout,
                         requires_grad=requires_grad, memory_format=memory_format),
        from_lns=False, f=f, b=b
    )

def ones(
        *size,
        out=None,
        layout=torch.strided,
        device=None,
        requires_grad=False,
        f=None,
        b=None
        ) -> LNSTensor:
    """
    Returns an LNSTensor filled with ones, with the specified shape and
    properties. See `torch.ones` for more details on the parameters.
    """
    result = lnstensor(
        torch.ones(*size, dtype=torch.float64, layout=layout,
                   device=device, requires_grad=requires_grad),
        from_lns=False, f=f, b=b
    )

    if out is not None:
        out.copy_(result)

    return result

def ones_like(
        input,
        *,
        layout=None,
        device=None,
        requires_grad=False,
        memory_format=torch.preserve_format,
        f=None,
        b=None,
        ) -> LNSTensor:
    """
    Returns an LNSTensor filled with ones, with the same shape and
    properties as the input tensor. See `torch.ones_like` for more
    details on the parameters.
    """
    if isinstance(input, LNSTensor):
        if f is None and b is None:
            b = input.base
        input = input._lns

    return lnstensor(
        torch.ones_like(input, dtype=torch.float64, device=device, layout=layout,
                        memory_format=memory_format, requires_grad=requires_grad),
        from_lns=False, f=f, b=b
    )

def full(
        size,
        fill_value,
        *,
        out=None,
        layout=torch.strided,
        device=None,
        requires_grad=False,
        f=None,
        b=None
        ) -> LNSTensor:
    """
    Returns an LNSTensor filled with `fill_value`, with the specified shape
    and properties. See `torch.full` for more details on the parameters.
    """
    result = lnstensor(
        torch.full(size, fill_value, dtype=torch.float64, layout=layout,
                   device=device, requires_grad=requires_grad),
        from_lns=False, f=f, b=b
    )

    if out is not None:
        out.copy_(result)

    return result

def full_like(
        input,
        fill_value,
        *,
        layout=None,
        device=None,
        requires_grad=False,
        memory_format=torch.preserve_format,
        f=None,
        b=None,
        ) -> LNSTensor:
    """
    Returns an LNSTensor filled with `fill_value`, with the same shape
    and properties as the input tensor. See `torch.full_like` for more
    details on the parameters.
    """
    if isinstance(input, LNSTensor):
        if f is None and b is None:
            b = input.base
        input = input._lns

    return lnstensor(
        torch.full_like(input, fill_value, dtype=torch.float64, device=device, layout=layout,
                        memory_format=memory_format, requires_grad=requires_grad),
        from_lns=False, f=f, b=b
    )

def rand(
        *size,
        generator=None,
        out=None,
        layout=torch.strided,
        device=None,
        requires_grad=False,
        pin_memory=False,
        f=None,
        b=None
        ) -> LNSTensor:
    """
    Returns an LNSTensor filled with random numbers from a uniform
    distribution on the interval [0, 1], with the specified shape and
    properties. See `torch.rand` for more details on the parameters.
    """
    result = lnstensor(
        torch.rand(size, generator=generator, dtype=torch.float64, layout=layout,
                   device=device, pin_memory=pin_memory, requires_grad=requires_grad),
        from_lns=False, f=f, b=b
    )

    if out is not None:
        out.copy_(result)

    return result

def rand_like(
        input,
        *,
        layout=None,
        device=None,
        requires_grad=False,
        memory_format=torch.preserve_format,
        f=None,
        b=None,
        ) -> LNSTensor:
    """
    Returns an LNSTensor filled with random numbers from a uniform
    distribution on the interval [0, 1], with the same shape and
    properties as the input tensor. See `torch.rand_like` for more
    details on the parameters.
    """
    if isinstance(input, LNSTensor):
        if f is None and b is None:
            b = input.base
        input = input._lns

    return lnstensor(
        torch.rand_like(input, dtype=torch.float64, device=device, layout=layout,
                        memory_format=memory_format, requires_grad=requires_grad),
        from_lns=False, f=f, b=b
    )

def randn(
        *size,
        generator=None,
        out=None,
        layout=torch.strided,
        device=None,
        requires_grad=False,
        pin_memory=False,
        f=None,
        b=None
        ) -> LNSTensor:
    """
    Returns an LNSTensor filled with random numbers from a normal
    distribution with mean 0 and variance 1, with the specified shape
    and properties. See `torch.rand` for more details on the parameters.
    """
    result = lnstensor(
        torch.randn(size, generator=generator, dtype=torch.float64, layout=layout,
                    device=device, pin_memory=pin_memory, requires_grad=requires_grad),
        from_lns=False, f=f, b=b
    )

    if out is not None:
        out.copy_(result)

    return result

def randn_like(
        input,
        *,
        layout=None,
        device=None,
        requires_grad=False,
        memory_format=torch.preserve_format,
        f=None,
        b=None,
        ) -> LNSTensor:
    """
    Returns an LNSTensor filled with random numbers from a normal
    distribution with mean 0 and variance 1, with the same shape and
    properties as the input tensor. See `torch.rand_like` for more
    details on the parameters.
    """
    if isinstance(input, LNSTensor):
        if f is None and b is None:
            b = input.base
        input = input._lns

    return lnstensor(
        torch.randn_like(input, dtype=torch.float64, device=device, layout=layout,
                         memory_format=memory_format, requires_grad=requires_grad),
        from_lns=False, f=f, b=b
    )
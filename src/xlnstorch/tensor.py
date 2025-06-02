from typing import Any, Union

import numpy as np
import torch
from torch import Tensor
import xlns as xl

class LNSTensor(object):
    r"""
    A logarithmic number system (LNS) wrapper for PyTorch tensors. 

    Internally each real number :math:`x` is stored as one packed integer

    .. math::

        \text{sign\_bit} = \begin{cases}
            0 & x \ge 0 \\
            1 & x <  0
        \end{cases}

        \text{exponent} = \mathrm{round}\!\left(
            \frac{\log(|x|)}{\log(\text{base})}
        \right)

        \text{packed}   = (\text{exponent} \ll 1)\;|\;\text{sign\_bit}

    The packed integers live in :pyattr:`_lns` (stored as a ``torch.float64``
    Tensor so that gradients can be stored for autograd).

    Parameters
    ----------
    :param data:
        *Real-valued* tensor to encode **or** a pre-packed LNS tensor when
        ``from_lns`` is ``True``.
    :type  data: ``torch.Tensor`` (dtype ``float64``)
    :param base:
        Scalar tensor that holds the logarithm base.
        Must be positive and **not** equal to 1.
    :type  base: ``torch.Tensor`` (shape ``()``, dtype ``float64``)
    :param bool from_lns:
        If ``True`` interpret *data* as already packed.
        Defaults to ``False``.
    :raises ValueError: If *base* is 0 or 1.

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

    @property
    def lns(self) -> Tensor:
        """
        The packed representation that **does** carry gradients.

        :returns: Tensor of dtype ``float64`` holding the packed integers.
        :rtype:   torch.Tensor
        """
        return self._lns

    @property
    def value(self) -> Tensor:
        """
        Decode the packed integers back to ordinary floating-point numbers.

        :returns: Real-valued tensor (dtype ``float64``).
        :rtype:   torch.Tensor
        """
        packed_int = self._lns.to(torch.int64)

        exponent = (packed_int >> 1).to(torch.float64)
        sign = torch.where((packed_int & 1).bool(), -1.0, 1.0)

        return sign * torch.pow(self.base, exponent)

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
    in particular, it supports all ``xlns`` data types (except the
    redundant types `xlnsr`, `xlnsnpr`).

    Base Selection
    --------------
    The LNSTensor ``base`` is chosen in the following order:

    1. If ``f`` is given:      ``base`` = 2.0 ** 2 ** (-f)``.
    2. Else if ``b is given``: use ``b`` (float *or* scalar tensor).
    3. Else                    default ``xlns.xlnsB`` (global constant).

    Parameters
    ----------
    :param data:
        - A real-valued tensor/array/scalar to *encode* **or**
        - A pre-packed representation (when ``from_lns`` is ``True``) **or**
        - An existing :class:`LNSTensor` (will be copied or converted).
    :type  data: ``torch.Tensor``, ``numpy.ndarray``, numbers, xlns types, :class:`LNSTensor`
    :param bool from_lns:
        If ``True``, interpret *data* as already packed.
        Defaults to ``False``.
    :param f:
        The number of fractional exponent bits. mutually exclusive with ``b``.
    :type  f: ``int``, optional
    :param b:
        The explicit logarithm base; mutually exclusive with ``f``.
    :type  b: ``float``, ``int``, ``torch.Tensor``, optional

    Returns
    -------
    :rtype: :class:`LNSTensor`
        The constructed LNSTensor.

    Raises
    ------
    :class:`ValueError`:
        If both ``f`` and ``b`` are provided, or if neither can be resolved
        to a valid base.
    :class:`TypeError`:
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
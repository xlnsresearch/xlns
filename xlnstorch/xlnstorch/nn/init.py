from typing import Optional, Union
import torch
from xlnstorch import LNSTensor, LNS_ZERO, LNS_ONE

__all__ = [
    "uniform_",
    "normal_",
    "zeros_",
    "ones_",
    "constant_",
    "eye_",
    "xavier_uniform_",
    "xavier_normal_",
]

def uniform_(
        tensor: LNSTensor,
        a: float = 0.0,
        b: float = 1.0,
        generator: Optional[torch.Generator] = None,
    ):
    """
    Fills the input tensor with random numbers from a uniform distribution.

    Parameters
    ----------
    tensor : LNSTensor
        The tensor to fill with random numbers.
    a : float, optional
        The lower bound of the uniform distribution (default is 0.0).
    b : float, optional
        The upper bound of the uniform distribution (default is 1.0).
    generator : torch.Generator, optional
        A random number generator to use for reproducibility (default is None).

    Returns
    -------
    LNSTensor
        The input tensor filled with random numbers from the uniform distribution.
    """
    torch_tensor = torch.empty(tensor.shape).uniform_(a, b, generator=generator)
    tensor._lns.data.copy_(LNSTensor.get_internal_tensor(torch_tensor, tensor.base))
    return tensor

def normal_(
        tensor: LNSTensor,
        mean: float = 0.0,
        std: float = 1.0,
        generator: Optional[torch.Generator] = None,
    ):
    """
    Fills the input tensor with random numbers from a normal distribution.

    Parameters
    ----------
    tensor : LNSTensor
        The tensor to fill with random numbers.
    mean : float, optional
        The mean of the normal distribution (default is 0.0).
    std : float, optional
        The standard deviation of the normal distribution (default is 1.0).
    generator : torch.Generator, optional
        A random number generator to use for reproducibility (default is None).

    Returns
    -------
    LNSTensor
        The input tensor filled with random numbers from the normal distribution.
    """
    torch_tensor = torch.normal(mean, std, size=tensor.shape, generator=generator)
    tensor._lns.data.copy_(LNSTensor.get_internal_tensor(torch_tensor, tensor.base))
    return tensor

def zeros_(
        tensor: LNSTensor,
    ):
    """
    Fills the input tensor with zeros.

    Parameters
    ----------
    tensor : LNSTensor
        The tensor to fill with zeros.

    Returns
    -------
    LNSTensor
        The input tensor filled with zeros.
    """
    tensor._lns.data.fill_(LNS_ZERO)
    return tensor

def ones_(
        tensor: LNSTensor,
    ):
    """
    Fills the input tensor with ones.

    Parameters
    ----------
    tensor : LNSTensor
        The tensor to fill with ones.

    Returns
    -------
    LNSTensor
        The input tensor filled with ones.
    """
    tensor._lns.data.fill_(LNS_ONE)
    return tensor

def constant_(
        tensor: LNSTensor,
        value: Union[float, LNSTensor],
    ):
    """
    Fills the input tensor with a constant value.

    Parameters
    ----------
    tensor : LNSTensor
        The tensor to fill with a constant value.
    value : float or LNSTensor
        The constant value to fill the tensor with.
        If a float is provided, the tensor will be filled with that float value.
        If an LNSTensor is provided, it must be a scalar (i.e., have a single element).
    """
    if isinstance(value, LNSTensor):
        if value.numel() != 1:
            raise ValueError("If 'value' is an LNSTensor, it must be a scalar (i.e., have a single element).")
        value_lns = value._lns.data.item()
    else:
        value_lns = LNSTensor.get_internal_tensor(value, tensor.base).data.item()

    tensor._lns.data.fill_(value_lns)
    return tensor

def eye_(
        tensor: LNSTensor,
    ):
    """
    Fills the input 2D tensor with ones on the diagonal and zeros elsewhere.

    Parameters
    ----------
    tensor : LNSTensor
        The 2D tensor to fill with the identity matrix.

    Returns
    -------
    LNSTensor
        The input tensor filled with the identity matrix.
    """
    if tensor.ndim != 2:
        raise ValueError("Input tensor must be 2-dimensional.")

    tensor._lns.data.fill_(LNS_ZERO)

    n = min(tensor.shape)
    for i in range(n):
        tensor._lns.data[i, i] = LNS_ONE

    return tensor

def _calculate_fan_in_and_fan_out(tensor: LNSTensor):
    dimensions = tensor.ndim
    if dimensions < 2:
        raise ValueError("Fan in and fan out can not be computed for tensor with fewer than 2 dimensions")

    num_input_fmaps = tensor.shape[1]
    num_output_fmaps = tensor.shape[0]

    receptive_field_size = 1
    if dimensions > 2:
        for s in tensor.shape[2:]:
            receptive_field_size *= s

    fan_in = num_input_fmaps * receptive_field_size
    fan_out = num_output_fmaps * receptive_field_size

    return fan_in, fan_out

def xavier_uniform_(
        tensor: LNSTensor,
        gain: float = 1.0,
        generator: Optional[torch.Generator] = None,
    ):
    """
    Fills the input tensor with values according to the Xavier uniform initialization.

    Parameters
    ----------
    tensor : LNSTensor
        The tensor to fill with values.
    gain : float, optional
        An optional scaling factor (default is 1.0).
    generator : torch.Generator, optional
        A random number generator to use for reproducibility (default is None).

    Returns
    -------
    LNSTensor
        The input tensor filled with values from the Xavier uniform distribution.
    """
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    std = gain * (2.0 / (fan_in + fan_out)) ** 0.5
    a = (3.0 ** 0.5) * std

    return uniform_(tensor, -a, a, generator)

def xavier_normal_(
        tensor: LNSTensor,
        gain: float = 1.0,
        generator: Optional[torch.Generator] = None,
    ):
    """
    Fills the input tensor with values according to the Xavier normal initialization.

    Parameters
    ----------
    tensor : LNSTensor
        The tensor to fill with values.
    gain : float, optional
        An optional scaling factor (default is 1.0).
    generator : torch.Generator, optional
        A random number generator to use for reproducibility (default is None).

    Returns
    -------
    LNSTensor
        The input tensor filled with values from the Xavier normal distribution.
    """
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    std = gain * (2.0 / (fan_in + fan_out)) ** 0.5

    return normal_(tensor, 0.0, std, generator)
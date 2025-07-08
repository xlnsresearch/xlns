import torch
from .. import LNSTensor, LNS_ZERO

__all__ = [
    "normal_",
    "zeros_",
]

def normal_(
        tensor: LNSTensor,
        mean: float = 0.0,
        std: float = 1.0,
        generator: torch.Generator | None = None,
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
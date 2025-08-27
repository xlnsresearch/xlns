from __future__ import annotations
from typing import Any, Dict, Tuple, Type, Callable, Union, Optional
import torch
from xlnstorch import LNSTensor, lnstensor

# optional import of torchvision.v2
try:
    import torchvision.transforms.v2
    _TV_AVAILABLE = True
except ImportError:
    _TV_AVAILABLE = False

# Here we define a custom collate function for LNSTensor that can be used with DataLoader.
def collate_lnstensor_fn(
        batch,
        *,
        collate_fn_map: Optional[Dict[Union[Type, Tuple[Type, ...]], Callable]] = None,
    ):
    return torch.stack(batch, 0)
torch.utils.data._utils.collate.default_collate_fn_map.update({LNSTensor: collate_lnstensor_fn})

class ToLNSTensor:
    """
    Convert an image-like object (PIL image or numpy.ndarray) with shape
    (H x W x C) in the range [0, 255] to an LNSTensor with shape (C x H x W)
    in the range [0., 1.]. This is analogous to torchvision.transforms.ToTensor
    but wraps the output in an LNSTensor.

    Non-floating point inputs are not converted to LNSTensor unless specified
    by the `wrap_all` parameter.

    Parameters
    ----------
    f : Optional[int], optional
        The precision of the output LNSTensor.
    b : Optional[float], optional
        The base of the output LNSTensor.
    wrap_all : bool, optional
        If True, all inputs are wrapped in an LNSTensor, even if they are not
        floating point. If False (default), only floating point inputs are
        wrapped.
    device : Optional[Union[str, torch.device]], optional
        The device on which the output LNSTensor should be allocated. If None,
        it defaults to the current device.

    Raises
    ------
    ImportError
        If torchvision is not available, an ImportError is raised when trying
        to use this class.
    """

    # Build the underlying torchvision pipeline once and reuse it
    _PIPELINE: Optional[torchvision.transforms.v2.Compose] = None

    def __init__(
            self,
            f: Optional[int] = None,
            b: Optional[float] = None,
            wrap_all: bool = False,
            device: Optional[Union[str, torch.device]] = None,
        ) -> None:
        if not _TV_AVAILABLE:
            raise ImportError(
                "ToLNSTensor requires torchvision.\n"
                "Install it with:  pip install torchvision"
            )
        self.f = f
        self.b = b
        self.wrap_all = wrap_all
        self.device = torch.device(device) if device is not None else None

        # Lazily create the Compose the very first time the class is used
        if ToLNSTensor._PIPELINE is None:
            ToLNSTensor._PIPELINE = torchvision.transforms.v2.Compose((
                torchvision.transforms.v2.ToImage(),
                torchvision.transforms.v2.ToDtype(torch.float64, scale=True),
            ))

    def __call__(self, img: Any) -> Union[LNSTensor, torch.Tensor]:
        """
        Parameters
        ----------
        img : Any
            The input image-like object to convert. This can be a PIL image,
            numpy.ndarray, or any other object that can be processed by
            torchvision.transforms.v2.ToImage.

        Returns
        -------
        Union[LNSTensor, torch.Tensor]
            The converted image as an LNSTensor if `wrap_all` is True or if the
            input is a floating point tensor. Otherwise, it returns the input
            tensor unchanged.
        """
        if torch.is_tensor(img):
            tensor = img
        else:
            tensor = ToLNSTensor._PIPELINE(img)

        if self.wrap_all or torch.is_floating_point(tensor):
            return lnstensor(tensor, f=self.f, b=self.b).to(self.device)
        else:
            return tensor.to(self.device)

class LNSNormalize:
    """
    Normalize an LNSTensor with mean and standard deviation. This is analogous
    to torchvision.transforms.Normalize but performs normalization on LNSTensor.

    Parameters
    ----------
    mean : Union[float, Tuple[float, ...]]
        The mean value(s) for normalization.
    std : Union[float, Tuple[float, ...]]
        The standard deviation value(s) for normalization.
    f : Optional[int] = None, optional
        The precision of the input LNSTensor.
    b : Optional[float] = None, optional
        The base of the input LNSTensor.

    Returns
    -------
    LNSTensor
        The normalized LNSTensor.
    """

    def __init__(
            self,
            mean: Union[float, Tuple[float, ...]],
            std: Union[float, Tuple[float, ...]],
            f: Optional[int] = None,
            b: Optional[float] = None,
    ):
        self.mean = lnstensor(mean, f=f, b=b)
        self.std = lnstensor(std, f=f, b=b)
        assert self.mean.ndim == 1 or self.mean.ndim == 0, "Mean must be a 1D or 0D tensor."
        assert self.std.ndim == 1 or self.std.ndim == 0, "Std must be a 1D or 0D tensor."
        self.mean = self.mean.unsqueeze(-1).unsqueeze(-1)
        self.std = self.std.unsqueeze(-1).unsqueeze(-1)

    def __call__(self, tensor: LNSTensor) -> LNSTensor:
        assert tensor.ndim >= 3, "Expected tensor to be an image of size" \
            f"(..., C, H, W) but got {tensor.ndim} dimensions."

        assert self.mean.numel() == tensor.shape[-3] or self.mean.numel() == 1, \
            "The number of channels in the tensor must match the length of mean. " \
            "Or the mean must be a scalar."

        assert self.std.numel() == tensor.shape[-3] or self.std.numel() == 1, \
            "The number of channels in the tensor must match the length of std. " \
            "Or the std must be a scalar."

        return (tensor - self.mean) / self.std
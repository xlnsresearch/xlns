from __future__ import annotations
from typing import Any, Dict, Tuple, Type, Callable
import torch
from . import LNSTensor, lnstensor

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
        collate_fn_map: Dict[Type | Tuple[Type, ...], Callable] | None = None,
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
    f : int | None, optional
        The precision of the output LNSTensor.
    b : float | None, optional
        The base of the output LNSTensor.
    wrap_all : bool, optional
        If True, all inputs are wrapped in an LNSTensor, even if they are not
        floating point. If False (default), only floating point inputs are
        wrapped.
    device : str | torch.device, optional
        The device on which the output LNSTensor should be allocated. If None,
        it defaults to the current device.

    Raises
    ------
    ImportError
        If torchvision is not available, an ImportError is raised when trying
        to use this class.
    """

    # Build the underlying torchvision pipeline once and reuse it
    _PIPELINE: torchvision.transforms.v2.Compose | None = None

    def __init__(
            self,
            f: int | None = None,
            b: float | None = None,
            wrap_all: bool = False,
            device: str | torch.device | None = None,
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

    def __call__(self, img: Any) -> LNSTensor | torch.Tensor:
        """
        Parameters
        ----------
        img : Any
            The input image-like object to convert. This can be a PIL image,
            numpy.ndarray, or any other object that can be processed by
            torchvision.transforms.v2.ToImage.

        Returns
        -------
        LNSTensor | torch.Tensor
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
from typing import Tuple, Union, Optional
import collections

import torch
from . import LNSModule

def _pair(x):
    if isinstance(x,collections.abc.Iterable):
        return tuple(x)
    return (x, x)

def _triple(x):
    if isinstance(x, collections.abc.Iterable):
        return tuple(x)
    return (x, x, x)

class LNSAvgPool1d(LNSModule):
    """
    An LNS 1D average pooling layer that applies a
    1D average pooling operation over the input tensor.

    See also: :py:class:`torch.nn.AvgPool1d`

    Parameters
    ----------
    kernel_size : int
        The size of the window to take the average over.
    stride : int, optional
        The stride of the window. Default is equal to `kernel_size`.
    padding : int, optional
        Implicit zero padding to be added on both sides of the input. Default is 0
    ceil_mode : bool, optional
        If True, will use ceil instead of floor to compute the output shape.
        Default is False.
    count_include_pad : bool, optional
        If True, will include the zero-padding in the averaging calculation.
        Default is True.
    """

    def __init__(
            self,
            kernel_size: int,
            stride: Optional[int] = None,
            padding: int = 0,
            ceil_mode: bool = False,
            count_include_pad: bool = True
        ):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.ceil_mode = ceil_mode
        self.count_include_pad = count_include_pad

    def forward(self, x):
        return torch.nn.functional.avg_pool1d(
            x, self.kernel_size, self.stride, self.padding,
            self.ceil_mode, self.count_include_pad
        )

class LNSAvgPool2d(LNSModule):
    """
    An LNS 2D average pooling layer that applies a
    2D average pooling operation over the input tensor.

    See also: :py:class:`torch.nn.AvgPool2d`

    Parameters
    ----------
    kernel_size : int or tuple
        The size of the window to take the average over.
    stride : int or tuple, optional
        The stride of the window. Default is equal to `kernel_size`.
    padding : int or tuple, optional
        Implicit zero padding to be added on both sides of the input. Default is 0
    ceil_mode : bool, optional
        If True, will use ceil instead of floor to compute the output shape.
        Default is False.
    count_include_pad : bool, optional
        If True, will include the zero-padding in the averaging calculation.
        Default is True.
    divisor_override : int, optional
        If specified, will use this value as the divisor instead of the kernel size.
        Default is None, which means the divisor will be the kernel size.
    """

    def __init__(
            self,
            kernel_size: Union[int, Tuple[int, int]],
            stride: Optional[Union[int, Tuple[int, int]]] = None,
            padding: Union[int, Tuple[int, int]] = 0,
            ceil_mode: bool = False,
            count_include_pad: bool = True,
            divisor_override: Optional[int] = None
        ):
        super().__init__()
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.ceil_mode = ceil_mode
        self.count_include_pad = count_include_pad
        self.divisor_override = divisor_override

    def forward(self, x):
        return torch.nn.functional.avg_pool2d(
            x, self.kernel_size, self.stride, self.padding,
            self.ceil_mode, self.count_include_pad, self.divisor_override
        )

class LNSAvgPool3d(LNSModule):
    """
    An LNS 3D average pooling layer that applies a
    3D average pooling operation over the input tensor.

    See also: :py:class:`torch.nn.AvgPool3d`

    Parameters
    ----------
    kernel_size : int or tuple
        The size of the window to take the average over.
    stride : int or tuple, optional
        The stride of the window. Default is equal to `kernel_size`.
    padding : int or tuple, optional
        Implicit zero padding to be added on both sides of the input. Default is 0
    ceil_mode : bool, optional
        If True, will use ceil instead of floor to compute the output shape.
        Default is False.
    count_include_pad : bool, optional
        If True, will include the zero-padding in the averaging calculation.
        Default is True.
    divisor_override : int, optional
        If specified, will use this value as the divisor instead of the kernel size.
        Default is None, which means the divisor will be the kernel size.
    """

    def __init__(
            self,
            kernel_size: Union[int, Tuple[int, int, int]],
            stride: Optional[Union[int, Tuple[int, int, int]]] = None,
            padding: Union[int, Tuple[int, int, int]] = 0,
            ceil_mode: bool = False,
            count_include_pad: bool = True,
            divisor_override: Optional[int] = None
        ):
        super().__init__()
        self.kernel_size = _triple(kernel_size)
        self.stride = _triple(stride)
        self.padding = _triple(padding)
        self.ceil_mode = ceil_mode
        self.count_include_pad = count_include_pad
        self.divisor_override = divisor_override

    def forward(self, x):
        return torch.nn.functional.avg_pool3d(
            x, self.kernel_size, self.stride, self.padding,
            self.ceil_mode, self.count_include_pad, self.divisor_override
        )

class LNSAdaptiveAvgPool1d(LNSModule):
    """
    An LNS 1D adaptive average pooling layer that applies a
    1D adaptive average pooling operation over the input tensor.

    See also: :py:class:`torch.nn.AdaptiveAvgPool1d`

    Parameters
    ----------
    output_size : int, tuple
        The target output size.
    """

    def __init__(self, output_size: int):
        super().__init__()
        self.output_size = output_size

    def forward(self, x):
        return torch.nn.functional.adaptive_avg_pool1d(x, self.output_size)

class LNSAdaptiveAvgPool2d(LNSModule):
    """
    An LNS 2D adaptive average pooling layer that applies a
    2D adaptive average pooling operation over the input tensor.

    See also: :py:class:`torch.nn.AdaptiveAvgPool2d`

    Parameters
    ----------
    output_size : int, tuple
        The target output size.
    """

    def __init__(self, output_size: int):
        super().__init__()
        self.output_size = output_size

    def forward(self, x):
        return torch.nn.functional.adaptive_avg_pool2d(x, self.output_size)

class LNSAdaptiveAvgPool3d(LNSModule):
    """
    An LNS 3D adaptive average pooling layer that applies a
    3D adaptive average pooling operation over the input tensor.

    See also: :py:class:`torch.nn.AdaptiveAvgPool3d`

    Parameters
    ----------
    output_size : int, tuple
        The target output size.
    """

    def __init__(self, output_size: int):
        super().__init__()
        self.output_size = output_size

    def forward(self, x):
        return torch.nn.functional.adaptive_avg_pool3d(x, self.output_size)

class LNSMaxPool1d(LNSModule):
    """
    An LNS 1D maximum pooling layer that applies a
    1D maximum pooling operation over the input tensor.

    See also: :py:class:`torch.nn.MaxPool1d`

    Parameters
    ----------
    kernel_size : int or tuple
        The size of the window to take the average over.
    stride : int or tuple, optional
        The stride of the window. Default is equal to `kernel_size`.
    padding : int or tuple, optional
        Implicit zero padding to be added on both sides of the input. Default is 0.
    dilation : int or tuple, optional
        The spacing between kernel elements. Default is 1.
    return_indices : bool, optional
        If True, will return the indices of the maximum values along with the output.
        Default is False.
    ceil_mode : bool, optional
        If True, will use ceil instead of floor to compute the output shape.
        Default is False.
    """

    def __init__(
            self,
            kernel_size: int,
            stride: Optional[int] = None,
            padding: int = 0,
            dilation: int = 1,
            return_indices: bool = False,
            ceil_mode: bool = False,
        ):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.return_indices = return_indices
        self.ceil_mode = ceil_mode

    def forward(self, x):
        return torch.nn.functional.max_pool1d(
            x, self.kernel_size, self.stride, self.padding,
            self.dilation, self.ceil_mode, self.return_indices
        )

class LNSMaxPool2d(LNSModule):
    """
    An LNS 2D maximum pooling layer that applies a
    2D maximum pooling operation over the input tensor.

    See also: :py:class:`torch.nn.MaxPool2d`

    Parameters
    ----------
    kernel_size : int or tuple
        The size of the window to take the average over.
    stride : int or tuple, optional
        The stride of the window. Default is equal to `kernel_size`.
    padding : int or tuple, optional
        Implicit zero padding to be added on both sides of the input. Default is 0.
    dilation : int or tuple, optional
        The spacing between kernel elements. Default is 1.
    return_indices : bool, optional
        If True, will return the indices of the maximum values along with the output.
        Default is False.
    ceil_mode : bool, optional
        If True, will use ceil instead of floor to compute the output shape.
        Default is False.
    """

    def __init__(
            self,
            kernel_size: Union[int, Tuple[int, int]],
            stride: Optional[Union[int, Tuple[int, int]]] = None,
            padding: Union[int, Tuple[int, int]] = 0,
            dilation: Union[int, Tuple[int, int]] = 1,
            return_indices: bool = False,
            ceil_mode: bool = False,
        ):
        super().__init__()
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.return_indices = return_indices
        self.ceil_mode = ceil_mode

    def forward(self, x):
        return torch.nn.functional.max_pool2d(
            x, self.kernel_size, self.stride, self.padding,
            self.dilation, self.ceil_mode, self.return_indices
        )

class LNSMaxPool3d(LNSModule):
    """
    An LNS 3D maximum pooling layer that applies a
    3D maximum pooling operation over the input tensor.

    See also: :py:class:`torch.nn.MaxPool3d`

    Parameters
    ----------
    kernel_size : int or tuple
        The size of the window to take the average over.
    stride : int or tuple, optional
        The stride of the window. Default is equal to `kernel_size`.
    padding : int or tuple, optional
        Implicit zero padding to be added on both sides of the input. Default is 0.
    dilation : int or tuple, optional
        The spacing between kernel elements. Default is 1.
    return_indices : bool, optional
        If True, will return the indices of the maximum values along with the output.
        Default is False.
    ceil_mode : bool, optional
        If True, will use ceil instead of floor to compute the output shape.
        Default is False.
    """

    def __init__(
            self,
            kernel_size: Union[int, Tuple[int, int, int]],
            stride: Optional[Union[int, Tuple[int, int, int]]] = None,
            padding: Union[int, Tuple[int, int, int]] = 0,
            dilation: Union[int, Tuple[int, int, int]] = 1,
            return_indices: bool = False,
            ceil_mode: bool = False,
        ):
        super().__init__()
        self.kernel_size = _triple(kernel_size)
        self.stride = _triple(stride)
        self.padding = _triple(padding)
        self.dilation = _triple(dilation)
        self.return_indices = return_indices
        self.ceil_mode = ceil_mode

    def forward(self, x):
        return torch.nn.functional.max_pool3d(
            x, self.kernel_size, self.stride, self.padding,
            self.dilation, self.ceil_mode, self.return_indices
        )
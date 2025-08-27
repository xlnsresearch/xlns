from typing import Tuple, Union, Optional
import collections
from itertools import repeat

import torch
from xlnstorch import rand
from . import LNSModule

def _pair(x):
    if isinstance(x,collections.abc.Iterable):
        return tuple(x)
    return (x, x)

def _triple(x):
    if isinstance(x, collections.abc.Iterable):
        return tuple(x)
    return (x, x, x)

class LNSConv1d(LNSModule):
    r"""
    An LNS 1D convolutional layer that applies a 1D convolution over the input tensor.

    See also: :py:class:`torch.nn.Conv1d`

    Parameters
    ----------
    in_channels : int
        Number of channels in the input tensor.
    out_channels : int
        Number of channels produced by the convolution.
    kernel_size : int or tuple
        Size of the convolving kernel.
    stride : int or tuple, optional
        Stride of the convolution. Default is 1.
    padding : int or tuple, optional
        Zero-padding added to both sides of the input. Default is 0.
    dilation : int or tuple, optional
        Spacing between kernel elements. Default is 1.
    groups : int, optional
        Number of blocked connections from input channels to output channels. Default is 1.
    bias : bool, optional
        If True, adds a learnable bias to the output. Default is True.
    padding_mode : str, optional
        Type of padding to use. Can be 'zeros', 'reflect', 'replicate',
        or 'circular'. Default is 'zeros'.
    device : torch.device, optional
        The device on which to create the layer's parameters. If None, uses the default device.
    weight_f : int, optional
        The number of fractional exponent bits for the weight. mutually exclusive with ``weight_b``.
    weight_b : Optional[Union[float, int, torch.Tensor]], optional
        The explicit logarithm base for the weight; mutually exclusive with ``weight_f``.
    bias_f : int, optional
        The number of fractional exponent bits for the bias. mutually exclusive with ``bias_b``.
    bias_b : Optional[Union[float, int, torch.Tensor]], optional
        The explicit logarithm base for the bias; mutually exclusive with ``bias_f``.

    Attributes
    ----------
    weight : LNSTensor
        The weight tensor of shape
        :math:`(\text{out_channels}, \frac{\text{in_channels}}{\text{groups}}, \text{kernel_size})`,
        initialized with random values uniformly distributed between
        :math:`-\sqrt{k}` and :math:`\sqrt{k}`,
        where :math:`k = \frac{\text{groups}}{\text{in_channels} \times \text{kernel_size}}`.
    bias : LNSTensor or None
        The bias vector of shape :math:`(\text{out_channels},)`,
        initialized with random values uniformly distributed between
        :math:`-\sqrt{k}` and :math:`\sqrt{k}`,
        where :math:`k = \frac{\text{groups}}{\text{in_channels} \times \text{kernel_size}}`.
        This is only created if `bias` is set to True.
    """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            stride: int = 1,
            padding: int = 0,
            dilation: int = 1,
            groups: int = 1,
            bias: bool = True,
            padding_mode: str = 'zeros',
            device=None,
            weight_f=None,
            weight_b=None,
            bias_f=None,
            bias_b=None,
        ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.has_bias = bias
        self.padding_mode = padding_mode
        self.device = device

        sqrt_k = (groups / (in_channels * kernel_size)) ** 0.5
        weight = rand(out_channels, in_channels // groups, kernel_size, device=device, f=weight_f, b=weight_b)
        self.register_parameter("weight", (weight * 2 - 1) * sqrt_k)

        if self.has_bias:
            bias = rand(out_channels, device=device, f=bias_f, b=bias_b)
            self.register_parameter("bias", (bias * 2 - 1) * sqrt_k)
        else:
            self.bias = None

    def forward(self, x):
        return torch.nn.functional.conv1d(
            x,
            self.weight,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )

class LNSConv2d(LNSModule):
    r"""
    An LNS 2D convolutional layer that applies a 2D convolution over the input tensor.

    See also: :py:class:`torch.nn.Conv2d`

    Parameters
    ----------
    in_channels : int
        Number of channels in the input tensor.
    out_channels : int
        Number of channels produced by the convolution.
    kernel_size : int or tuple
        Size of the convolving kernel.
    stride : int or tuple, optional
        Stride of the convolution. Default is 1.
    padding : int or tuple, optional
        Zero-padding added to both sides of the input. Default is 0.
    dilation : int or tuple, optional
        Spacing between kernel elements. Default is 1.
    groups : int, optional
        Number of blocked connections from input channels to output channels. Default is 1.
    bias : bool, optional
        If True, adds a learnable bias to the output. Default is True.
    padding_mode : str, optional
        Type of padding to use. Can be 'zeros', 'reflect', 'replicate',
        or 'circular'. Default is 'zeros'.
    device : torch.device, optional
        The device on which to create the layer's parameters. If None, uses the default device.
    weight_f : int, optional
        The number of fractional exponent bits for the weight. mutually exclusive with ``weight_b``.
    weight_b : Optional[Union[float, int, torch.Tensor]], optional
        The explicit logarithm base for the weight; mutually exclusive with ``weight_f``.
    bias_f : int, optional
        The number of fractional exponent bits for the bias. mutually exclusive with ``bias_b``.
    bias_b : Optional[Union[float, int, torch.Tensor]], optional
        The explicit logarithm base for the bias; mutually exclusive with ``bias_f``.

    Attributes
    ----------
    weight : LNSTensor
        The weight tensor of shape
        :math:`(\text{out_channels}, \frac{\text{in_channels}}{\text{groups}}, \text{kernel_size[0]}, \text{kernel_size[1]})`,
        initialized with random values uniformly distributed between
        :math:`-\sqrt{k}` and :math:`\sqrt{k}`,
        where :math:`k = \frac{\text{groups}}{\text{in_channels} \times \prod_{i=0}^{1} \text{kernel_size[{i}]}}`.
    bias : LNSTensor or None
        The bias vector of shape :math:`(\text{out_channels},)`,
        initialized with random values uniformly distributed between
        :math:`-\sqrt{k}` and :math:`\sqrt{k}`,
        where :math:`k = \frac{\text{groups}}{\text{in_channels} \times \prod_{i=0}^{1} \text{kernel_size[{i}]}}`.
        This is only created if `bias` is set to True.
    """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: Union[int, Tuple[int, int]],
            stride: Union[int, Tuple[int, int]] = 1,
            padding: Union[int, Tuple[int, int]] = 0,
            dilation: Union[int, Tuple[int, int]] = 1,
            groups: Union[int, Tuple[int, int]] = 1,
            bias: bool = True,
            padding_mode: str = 'zeros',
            device=None,
            weight_f=None,
            weight_b=None,
            bias_f=None,
            bias_b=None,
        ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups
        self.has_bias = bias
        self.padding_mode = padding_mode
        self.device = device

        sqrt_k = (groups / (in_channels * self.kernel_size[0] * self.kernel_size[1])) ** 0.5
        weight = rand(out_channels, in_channels // groups, self.kernel_size[0],
                      self.kernel_size[1], device=device, f=weight_f, b=weight_b)
        self.register_parameter("weight", (weight * 2 - 1) * sqrt_k)

        if self.has_bias:
            bias = rand(out_channels, device=device, f=bias_f, b=bias_b)
            self.register_parameter("bias", (bias * 2 - 1) * sqrt_k)
        else:
            self.bias = None

    def forward(self, x):
        return torch.nn.functional.conv2d(
            x,
            self.weight,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )

class LNSConv3d(LNSModule):
    r"""
    An LNS 3D convolutional layer that applies a 3D convolution over the input tensor.

    See also: :py:class:`torch.nn.Conv3d`

    Parameters
    ----------
    in_channels : int
        Number of channels in the input tensor.
    out_channels : int
        Number of channels produced by the convolution.
    kernel_size : int or tuple
        Size of the convolving kernel.
    stride : int or tuple, optional
        Stride of the convolution. Default is 1.
    padding : int or tuple, optional
        Zero-padding added to both sides of the input. Default is 0.
    dilation : int or tuple, optional
        Spacing between kernel elements. Default is 1.
    groups : int, optional
        Number of blocked connections from input channels to output channels. Default is 1.
    bias : bool, optional
        If True, adds a learnable bias to the output. Default is True.
    padding_mode : str, optional
        Type of padding to use. Can be 'zeros', 'reflect', 'replicate',
        or 'circular'. Default is 'zeros'.
    device : torch.device, optional
        The device on which to create the layer's parameters. If None, uses the default device.
    weight_f : int, optional
        The number of fractional exponent bits for the weight. mutually exclusive with ``weight_b``.
    weight_b : Optional[Union[float, int, torch.Tensor]], optional
        The explicit logarithm base for the weight; mutually exclusive with ``weight_f``.
    bias_f : int, optional
        The number of fractional exponent bits for the bias. mutually exclusive with ``bias_b``.
    bias_b : Optional[Union[float, int, torch.Tensor]], optional
        The explicit logarithm base for the bias; mutually exclusive with ``bias_f``.

    Attributes
    ----------
    weight : LNSTensor
        The weight tensor of shape
        :math:`(\text{out_channels}, \frac{\text{in_channels}}{\text{groups}},
        \text{kernel_size[0]}, \text{kernel_size[1]}, \text{kernel_size[2]})`,
        initialized with random values uniformly distributed between
        :math:`-\sqrt{k}` and :math:`\sqrt{k}`,
        where :math:`k = \frac{\text{groups}}{\text{in_channels} \times \prod_{i=0}^{2} \text{kernel_size[{i}]}}`.
    bias : LNSTensor or None
        The bias vector of shape :math:`(\text{out_channels},)`,
        initialized with random values uniformly distributed between
        :math:`-\sqrt{k}` and :math:`\sqrt{k}`,
        where :math:`k = \frac{\text{groups}}{\text{in_channels} \times \prod_{i=0}^{2} \text{kernel_size[{i}]}}`.
        This is only created if `bias` is set to True.
    """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: Union[int, Tuple[int, int, int]],
            stride: Union[int, Tuple[int, int, int]] = 1,
            padding: Union[int, Tuple[int, int, int]] = 0,
            dilation: Union[int, Tuple[int, int, int]] = 1,
            groups: Union[int, Tuple[int, int, int]] = 1,
            bias: bool = True,
            padding_mode: str = 'zeros',
            device=None,
            weight_f=None,
            weight_b=None,
            bias_f=None,
            bias_b=None,
        ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _triple(kernel_size)
        self.stride = _triple(stride)
        self.padding = _triple(padding)
        self.dilation = _triple(dilation)
        self.groups = groups
        self.has_bias = bias
        self.padding_mode = padding_mode
        self.device = device

        sqrt_k = (groups / (in_channels * self.kernel_size[0] * self.kernel_size[1] * self.kernel_size[2])) ** 0.5
        weight = rand(out_channels, in_channels // groups, self.kernel_size[0], self.kernel_size[1],
                      self.kernel_size[2], device=device, f=weight_f, b=weight_b)
        self.register_parameter("weight", (weight * 2 - 1) * sqrt_k)

        if self.has_bias:
            bias = rand(out_channels, device=device, f=bias_f, b=bias_b)
            self.register_parameter("bias", (bias * 2 - 1) * sqrt_k)
        else:
            self.bias = None

    def forward(self, x):
        return torch.nn.functional.conv3d(
            x,
            self.weight,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )
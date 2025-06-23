import torch
from . import LNSModule
from .. import rand

class LNSConv1d(LNSModule):
    """
    An LNS 1D convolutional layer that applies a 1D convolution over the input tensor.

    See also: `torch.nn.Conv1d`

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
            device=None):
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
        self.register_parameter("weight", (rand(out_channels, in_channels / groups, kernel_size, device=device) * 2 - 1) * sqrt_k)

        if self.has_bias:
            self.register_parameter("bias", (rand(out_channels, device=device) * 2 - 1) * sqrt_k)
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
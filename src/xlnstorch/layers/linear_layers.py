import torch
from . import LNSModule
from .. import rand

class LNSIdentity(LNSModule):
    """
    An LNS identity layer that does not change the input. This is not strictly
    necessary, since there are no trainable parameters, but it is included for
    completeness.

    See also: :py:class:`torch.nn.Identity`
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, input):
        return input

class LNSLinear(LNSModule):
    r"""
    An LNS linear layer that performs a linear transformation on the input, :math:`x`.
    It applies the affine linear transformation :math:`y = xA^T + b`, where :math:`A`
    is the weight matrix, and :math:`b` is the bias vector if bias is enabled.

    See also: :py:class:`torch.nn.Linear`

    Parameters
    ----------
    in_features : int
        The number of input features.
    out_features : int
        The number of output features.
    bias : bool, optional
        Whether to include a bias term in the transformation. Default is True.
    device : torch.device, optional
        The device on which to create the layer's parameters. If None, defaults to the current.

    Attributes
    ---------
    weight : LNSTensor
        The weight matrix of shape :math:`(\text{out_features}, \text{in_features})`,
        initialized with random values uniformly distributed between
        :math:`-\sqrt{k}` and :math:`\sqrt{k}`,
        where :math:`k = \frac{1}{\text{in_features}}`.
    bias : LNSTensor
        The bias vector of shape :math:`(\text{out_features},)`,
        initialized with random values uniformly distributed between
        :math:`-\sqrt{k}` and :math:`\sqrt{k}`,
        where :math:`k = \frac{1}{\text{in_features}}`.
        This is only created if `bias` is set to True.
    """

    def __init__(self, in_features, out_features, bias=True, device=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.has_bias = bias

        sqrt_k = (1.0 / in_features) ** 0.5
        self.register_parameter("weight", (rand(out_features, in_features, device=device) * 2 - 1) * sqrt_k)

        if self.has_bias:
            self.register_parameter("bias", (rand(out_features, device=device) * 2 - 1) * sqrt_k)
        else:
            self.bias = None

    def forward(self, x):
        return torch.nn.functional.linear(x, self.weight, self.bias)

class LNSBilinear(LNSModule):
    r"""
    An LNS bilinear layer that performs a bilinear transformation on two inputs, :math:`x_1`
    and :math:`x_2`. It applies the transformation :math:`y = x_1^T A x_2 + b`, where :math:`A`
    is the weight tensor, and :math:`b` is the bias vector if bias is enabled.

    See also: :py:class:`torch.nn.Bilinear`

    Parameters
    ----------
    in1_features : int
        The number of input features for the first input.
    in2_features : int
        The number of input features for the second input.
    out_features : int
        The number of output features.
    bias : bool, optional
        Whether to include a bias term in the transformation. Default is True.
    device : torch.device, optional
        The device on which to create the layer's parameters. If None, defaults to the current
        device.

    Attributes
    ---------
    weight : LNSTensor
        The weight tensor of shape
        :math:`(\text{out_features}, \text{in1_features}, \text{in2_features})`,
        initialized with random values uniformly distributed between
        :math:`-\sqrt{k}` and :math:`\sqrt{k}`,
        where :math:`k = \frac{1}{\text{in1_features}}`.
    bias : LNSTensor
        The bias vector of shape :math:`(\text{out_features},)`,
        initialized with random values uniformly distributed between
        :math:`-\sqrt{k}` and :math:`\sqrt{k}`,
        where :math:`k = \frac{1}{\text{in1_features}}`.
        This is only created if `bias` is set to True.
    """

    def __init__(self, in1_features, in2_features, out_features, bias=True, device=None):
        super().__init__()
        self.in1_features = in1_features
        self.in2_features = in2_features
        self.out_features = out_features
        self.has_bias = bias

        sqrt_k = (1.0 / (in1_features + in2_features)) ** 0.5
        self.register_parameter("weight", (rand(out_features, in1_features, in2_features, device=device) * 2 - 1) * sqrt_k)

        if self.has_bias:
            self.register_parameter("bias", (rand(out_features, device=device) * 2 - 1) * sqrt_k)
        else:
            self.bias = None

    def forward(self, x1, x2):
        return torch.nn.functional.bilinear(x1, x2, self.weight, self.bias)

class LNSLazyLinear(LNSModule):
    r"""
    An LNS lazy linear layer that performs a linear transformation on the input, :math:`x`,
    without initializing the weight and bias parameters until the first forward pass.
    The `in_features` argument is inferred from the ``input.shape[-1]`` during the first
    call.

    See also: :py:class:`torch.nn.LazyLinear`

    Parameters
    ----------
    out_features : int
        The number of output features.
    bias : bool, optional
        Whether to include a bias term in the transformation. Default is True.
    device : torch.device, optional
        The device on which to create the layer's parameters. If None, defaults to the current
        device.

    Attributes
    ---------
    weight : LNSTensor
        The weight matrix of shape :math:`(\text{out_features}, \text{in_features})`,
        initialized with random values uniformly distributed between
        :math:`-\sqrt{k}` and :math:`\sqrt{k}`,
        where :math:`k = \frac{1}{\text{in_features}}`.
    bias : LNSTensor
        The bias vector of shape :math:`(\text{out_features},)`,
        initialized with random values uniformly distributed between
        :math:`-\sqrt{k}` and :math:`\sqrt{k}`,
        where :math:`k = \frac{1}{\text{in_features}}`.
        This is only created if `bias` is set to True.
    """

    def __init__(self, out_features, bias=True, device=None):
        super().__init__()
        self.out_features = out_features
        self.has_bias = bias
        self.device = device
        self.weight = None
        self.bias = None

    def forward(self, x):
        if self.weight is None:
            self.in_features = x.shape[-1]
            sqrt_k = (1.0 / self.in_features) ** 0.5

            self.register_parameter("weight", (rand(self.out_features, self.in_features, device=self.device) * 2 - 1) * sqrt_k)
            if self.has_bias:
                self.register_parameter("bias", (rand(self.out_features, device=self.device) * 2 - 1) * sqrt_k)

        return torch.nn.functional.linear(x, self.weight, self.bias)
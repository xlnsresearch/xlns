from typing import Union, Optional, Tuple
import torch
import xlnstorch
from . import LNSModule

class _BatchNorm(LNSModule):

    def __init__(
            self,
            num_features: int,
            eps: float = 1e-5,
            momentum: float = 0.1,
            affine: bool = True,
            track_running_stats: bool = True,
            weight_f: int = None,
            weight_b: float = None,
            bias_f: int = None,
            bias_b: float = None,
            running_mean_f: int = None,
            running_mean_b: float = None,
            running_var_f: int = None,
            running_var_b: float = None,
        ):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats

        if self.affine:
            weight = xlnstorch.ones(num_features, f=weight_f, b=weight_b)
            self.register_parameter("weight", weight)

            bias = xlnstorch.zeros(num_features, f=bias_f, b=bias_b)
            self.register_parameter("bias", bias)

        else:
            self.weight = None
            self.bias = None

        if self.track_running_stats:
            self.running_mean = xlnstorch.zeros(num_features, f=running_mean_f, b=running_mean_b)
            self.running_var = xlnstorch.ones(num_features, f=running_var_f, b=running_var_b)
            self.num_batches_tracked = xlnstorch.lnstensor(0)

        else:
            self.running_mean = None
            self.running_var = None
            self.num_batches_tracked = None

    def _check_input_dim(self, input):
        raise NotImplementedError

    def forward(self, x):
        self._check_input_dim(x)

        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:
                    exponential_average_factor = 1.0 / self.num_batches_tracked
                else:
                    exponential_average_factor = self.momentum

        if self.training:
            bn_training = True
        else:
            bn_training = (self.running_mean is None) and (self.running_var is None)

        return torch.nn.functional.batch_norm(
            x,
            self.running_mean if not self.training or self.track_running_stats else None,
            self.running_var if not self.training or self.track_running_stats else None,
            self.weight,
            self.bias,
            bn_training,
            exponential_average_factor,
            self.eps,
        )

class LNSBatchNorm1d(_BatchNorm):
    r"""
    An LNS 1D normalization layer that applies a 1D
    batch normalization over the input tensor.

    See also: :py:class:`torch.nn.BatchNorm1d`

    Parameters
    ----------
    num_features : int
        C from an expected input of size (N, C, L) or (N, C)
    eps : float, LNSTensor, optional
        A value added to the denominator for numerical stability. Default: 1e-5
    momentum : float, LNSTensor, optional
        The value used for the running_mean and running_var computation.
        Can be set to None for cumulative moving average (i.e. simple average).
        Default: 0.1
    affine : bool, optional
        If True, this module has learnable affine parameters. Default: True
    track_running_stats : bool, optional
        If True, this module tracks the running mean and variance,
        otherwise, it does not track such statistics and always uses batch
        statistics in both training and eval modes. Default: True.
    weight_f : int, optional
        The number of fractional exponent bits for the weight. mutually exclusive with ``weight_b``.
    weight_b : Optional[Union[float, int, torch.Tensor]], optional
        The explicit logarithm base for the weight; mutually exclusive with ``weight_f``.
    bias_f : int, optional
        The number of fractional exponent bits for the bias. mutually exclusive with ``bias_b``.
    bias_b : Optional[Union[float, int, torch.Tensor]], optional
        The explicit logarithm base for the bias; mutually exclusive with ``bias_f``.
    running_mean_f : int, optional
        The number of fractional exponent bits for the running mean. mutually exclusive with ``running_mean_b``.
    running_mean_b : Optional[Union[float, int, torch.Tensor]], optional
        The explicit logarithm base for the running mean; mutually exclusive with ``running_mean_f``.
    running_var_f : int, optional
        The number of fractional exponent bits for the running variance. mutually exclusive with ``running_var_b``.
    running_var_b : Optional[Union[float, int, torch.Tensor]], optional
        The explicit logarithm base for the running variance; mutually exclusive with ``running_var_f``

    Attributes
    ---------
    weight : LNSTensor
        The learnable weight of shape :math:`(\text{num_features},)`. Initialized to ones.
    bias : LNSTensor
        The learnable bias of shape :math:`(\text{num_features},)`. Initialized to zeros.
    running_mean : LNSTensor
        The running mean of shape :math:`(\text{num_features},)`. Initialized to zeros.
    running_var : LNSTensor
        The running variance of shape :math:`(\text{num_features},)`. Initialized to ones.
    num_batches_tracked : LNSTensor
        The number of batches tracked. Initialized to zero.
    """

    def _check_input_dim(self, input):
        if input.dim() != 2 and input.dim() != 3:
            raise ValueError("expected 2D or 3D input (got {}D input)".format(input.dim()))

class LNSBatchNorm2d(_BatchNorm):
    r"""
    An LNS 2D normalization layer that applies a 2D
    batch normalization over the input tensor.

    See also: :py:class:`torch.nn.BatchNorm2d`

    Parameters
    ----------
    num_features : int
        C from an expected input of size (N, C, H, W)
    eps : float, LNSTensor, optional
        A value added to the denominator for numerical stability. Default: 1e-5
    momentum : float, LNSTensor, optional
        The value used for the running_mean and running_var computation.
        Can be set to None for cumulative moving average (i.e. simple average).
        Default: 0.1
    affine : bool, optional
        If True, this module has learnable affine parameters. Default: True
    track_running_stats : bool, optional
        If True, this module tracks the running mean and variance,
        otherwise, it does not track such statistics and always uses batch
        statistics in both training and eval modes. Default: True.
    weight_f : int, optional
        The number of fractional exponent bits for the weight. mutually exclusive with ``weight_b``.
    weight_b : Optional[Union[float, int, torch.Tensor]], optional
        The explicit logarithm base for the weight; mutually exclusive with ``weight_f``.
    bias_f : int, optional
        The number of fractional exponent bits for the bias. mutually exclusive with ``bias_b``.
    bias_b : Optional[Union[float, int, torch.Tensor]], optional
        The explicit logarithm base for the bias; mutually exclusive with ``bias_f``.
    running_mean_f : int, optional
        The number of fractional exponent bits for the running mean. mutually exclusive with ``running_mean_b``.
    running_mean_b : Optional[Union[float, int, torch.Tensor]], optional
        The explicit logarithm base for the running mean; mutually exclusive with ``running_mean_f``.
    running_var_f : int, optional
        The number of fractional exponent bits for the running variance. mutually exclusive with ``running_var_b``.
    running_var_b : Optional[Union[float, int, torch.Tensor]], optional
        The explicit logarithm base for the running variance; mutually exclusive with ``running_var_f``

    Attributes
    ---------
    weight : LNSTensor
        The learnable weight of shape :math:`(\text{num_features},)`. Initialized to ones.
    bias : LNSTensor
        The learnable bias of shape :math:`(\text{num_features},)`. Initialized to zeros.
    running_mean : LNSTensor
        The running mean of shape :math:`(\text{num_features},)`. Initialized to zeros.
    running_var : LNSTensor
        The running variance of shape :math:`(\text{num_features},)`. Initialized to ones.
    num_batches_tracked : LNSTensor
        The number of batches tracked. Initialized to zero.
    """

    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError("expected 4D input (got {}D input)".format(input.dim()))

class LNSBatchNorm3d(_BatchNorm):
    r"""
    An LNS 1D normalization layer that applies a 1D
    batch normalization over the input tensor.

    See also: :py:class:`torch.nn.BatchNorm1d`

    Parameters
    ----------
    num_features : int
        C from an expected input of size (N, C, D, H, W)
    eps : float, LNSTensor, optional
        A value added to the denominator for numerical stability. Default: 1e-5
    momentum : float, LNSTensor, optional
        The value used for the running_mean and running_var computation.
        Can be set to None for cumulative moving average (i.e. simple average).
        Default: 0.1
    affine : bool, optional
        If True, this module has learnable affine parameters. Default: True
    track_running_stats : bool, optional
        If True, this module tracks the running mean and variance,
        otherwise, it does not track such statistics and always uses batch
        statistics in both training and eval modes. Default: True.
    weight_f : int, optional
        The number of fractional exponent bits for the weight. mutually exclusive with ``weight_b``.
    weight_b : Optional[Union[float, int, torch.Tensor]], optional
        The explicit logarithm base for the weight; mutually exclusive with ``weight_f``.
    bias_f : int, optional
        The number of fractional exponent bits for the bias. mutually exclusive with ``bias_b``.
    bias_b : Optional[Union[float, int, torch.Tensor]], optional
        The explicit logarithm base for the bias; mutually exclusive with ``bias_f``.
    running_mean_f : int, optional
        The number of fractional exponent bits for the running mean. mutually exclusive with ``running_mean_b``.
    running_mean_b : Optional[Union[float, int, torch.Tensor]], optional
        The explicit logarithm base for the running mean; mutually exclusive with ``running_mean_f``.
    running_var_f : int, optional
        The number of fractional exponent bits for the running variance. mutually exclusive with ``running_var_b``.
    running_var_b : Optional[Union[float, int, torch.Tensor]], optional
        The explicit logarithm base for the running variance; mutually exclusive with ``running_var_f``

    Attributes
    ---------
    weight : LNSTensor
        The learnable weight of shape :math:`(\text{num_features},)`. Initialized to ones.
    bias : LNSTensor
        The learnable bias of shape :math:`(\text{num_features},)`. Initialized to zeros.
    running_mean : LNSTensor
        The running mean of shape :math:`(\text{num_features},)`. Initialized to zeros.
    running_var : LNSTensor
        The running variance of shape :math:`(\text{num_features},)`. Initialized to ones.
    num_batches_tracked : LNSTensor
        The number of batches tracked. Initialized to zero.
    """

    def _check_input_dim(self, input):
        if input.dim() != 5:
            raise ValueError("expected 5D input (got {}D input)".format(input.dim()))

class LNSLayerNorm(LNSModule):
    r"""
    Applies Layer Normalization over a mini-batch of inputs.

    See also: :py:class:`torch.nn.LayerNorm`

    Parameters
    ----------
    normalized_shape : int or tuple of int
        Input shape from an expected input of size
        :math:`(N, *)` where `*` means any number of additional
        dimensions. If a single integer is used, it is treated as a singleton
        tuple.
    eps : float, LNSTensor, optional
        A value added to the denominator for numerical stability. Default: 1e-5.
    elementwise_affine : bool, optional
        If True, this module has learnable per-element affine parameters
        initialized to ones (for weights) and zeros (for biases). Default: True.
    bias : bool, optional
        If True, this module has learnable bias parameters. Default: True.
    weight_f : int, optional
        The number of fractional exponent bits for the weight. mutually exclusive with ``weight_b``.
    weight_b : float, int, torch.Tensor, optional
        The explicit logarithm base for the weight; mutually exclusive with ``weight_f``.
    bias_f : int, optional
        The number of fractional exponent bits for the bias. mutually exclusive with ``bias_b``.
    bias_b : float, int, torch.Tensor, optional
        The explicit logarithm base for the bias; mutually exclusive with ``bias_f``.

    Attributes
    ---------
    weight : LNSTensor
        The learnable weight of shape :math:`(\text{normalized\_shape},)`. Initialized to ones.
    bias : LNSTensor
        The learnable bias of shape :math:`(\text{normalized\_shape},)`. Initialized to zeros.
    """

    def __init__(
            self,
            normalized_shape: Union[int, tuple[int]],
            eps: float = 1e-5,
            elementwise_affine: bool = True,
            bias: bool = True,
            weight_f: int = None,
            weight_b: float = None,
            bias_f: int = None,
            bias_b: float = None,
        ):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if self.elementwise_affine:
            weight = xlnstorch.ones(*self.normalized_shape, f=weight_f, b=weight_b)
            self.register_parameter("weight", weight)

            if bias:
                bias = xlnstorch.zeros(*self.normalized_shape, f=bias_f, b=bias_b)
                self.register_parameter("bias", bias)

            else:
                self.bias = None

        else:
            self.weight = None
            self.bias = None

    def forward(self, x):
        return torch.nn.functional.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
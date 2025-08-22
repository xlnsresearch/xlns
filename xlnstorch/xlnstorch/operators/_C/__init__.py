from . import addition_ops
from . import convolution_ops
from . import pooling_ops

import torch

# this list of operators using c++ implementations is
# useful to be able to switch between c++ and pure python
# implementations for testing and benchmarking. Each entry
# maps a torch operator to a tuple of the keys for the
# python implementation and c++ implementation respectively.
CPP_IMPLEMENTED_OPERATORS = {
    torch.add: ("default", "default_cpp"),
    torch.sum: ("default", "default_cpp"),
    torch.matmul: ("default", "default_cpp"),
    torch.nn.functional.conv1d: ("default", "default_cpp"),
    torch.nn.functional.conv2d: ("default", "default_cpp"),
    torch.nn.functional.conv3d: ("default", "default_cpp"),
    torch.nn.functional.avg_pool1d: ("default", "default_cpp"),
}

__all__ = [
    "CPP_IMPLEMENTED_OPERATORS",
]
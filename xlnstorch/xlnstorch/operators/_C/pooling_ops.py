import torch
import xlnstorch.csrc
from xlnstorch import lnstensor, format_lnstensor_operands, implements, CSRC_AVAILABLE
from xlnstorch.autograd import LNSFunction

class LNSAvgPool1dCPPFunction(LNSFunction):

    @staticmethod
    def forward(x, kernel_size, base, stride=None, padding=0, ceil_mode=False, count_include_pad=True):
        x_packed = x.to(torch.int64)

        return xlnstorch.csrc.avg_pool1d_forward(x_packed, kernel_size, base, stride,
                                                 padding, ceil_mode, count_include_pad).to(torch.float64)

    @staticmethod
    def setup_context(ctx, inputs, output):
        pass

@implements(torch.nn.functional.avg_pool1d, LNSAvgPool1dCPPFunction.forward, "default_cpp", default=CSRC_AVAILABLE)
def avg_pool1d(x, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True):

    kernel_size = kernel_size[0] if isinstance(kernel_size, (list, tuple)) else kernel_size
    stride = stride[0] if isinstance(stride, (list, tuple)) else stride
    padding = padding[0] if isinstance(padding, (list, tuple)) else padding

    result = LNSAvgPool1dCPPFunction.apply(x, kernel_size, x.base, stride,
                                           padding, ceil_mode, count_include_pad)

    return lnstensor(result, from_lns=True, b=x.base)
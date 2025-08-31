import torch
import xlnstorch.csrc
from xlnstorch import lnstensor, implements, CSRC_AVAILABLE
from xlnstorch.autograd import LNSFunction

def _avg_pool1d_cpp(ops, x, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True):
    return xlnstorch.csrc.avg_pool1d_forward(x, kernel_size, ops.base, stride,
                                             padding, ceil_mode, count_include_pad)

class LNSAvgPool1dCPPFunction(LNSFunction):

    @staticmethod
    def forward(ops, x, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True):
        x = x.view(torch.int64)
        result = _avg_pool1d_cpp(ops, x, kernel_size, stride,
                                 padding, ceil_mode, count_include_pad)
        return result.view(torch.float64)

    @staticmethod
    def setup_context(ctx, ops, inputs, output):
        x, kernel_size, stride, padding, ceil_mode, count_include_pad = inputs
        if stride is None:
            stride = kernel_size
        ctx.save_for_backward(x)
        ctx.kernel_size = kernel_size
        ctx.stride = stride
        ctx.padding = padding
        ctx.ceil_mode = ceil_mode
        ctx.count_include_pad = count_include_pad

    @staticmethod
    def backward(ctx, ops, grad_output):
        x, = ctx.saved_tensors
        kernel_size = ctx.kernel_size
        stride = ctx.stride
        padding = ctx.padding
        ceil_mode = ctx.ceil_mode
        count_include_pad = ctx.count_include_pad

        x, grad_output = x.view(torch.int64), grad_output.view(torch.int64)

        grad_x = xlnstorch.csrc.avg_pool1d_backward(grad_output, x, kernel_size, ops.base, stride,
                                                    padding, ceil_mode, count_include_pad).view(torch.float64)

        return grad_x.view(torch.float64), None, None, None, None, None

@implements(torch.nn.functional.avg_pool1d, _avg_pool1d_cpp, "default_cpp", default=CSRC_AVAILABLE)
def avg_pool1d(x, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True):

    kernel_size = kernel_size[0] if isinstance(kernel_size, (list, tuple)) else kernel_size
    stride = stride[0] if isinstance(stride, (list, tuple)) else stride
    padding = padding[0] if isinstance(padding, (list, tuple)) else padding

    result = LNSAvgPool1dCPPFunction.apply(x, kernel_size, stride, padding,
                                           ceil_mode, count_include_pad)

    return lnstensor(result, from_lns=True, b=x.base)
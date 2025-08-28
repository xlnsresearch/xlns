import torch
import xlnstorch.csrc
from xlnstorch import lnstensor, implements, CSRC_AVAILABLE
from xlnstorch.autograd import LNSFunction

class LNSAvgPool1dCPPFunction(LNSFunction):

    @staticmethod
    def forward(x, kernel_size, base, stride=None, padding=0, ceil_mode=False, count_include_pad=True):
        x_packed = x.to(torch.int64)

        return xlnstorch.csrc.avg_pool1d_forward(x_packed, kernel_size, base, stride,
                                                 padding, ceil_mode, count_include_pad).to(torch.float64)

    @staticmethod
    def setup_context(ctx, inputs, output):
        x, kernel_size, base, stride, padding, ceil_mode, count_include_pad = inputs
        if stride is None:
            stride = kernel_size
        ctx.save_for_backward(x, base)
        ctx.kernel_size = kernel_size
        ctx.stride = stride
        ctx.padding = padding
        ctx.ceil_mode = ceil_mode
        ctx.count_include_pad = count_include_pad

    @staticmethod
    def backward(ctx, grad_output):
        x, base = ctx.saved_tensors
        kernel_size = ctx.kernel_size
        stride = ctx.stride
        padding = ctx.padding
        ceil_mode = ctx.ceil_mode
        count_include_pad = ctx.count_include_pad

        grad_output_packed = grad_output.to(torch.int64)
        x_packed = x.to(torch.int64)
        grad_x = xlnstorch.csrc.avg_pool1d_backward(grad_output_packed, x_packed, kernel_size, base,
                                                    stride, padding, ceil_mode, count_include_pad).to(torch.float64)

        return grad_x, None, None, None, None, None, None

@implements(torch.nn.functional.avg_pool1d, LNSAvgPool1dCPPFunction.forward, "default_cpp", default=CSRC_AVAILABLE)
def avg_pool1d(x, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True):

    kernel_size = kernel_size[0] if isinstance(kernel_size, (list, tuple)) else kernel_size
    stride = stride[0] if isinstance(stride, (list, tuple)) else stride
    padding = padding[0] if isinstance(padding, (list, tuple)) else padding

    result = LNSAvgPool1dCPPFunction.apply(x, kernel_size, x.base, stride,
                                           padding, ceil_mode, count_include_pad)

    return lnstensor(result, from_lns=True, b=x.base)
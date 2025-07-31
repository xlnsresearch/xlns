import torch
import xlnstorch._C
from xlnstorch import LNS_ZERO, lnstensor, format_lnstensor_operands, implements
from xlnstorch.autograd import LNSFunction

class LNSConv1dCPPFunction(LNSFunction):

    @staticmethod
    def forward(x, weight, bias, base, stride=1, padding=0, dilation=1, groups=1):
        x_packed = x.to(torch.int64)
        weight_packed = weight.to(torch.int64)
        bias_packed = torch.full((weight.size(0),), LNS_ZERO.item(), dtype=torch.int64) if bias is None else bias.to(torch.int64)

        return xlnstorch._C.conv1d_forward(x_packed, weight_packed, bias_packed,
                                           base, stride, padding, dilation, groups)

    @staticmethod
    def setup_context(ctx, inputs, output):
        pass

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None, None, None, None, None

@implements(torch.nn.functional.conv1d, LNSConv1dCPPFunction.forward, "default_cpp", default=True)
def conv1d(x, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):

    if bias is not None:
        x, weight, bias = format_lnstensor_operands(x, weight, bias)
    else:
        x, weight = format_lnstensor_operands(x, weight)

    result = LNSConv1dCPPFunction.apply(x, weight, bias, x.base, stride,
                                     padding, dilation, groups)

    return lnstensor(result, from_lns=True, b=x.base)
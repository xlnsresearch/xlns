import torch
import xlnstorch._C
from xlnstorch import LNS_ZERO, lnstensor, format_lnstensor_operands, implements
from xlnstorch.autograd import LNSFunction

class LNSConv1dCPPFunction(LNSFunction):

    @staticmethod
    def forward(x, weight, bias, base, stride=1, padding=0, dilation=1, groups=1):
        x_packed = x.to(torch.int64)
        weight_packed = weight.to(torch.int64)
        bias_packed = bias.to(torch.int64) if bias is not None else None

        return xlnstorch._C.conv1d_forward(x_packed, weight_packed, bias_packed,
                                           base, stride, padding, dilation, groups).to(torch.float64)

    @staticmethod
    def setup_context(ctx, inputs, output):
        x, weight, bias, base, stride, padding, dilation, groups = inputs
        ctx.save_for_backward(x, weight, base)
        ctx.bias_defined = bias is not None
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.groups = groups

    @staticmethod
    def backward(ctx, grad_output):
        x, weight, base = ctx.saved_tensors
        x_packed = x.to(torch.int64)
        weight_packed = weight.to(torch.int64)
        grad_packed = grad_output.to(torch.int64)

        grads = xlnstorch._C.conv1d_backward(
            grad_packed, x_packed, weight_packed, base, ctx.bias_defined,
            ctx.stride, ctx.padding, ctx.dilation, ctx.groups)

        if ctx.bias_defined:
            return (grads[0].to(torch.float64), grads[1].to(torch.float64), 
                    grads[2].to(torch.float64), None, None, None, None, None)

        return (grads[0].to(torch.float64), grads[1].to(torch.float64),
                None, None, None, None, None, None)

@implements(torch.nn.functional.conv1d, LNSConv1dCPPFunction.forward, "default_cpp", default=True)
def conv1d(x, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):

    if bias is not None:
        x, weight, bias = format_lnstensor_operands(x, weight, bias)
    else:
        x, weight = format_lnstensor_operands(x, weight)

    result = LNSConv1dCPPFunction.apply(x, weight, bias, x.base, stride,
                                     padding, dilation, groups)

    return lnstensor(result, from_lns=True, b=x.base)
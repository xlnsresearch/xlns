import torch
import xlnstorch.csrc
from xlnstorch import CSRC_AVAILABLE, lnstensor, format_lnstensor_operands, implements
from xlnstorch.autograd import LNSFunction

def _conv1d_cpp(ops, x, weight, bias, stride=1, padding=0, dilation=1, groups=1):
    return xlnstorch.csrc.conv1d_forward(x, weight, bias, ops.base, stride,
                                         padding, dilation, groups)

class LNSConv1dCPPFunction(LNSFunction):

    @staticmethod
    def forward(ops, x, weight, bias, stride=1, padding=0, dilation=1, groups=1):
        x, weight = x.view(torch.int64), weight.view(torch.int64)
        bias = bias.view(torch.int64) if bias is not None else None

        result = _conv1d_cpp(ops, x, weight, bias, stride,
                             padding, dilation, groups)
        return result.view(torch.float64)

    @staticmethod
    def setup_context(ctx, inputs, output):
        x, weight, bias, stride, padding, dilation, groups = inputs
        ctx.save_for_backward(x, weight)
        ctx.bias_defined = bias is not None
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.groups = groups

    @staticmethod
    def backward(ctx, ops, grad_output):
        x, weight = ctx.saved_tensors
        x, weight, grad_output = x.view(torch.int64), weight.view(torch.int64), grad_output.view(torch.int64)

        grads = xlnstorch.csrc.conv1d_backward(
            grad_output, x, weight, ops.base, ctx.bias_defined,
            ctx.stride, ctx.padding, ctx.dilation, ctx.groups)

        if ctx.bias_defined:
            return (grads[0].view(torch.float64), grads[1].view(torch.float64), 
                    grads[2].view(torch.float64), None, None, None, None)

        return (grads[0].view(torch.float64), grads[1].view(torch.float64),
                None, None, None, None, None)

@implements(torch.nn.functional.conv1d, _conv1d_cpp, "default_cpp", default=CSRC_AVAILABLE)
def conv1d(x, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):

    x, weight, bias = format_lnstensor_operands(x, weight, bias)
    result = LNSConv1dCPPFunction.apply(x, weight, bias, stride,
                                        padding, dilation, groups)

    return lnstensor(result, from_lns=True, b=x.base)

def _conv2d_cpp(ops, x, weight, bias, stride=1, padding=0, dilation=1, groups=1):
    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(padding, int):
        padding = (padding, padding)
    if isinstance(dilation, int):
        dilation = (dilation, dilation)

    return xlnstorch.csrc.conv2d_forward(x, weight, bias, ops.base, *stride,
                                         *padding, *dilation, groups)

class LNSConv2dCPPFunction(LNSFunction):

    @staticmethod
    def forward(ops, x, weight, bias, stride=1, padding=0, dilation=1, groups=1):
        x, weight = x.view(torch.int64), weight.view(torch.int64)
        bias = bias.view(torch.int64) if bias is not None else None

        result = _conv2d_cpp(ops, x, weight, bias, stride,
                             padding, dilation, groups)
        return result.view(torch.float64)

    @staticmethod
    def setup_context(ctx, ops, inputs, output):
        x, weight, bias, stride, padding, dilation, groups = inputs
        ctx.save_for_backward(x, weight)
        ctx.bias_defined = bias is not None
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.groups = groups

    @staticmethod
    def backward(ctx, ops, grad_output):
        x, weight = ctx.saved_tensors
        x, weight, grad_output = x.view(torch.int64), weight.view(torch.int64), grad_output.view(torch.int64)

        if isinstance(ctx.stride, int):
            ctx.stride = (ctx.stride, ctx.stride)
        if isinstance(ctx.padding, int):
            ctx.padding = (ctx.padding, ctx.padding)
        if isinstance(ctx.dilation, int):
            ctx.dilation = (ctx.dilation, ctx.dilation)

        grads = xlnstorch.csrc.conv2d_backward(
            grad_output, x, weight, ops.base,
            ctx.bias_defined, *ctx.stride,
            *ctx.padding, *ctx.dilation, ctx.groups)

        if ctx.bias_defined:
            return (grads[0].view(torch.float64), grads[1].view(torch.float64), 
                    grads[2].view(torch.float64), None, None, None, None)

        return (grads[0].to(torch.float64), grads[1].to(torch.float64),
                None, None, None, None, None)

@implements(torch.nn.functional.conv2d, _conv2d_cpp, "default_cpp", default=CSRC_AVAILABLE)
def conv2d(x, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):

    x, weight, bias = format_lnstensor_operands(x, weight, bias)
    result = LNSConv2dCPPFunction.apply(x, weight, bias, stride,
                                        padding, dilation, groups)

    return lnstensor(result, from_lns=True, b=x.base)

def _conv3d_cpp(ops, x, weight, bias, stride=1, padding=0, dilation=1, groups=1):
    if isinstance(stride, int):
        stride = (stride, stride, stride)
    if isinstance(padding, int):
        padding = (padding, padding, padding)
    if isinstance(dilation, int):
        dilation = (dilation, dilation, dilation)

    return xlnstorch.csrc.conv3d_forward(x, weight, bias, ops.base, *stride,
                                         *padding, *dilation, groups)

class LNSConv3dCPPFunction(LNSFunction):

    @staticmethod
    def forward(ops, x, weight, bias, stride=1, padding=0, dilation=1, groups=1):
        x, weight = x.view(torch.int64), weight.view(torch.int64)
        bias = bias.view(torch.int64) if bias is not None else None

        result = _conv3d_cpp(ops, x, weight, bias, stride,
                             padding, dilation, groups)
        return result.view(torch.float64)

    @staticmethod
    def setup_context(ctx, ops, inputs, output):
        x, weight, bias, stride, padding, dilation, groups = inputs
        ctx.save_for_backward(x, weight)
        ctx.bias_defined = bias is not None
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.groups = groups

    @staticmethod
    def backward(ctx, ops, grad_output):
        x, weight = ctx.saved_tensors
        x, weight, grad_output = x.view(torch.int64), weight.view(torch.int64), grad_output.view(torch.int64)

        if isinstance(ctx.stride, int):
            ctx.stride = (ctx.stride, ctx.stride, ctx.stride)
        if isinstance(ctx.padding, int):
            ctx.padding = (ctx.padding, ctx.padding, ctx.padding)
        if isinstance(ctx.dilation, int):
            ctx.dilation = (ctx.dilation, ctx.dilation, ctx.dilation)

        grads = xlnstorch.csrc.conv3d_backward(
            grad_output, x, weight, ops.base,
            ctx.bias_defined, *ctx.stride,
            *ctx.padding, *ctx.dilation, ctx.groups)

        if ctx.bias_defined:
            return (grads[0].view(torch.float64), grads[1].view(torch.float64), 
                    grads[2].view(torch.float64), None, None, None, None)

        return (grads[0].view(torch.float64), grads[1].view(torch.float64),
                None, None, None, None, None)

@implements(torch.nn.functional.conv3d, _conv3d_cpp, "default_cpp", default=CSRC_AVAILABLE)
def conv3d(x, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):

    x, weight, bias = format_lnstensor_operands(x, weight, bias)
    result = LNSConv3dCPPFunction.apply(x, weight, bias, stride,
                                        padding, dilation, groups)

    return lnstensor(result, from_lns=True, b=x.base)
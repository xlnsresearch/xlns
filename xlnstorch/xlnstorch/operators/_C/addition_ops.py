import torch
import xlnstorch.csrc
from xlnstorch import lnstensor, format_lnstensor_operands, implements, CSRC_AVAILABLE
from xlnstorch.sbdb_dispatch_table import DEFAULT_SBDB_FUNC
from xlnstorch.operators.addition_ops import _add
from xlnstorch.autograd import LNSFunction

SBDB_CPP_FUNCS = [
    "ideal",
    "tab"
]

def _add_cpp(ops, x, y):
    if DEFAULT_SBDB_FUNC in SBDB_CPP_FUNCS:
        return xlnstorch.csrc.add_forward(x, y, ops.base)

    return _add(ops, x, y)

class LNSAddCPPFunction(LNSFunction):

    @staticmethod
    def forward(ops, x, y):
        x, y = x.view(torch.int64), y.view(torch.int64)
        result = _add_cpp(ops, x, y)
        return result.view(torch.float64)

    @staticmethod
    def setup_context(ctx, ops, inputs, output):
        x, y = inputs
        ctx.save_for_backward(x, y)

    @staticmethod
    def backward(ctx, ops, grad_output):
        x, y = ctx.saved_tensors
        x, y, grad_output = x.view(torch.int64), y.view(torch.int64), grad_output.view(torch.int64)

        grad_x = ops.sum_to_size(grad_output, x.shape)
        grad_y = ops.sum_to_size(grad_output, y.shape)

        return grad_x.view(torch.float64), grad_y.view(torch.float64)

@implements(torch.add, _add_cpp, key='default_cpp', default=CSRC_AVAILABLE)
def add(x, y, *, alpha=1, out=None):

    x, y = format_lnstensor_operands(x, y)

    if alpha != 1:
        y = torch.mul(y, alpha)

    result = LNSAddCPPFunction.apply(x, y)

    if out is not None:
        return out._inplace_copy(result)

    return lnstensor(result, from_lns=True, b=x.base)

def _sum_cpp(ops, x, dim=None, keepdim=False):
    dim = [] if dim is None else ((dim,) if isinstance(dim, int) else dim)
    return xlnstorch.csrc.sum_forward(x, ops.base, dim, keepdim)

class LNSSumCPPFunction(LNSFunction):

    @staticmethod
    def forward(ops, x, dim=None, keepdim=False):
        x = x.view(torch.int64)
        result = _sum_cpp(ops, x, dim, keepdim)
        return result.view(torch.float64)

    @staticmethod
    def setup_context(ctx, ops, inputs, output):
        x, dim, keepdim = inputs
        ctx.save_for_backward(x)
        ctx.dim = dim
        ctx.keepdim = keepdim

    @staticmethod
    def backward(ctx, ops, grad_output):
        x, = ctx.saved_tensors
        grad_output = grad_output.view(torch.int64)

        grad_x = grad_output
        if ctx.dim is None:
            grad_x = grad_x.expand(x.shape)

        else:
            red_dims = (ctx.dim,) if isinstance(ctx.dim, int) else tuple(ctx.dim)
            red_dims = tuple(d % x.dim() for d in red_dims)

            if not ctx.keepdim:
                for d in sorted(red_dims):
                    grad_x = grad_x.unsqueeze(d)

            grad_x = grad_x.expand(x.shape)

        return grad_x, None, None

@implements(torch.sum, _sum_cpp, "default_cpp", default=CSRC_AVAILABLE)
def sum(x, dim=None, keepdim=False, *, out=None):

    result = LNSSumCPPFunction.apply(x, dim, keepdim)

    if out is not None:
        return out._inplace_copy(result)

    return lnstensor(result, from_lns=True, b=x.base)

def _matmul_cpp(ops, A, B):
    return xlnstorch.csrc.matmul_forward(A, B, ops.base)

class LNSMatmulCPPFunction(LNSFunction):

    @staticmethod
    def forward(ops, A, B):
        A, B = A.view(torch.int64), B.view(torch.int64)
        result = _matmul_cpp(ops, A, B)
        return result.view(torch.float64)

    @staticmethod
    def setup_context(ctx, ops, inputs, output):
        A, B = inputs
        ctx.save_for_backward(A, B)

    @staticmethod
    def backward(ctx, ops, grad_output):
        A, B = ctx.saved_tensors
        A, B, grad_output = A.view(torch.int64), B.view(torch.int64), grad_output.view(torch.int64)

        grad_A, grad_B = xlnstorch.csrc.matmul_backward(grad_output, A, B, ops.base)
        return grad_A.view(torch.float64), grad_B.view(torch.float64)

@implements(torch.matmul, _matmul_cpp, "default_cpp", default=CSRC_AVAILABLE)
def matmul(A, B, *, out=None):

    A, B = format_lnstensor_operands(A, B)
    result = LNSMatmulCPPFunction.apply(A, B)

    if out is not None:
        return out._inplace_copy(result)

    return lnstensor(result, from_lns=True, b=A.base)
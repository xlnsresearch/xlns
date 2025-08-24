import torch
import xlnstorch.csrc
from xlnstorch import lnstensor, format_lnstensor_operands, implements, CSRC_AVAILABLE
from xlnstorch.sbdb_dispatch_table import DEFAULT_SBDB_FUNC
from xlnstorch.operators import lns_sum_to_size
from xlnstorch.operators.addition_ops import LNSAddFunction
from xlnstorch.autograd import LNSFunction

SBDB_CPP_FUNCS = [
    "ideal",
    "tab"
]

class LNSAddCPPFunction(LNSFunction):

    @staticmethod
    def forward(x, y, base):
        if DEFAULT_SBDB_FUNC in SBDB_CPP_FUNCS:
            x_packed, y_packed = x.to(torch.int64), y.to(torch.int64)
            return xlnstorch.csrc.add_forward(x_packed, y_packed, base).to(torch.float64)
        return LNSAddFunction.forward(x, y, base)

    @staticmethod
    def setup_context(ctx, inputs, output):
        x, y, base = inputs
        ctx.save_for_backward(x, y, base)

    @staticmethod
    def backward(ctx, grad_output):
        x, y, base = ctx.saved_tensors

        grad_x = lns_sum_to_size(grad_output, base, x.shape)
        grad_y = lns_sum_to_size(grad_output, base, y.shape)

        return grad_x, grad_y, None

@implements(torch.add, LNSAddCPPFunction.forward, key='default_cpp', default=CSRC_AVAILABLE)
def add(x, y, *, alpha=1, out=None):

    x, y = format_lnstensor_operands(x, y)

    if alpha != 1:
        y = torch.mul(y, alpha)

    result = LNSAddCPPFunction.apply(x, y, x.base)

    if out is not None:
        return out._inplace_copy(result)

    return lnstensor(result, from_lns=True, b=x.base)

class LNSSumCPPFunction(LNSFunction):

    @staticmethod
    def forward(x, base, dim=None, keepdim=False):
        x_packed = x.to(torch.int64)
        dim = [] if dim is None else ((dim,) if isinstance(dim, int) else dim)

        return xlnstorch.csrc.sum_forward(x_packed, base, dim, keepdim).to(torch.float64)

    @staticmethod
    def setup_context(ctx, inputs, output):
        x, _, dim, keepdim = inputs
        ctx.save_for_backward(x)
        ctx.dim = dim
        ctx.keepdim = keepdim

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors

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

        return grad_x, None, None, None

@implements(torch.sum, LNSSumCPPFunction.forward, "default_cpp", default=CSRC_AVAILABLE)
def sum(x, dim=None, keepdim=False, *, out=None):

    result = LNSSumCPPFunction.apply(x, x.base, dim, keepdim)

    if out is not None:
        return out._inplace_copy(result)

    return lnstensor(result, from_lns=True, b=x.base)

class LNSMatmulCPPFunction(LNSFunction):

    @staticmethod
    def forward(A, B, base):
        A_packed, B_packed = A.to(torch.int64), B.to(torch.int64)
        return xlnstorch.csrc.matmul_forward(A_packed, B_packed, base).to(torch.float64)

    @staticmethod
    def setup_context(ctx, inputs, output):
        A, B, base = inputs
        ctx.save_for_backward(A, B, base)

    @staticmethod
    def backward(ctx, grad_output):
        A, B, base = ctx.saved_tensors
        grad_packed = grad_output.to(torch.int64)
        A_packed, B_packed = A.to(torch.int64), B.to(torch.int64)

        grad_A, grad_B = xlnstorch.csrc.matmul_backward(grad_packed, A_packed, B_packed, base)
        return grad_A, grad_B, None

@implements(torch.matmul, LNSMatmulCPPFunction.forward, "default_cpp", default=CSRC_AVAILABLE)
def matmul(A, B, *, out=None):

    A, B = format_lnstensor_operands(A, B)
    result = LNSMatmulCPPFunction.apply(A, B, A.base)

    if out is not None:
        return out._inplace_copy(result)

    return lnstensor(result, from_lns=True, b=A.base)
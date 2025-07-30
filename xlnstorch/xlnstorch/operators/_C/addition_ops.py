import torch
import xlnstorch._C
from xlnstorch import lnstensor, format_lnstensor_operands, implements, LNS_ONE
from xlnstorch.operators.addition_ops import DEFAULT_SBDB_FUNC, LNSAddFunction
from xlnstorch.autograd import LNSFunction

SBDB_CPP_FUNCS = [
    "ideal"
]

class LNSAddCPPFunction(LNSFunction):

    @staticmethod
    def forward(x, y, base):
        if DEFAULT_SBDB_FUNC in SBDB_CPP_FUNCS:
            x_packed, y_packed = x.to(torch.int64), y.to(torch.int64)
            return xlnstorch._C.add_forward(x_packed, y_packed, base)
        return LNSAddFunction.forward(x, y, base)

    @staticmethod
    def setup_context(ctx, inputs, output):
        pass

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, grad_output, None

@implements(torch.add, LNSAddCPPFunction.forward, key='default_cpp', default=True)
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

        return xlnstorch._C.sum_forward(x_packed, base, dim, keepdim)

    @staticmethod
    def setup_context(ctx, inputs, output):
        x, _, _, _ = inputs
        ctx.save_for_backward(x)

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        return torch.full_like(x, LNS_ONE.item()), None, None, None

@implements(torch.sum, LNSSumCPPFunction.forward, "default_cpp", default=True)
def sum(x, dim=None, keepdim=False, *, out=None):

    result = LNSSumCPPFunction.apply(x, x.base, dim, keepdim)

    if out is not None:
        return out._inplace_copy(result)

    return lnstensor(result, from_lns=True, b=x.base)

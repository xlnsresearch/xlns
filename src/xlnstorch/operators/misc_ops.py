import torch
from .. import LNS_ZERO, LNSTensor, lnstensor, format_lnstensor_operands, implements, rand
from ..autograd import LNSFunction
from . import (
    lns_sum,
)

class LNSExpandFunction(LNSFunction):

    @staticmethod
    def forward(x, shape, base):
        return torch.broadcast_to(x, shape)

    @staticmethod
    def setup_context(ctx, inputs, output):
        x, shape, base = inputs
        ctx.save_for_backward(x, base)
        ctx.shape = shape

    @staticmethod
    def backward(ctx, grad_output):
        x, base = ctx.saved_tensors

        # Sum over the broadcasted dimensions
        # First, handle prepended dimensions (when original tensor had fewer dims)
        ndims_added = grad_output.ndim - len(x.shape)
        for i in range(ndims_added):
            grad_output = lns_sum(grad_output, base, dim=0, keepdim=False)

        # Then, handle expanded dimensions (where original dim was 1)
        for i, (orig_size, grad_size) in enumerate(zip(x.shape, grad_output.shape)):
            if orig_size == 1 and grad_size > 1:
                grad_output = lns_sum(grad_output, base, dim=i, keepdim=True)

        return grad_output, None, None

# note that torch.broadcast_to is equivalent to torch.Tensor.expand
@implements(torch.broadcast_to, LNSExpandFunction.forward, "default", default=True)
def broadcast_to(x, shape):

    result = LNSExpandFunction.apply(x, shape, x.base)
    return lnstensor(result, from_lns=True, b=x.base)

class LNSCloneFunction(LNSFunction):

    @staticmethod
    def forward(x, memory_format=torch.preserve_format):
        return x.clone(memory_format=memory_format)

    @staticmethod
    def setup_context(ctx, inputs, output):
        x, _ = inputs
        ctx.save_for_backward(x)

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        return grad_output, None

@implements(torch.clone, LNSCloneFunction.forward, "default", default=True)
def clone(x, memory_format=torch.preserve_format):

    result = LNSCloneFunction.apply(x, memory_format)
    return lnstensor(result, from_lns=True, b=x.base)

class LNSSqueezeFunction(LNSFunction):

    @staticmethod
    def forward(x, dim=None):
        return torch.squeeze(x, dim=dim)

    @staticmethod
    def setup_context(ctx, inputs, output):
        x, dim = inputs
        ctx.save_for_backward(x)
        ctx.dim = dim

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        # Unsqueeze to restore original shape
        grad_x = grad_output.view(x.shape)
        return grad_x, None

@implements(torch.squeeze, LNSSqueezeFunction.forward, "default", default=True)
def squeeze(x, dim=None):

    result = LNSSqueezeFunction.apply(x, dim)
    return lnstensor(result, from_lns=True, b=x.base)

class LNSUnsqueezeFunction(LNSFunction):

    @staticmethod
    def forward(x, dim):
        return torch.unsqueeze(x, dim=dim)

    @staticmethod
    def setup_context(ctx, inputs, output):
        x, dim = inputs
        ctx.save_for_backward(x)
        ctx.dim = dim

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        grad_input = torch.squeeze(grad_output, dim=ctx.dim)
        return grad_input, None

@implements(torch.unsqueeze, LNSUnsqueezeFunction.forward, "default", default=True)
def unsqueeze(x, dim):

    result = LNSUnsqueezeFunction.apply(x, dim)
    return lnstensor(result, from_lns=True, b=x.base)


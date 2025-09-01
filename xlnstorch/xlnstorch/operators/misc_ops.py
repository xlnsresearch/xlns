import torch
from xlnstorch import LNS_ZERO, LNSTensor, lnstensor, format_lnstensor_operands, implements, ones
from xlnstorch.autograd import LNSFunction

class LNSExpandFunction(LNSFunction):

    @staticmethod
    def forward(ops, x, shape):
        return torch.broadcast_to(x, shape)

    @staticmethod
    def setup_context(ctx, ops, inputs, output):
        x, shape = inputs
        ctx.save_for_backward(x)
        ctx.shape = shape

    @staticmethod
    def backward(ctx, ops, grad_output):
        x, = ctx.saved_tensors

        # Sum over the broadcasted dimensions
        # First, handle prepended dimensions (when original tensor had fewer dims)
        ndims_added = grad_output.ndim - len(x.shape)
        grad_x = grad_output
        for i in range(ndims_added):
            grad_x = ops.sum(grad_x, dim=0, keepdim=False)

        # Then, handle expanded dimensions (where original dim was 1)
        for i, (orig_size, grad_size) in enumerate(zip(x.shape, grad_x.shape)):
            if orig_size == 1 and grad_size > 1:
                grad_x = ops.sum(grad_x, dim=i, keepdim=True)

        return grad_x, None

# note that torch.broadcast_to is equivalent to torch.Tensor.expand
@implements(torch.broadcast_to, LNSExpandFunction.forward, "default", default=True)
def broadcast_to(x, shape):

    result = LNSExpandFunction.apply(x, shape)
    return lnstensor(result, from_lns=True, b=x.base)

class LNSCloneFunction(LNSFunction):

    @staticmethod
    def forward(ops, x, memory_format=torch.preserve_format):
        return x.clone(memory_format=memory_format)

    @staticmethod
    def setup_context(ctx, ops, inputs, output):
        pass

    @staticmethod
    def backward(ctx, ops, grad_output):
        return grad_output, None

@implements(torch.clone, LNSCloneFunction.forward, "default", default=True)
def clone(x, memory_format=torch.preserve_format):

    result = LNSCloneFunction.apply(x, memory_format)
    return lnstensor(result, from_lns=True, b=x.base)

class LNSSqueezeFunction(LNSFunction):

    @staticmethod
    def forward(ops, x, dim=None):
        return torch.squeeze(x, dim=dim)

    @staticmethod
    def setup_context(ctx, ops, inputs, output):
        x, dim = inputs
        ctx.save_for_backward(x)
        ctx.dim = dim

    @staticmethod
    def backward(ctx, ops, grad_output):
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
    def forward(ops, x, dim):
        return torch.unsqueeze(x, dim=dim)

    @staticmethod
    def setup_context(ctx, ops, inputs, output):
        _, dim = inputs
        ctx.dim = dim

    @staticmethod
    def backward(ctx, ops, grad_output):
        grad_x = torch.squeeze(grad_output, dim=ctx.dim)
        return grad_x, None

@implements(torch.unsqueeze, LNSUnsqueezeFunction.forward, "default", default=True)
def unsqueeze(x, dim):

    result = LNSUnsqueezeFunction.apply(x, dim)
    return lnstensor(result, from_lns=True, b=x.base)

class LNSIndexPutFunction(LNSFunction):

    @staticmethod
    def forward(ops, x, idx, value, accumulate=False):
        return torch.index_put(x, idx, value, accumulate=accumulate)

    @staticmethod
    def setup_context(ctx, ops, inputs, output):
        _, idx, value, _ = inputs
        ctx.is_idx_tensor = torch.is_tensor(idx)

        if ctx.is_idx_tensor:
            ctx.save_for_backward(idx, value)
        else:
            ctx.save_for_backward(value)
            ctx.idx = idx

    @staticmethod
    def backward(ctx, ops, grad_output):

        if ctx.is_idx_tensor:
            idx, value = ctx.saved_tensors
        else:
            value, = ctx.saved_tensors
            idx = ctx.idx

        grad_x = grad_output.clone()
        grad_x[idx] = LNS_ZERO

        grad_value = grad_output.clone()[idx]
        if grad_value.shape != value.shape:

            # Find the dims that were broadcast (= size 1 in value but >1 in grad_value)
            extra_dims = (
                [i for i, (gv, v) in enumerate(zip(grad_value.shape[-len(value.shape):],
                                                   value.shape)) if v == 1 and gv != 1]
                + list(range(len(grad_value.shape) - len(value.shape)))  # leading dims
            )
            grad_value = ops.sum(grad_value, dim=extra_dims, keepdim=True)
            grad_value = grad_value.reshape(value.shape)

        return grad_x, None, grad_value, None

@implements(torch.index_put, LNSIndexPutFunction.forward, "default", default=True)
def index_put(x, indices, values, accumulate=False):

    x, values = format_lnstensor_operands(x, values)
    result = LNSIndexPutFunction.apply(x, indices, values, accumulate)

    return lnstensor(result, from_lns=True, b=x.base)

@implements(torch.index_put_, LNSIndexPutFunction.forward, "default", default=True)
def index_put_(x, indices, values, accumulate=False):

    x, values = format_lnstensor_operands(x, values)
    result = LNSIndexPutFunction.apply(x, indices, values, accumulate)

    return x._inplace_copy(result)

class LNSStackFunction(LNSFunction):

    @staticmethod
    def forward(ops, dim, *tensors):
        return torch.stack(tensors, dim=dim)

    @staticmethod
    def setup_context(ctx, ops, inputs, output):
        dim, *tensors = inputs
        ctx.save_for_backward(*tensors)
        ctx.dim = dim

    @staticmethod
    def backward(ctx, ops, grad_output):
        grad_tensor = grad_output.unbind(ctx.dim)
        return (None, *grad_tensor)

@implements(torch.stack, LNSStackFunction.forward, "default", default=True)
def stack(tensors, dim=0):

    tensors = format_lnstensor_operands(*tensors)
    result = LNSStackFunction.apply(dim, *tensors)

    return lnstensor(result, from_lns=True, b=tensors[0].base)

class LNSCatFunction(LNSFunction):

    @staticmethod
    def forward(ops, dim, *tensors):
        return torch.cat(tensors, dim=dim)

    @staticmethod
    def setup_context(ctx, ops, inputs, output):
        dim, *tensors = inputs
        ctx.sizes = [t.size(dim) for t in tensors]
        ctx.dim = dim

    @staticmethod
    def backward(ctx, ops, grad_output):
        grad_tensors = torch.split(grad_output, ctx.sizes, dim=ctx.dim)
        return (None, *grad_tensors)

@implements(torch.cat, LNSCatFunction.forward, "default", default=True)
def cat(tensors, dim=0):

    tensors = format_lnstensor_operands(*tensors)
    result = LNSCatFunction.apply(dim, *tensors)

    return lnstensor(result, from_lns=True, b=tensors[0].base)

class LNSChunkFunction(LNSFunction):

    @staticmethod
    def forward(ops, x, chunks, dim=0):
        return torch.chunk(x, chunks, dim=dim)

    @staticmethod
    def setup_context(ctx, ops, inputs, output):
        _, _, dim = inputs
        ctx.dim = dim
        ctx.out_shapes = [o.shape for o in output]
        ctx.set_materialize_grads(False)

    @staticmethod
    def backward(ctx, ops, *grad_outputs):
        parts = []

        for g, shape in zip(grad_outputs, ctx.out_shapes):
            if g is None:
                g = ops.zeros(*shape)
            parts.append(g)

        grad_x = torch.cat(parts, dim=ctx.dim)
        return grad_x, None, None

@implements(torch.chunk, LNSChunkFunction.forward, "default", default=True)
def chunk(x, chunks, dim=0):

    result = LNSChunkFunction.apply(x, chunks, dim)
    return tuple(lnstensor(r, from_lns=True, b=x.base) for r in result)

def _where(ops, condition, x, y):
    return torch.where(condition, x, y)

class LNSWhereFunction(LNSFunction):

    @staticmethod
    def forward(ops, condition, x, y):
        return _where(ops, condition, x, y)

    @staticmethod
    def setup_context(ctx, ops, inputs, output):
        condition, x, y = inputs
        ctx.save_for_backward(condition, x, y)

    @staticmethod
    def backward(ctx, ops, grad_output):
        condition, x, y = ctx.saved_tensors

        grad_x = torch.where(condition, grad_output, LNS_ZERO)
        grad_y = torch.where(condition, LNS_ZERO, grad_output)

        grad_x = ops.sum_to_size(grad_x, x.shape)
        grad_y = ops.sum_to_size(grad_y, y.shape)

        return None, grad_x, grad_y

@implements(torch.where, _where, "default", default=True)
def where(condition, x, y, *, out=None):

    x, y = format_lnstensor_operands(x, y)
    result = LNSWhereFunction.apply(condition, x, y)

    if out is not None:
        return out._inplace_copy(result)

    return lnstensor(result, from_lns=True, b=x.base)

def _unpad_along_dim(ops, g, left, right, dim, mode):

    if left == right == 0:
        return g

    if mode == "constant":
        return g.narrow(dim, left, g.size(dim) - left - right).clone()

    interior_len = g.size(dim) - left - right
    grad_x = g.narrow(dim, left, interior_len).clone()

    first = 0
    last = interior_len - 1

    if mode == "replicate":
        if left:
            grad_x.select(dim, first).copy_(ops.add(
                grad_x.select(dim, first),
                ops.sum(g.narrow(dim, 0, left), dim=dim),
            ))
        if right:
            grad_x.select(dim, last).copy_(ops.add(
                grad_x.select(dim, last),
                ops.sum(g.narrow(dim, g.size(dim) - right, right), dim=dim),
            ))

    elif mode == "reflect":

        for i in range(left):
            target = left - i
            grad_x.select(dim, target).copy_(ops.add(
                grad_x.select(dim, target),
                g.select(dim, i),
            ))

        for i in range(right):
            target = last - 1 - i
            grad_x.select(dim, target).copy_(ops.add(
                grad_x.select(dim, target),
                g.select(dim, g.size(dim) - 1 - i),
            ))

    elif mode == "circular":

        if left:
            grad_x.narrow(dim, interior_len-left, left).copy_(ops.add(
                grad_x.narrow(dim, interior_len - left, left),
                g.narrow(dim, 0, left),
            ))

        if right:
            grad_x.narrow(dim, 0, right).copy_(ops.add(
                grad_x.narrow(dim, 0, right),
                g.narrow(dim, g.size(dim) - right, right),
            ))

    return grad_x

def _pad(x, pad, mode="constant", value=None):
    return torch.nn.functional.pad(x, pad, mode=mode, value=value)

class LNSPadFunction(LNSFunction):

    @staticmethod
    def forward(ops, x, pad, mode="constant", value=None):
        return _pad(x, pad, mode=mode, value=value)

    @staticmethod
    def setup_context(ctx, ops, inputs, output):
        _, pad, mode, _ = inputs
        ctx.pad = pad
        ctx.mode = mode

    @staticmethod
    def backward(ctx, ops, grad_output):

        ndim_pad = len(ctx.pad) // 2
        grad_x = grad_output
        for i in range(ndim_pad):
            left = ctx.pad[2 * i]
            right = ctx.pad[2 * i + 1]

            dim = grad_output.dim() - 1 - i
            grad_x = _unpad_along_dim(ops, grad_x, left, right, dim, ctx.mode)

        return grad_x, None, None, None

@implements(torch.nn.functional.pad, _pad, "default", default=True)
def pad(x, pad, mode="constant", value=0):

    if mode == "constant":
        x, value = format_lnstensor_operands(x, value)

    result = LNSPadFunction.apply(x, pad, mode, value)
    return lnstensor(result, from_lns=True, b=x.base)
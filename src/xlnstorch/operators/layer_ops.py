import warnings

import torch
from .. import LNS_ZERO, LNSTensor, lnstensor, format_lnstensor_operands, implements, zeros_like
from . import (
    lns_mul
)

class LNSDropoutFunction(torch.autograd.Function):

    @staticmethod
    def forward(x, base, p=0.5):
        x_packed = x.to(torch.int64)

        mask = LNSTensor.get_internal_tensor(torch.bernoulli(torch.full_like(x, 1 - p)), base)
        result = lns_mul(x_packed, mask)

        return result.to(torch.float64)

    @staticmethod
    def setup_context(ctx, inputs, output):
        _, base, p = inputs
        ctx.save_for_backward(output, base)
        ctx.p = p

    @staticmethod
    def backward(ctx, grad_output):
        output, base = ctx.saved_tensors

        grad_x = torch.where(output == LNS_ZERO, LNS_ZERO, LNSTensor.get_internal_tensor(1 / (1 - ctx.p), base))
        grad_x = lns_mul(grad_output, grad_x)

        return grad_x, None, None

@implements(torch.nn.functional.dropout, LNSDropoutFunction.forward, "default", default=True)
def dropout(x, p=0.5, training=True, inplace=False):

    if not training or p == 0.0:
        return x

    if p < 0.0 or p > 1.0:
        raise ValueError(f"Dropout probability p must be in the range [0, 1], but got {p}.")

    result = LNSDropoutFunction.apply(x._lns, x.base, p)

    if inplace:
        x._lns = result
        return x

    return lnstensor(result, from_lns=True, b=x.base)

class LNSDropout1dFunction(torch.autograd.Function):

    @staticmethod
    def forward(x, base, p=0.5):
        x_packed = x.to(torch.int64)

        if x.dim() == 2:
            channel_shape = (x.size(0), 1)
        elif x.dim() == 3:
            channel_shape = (x.size(0), x.size(1), 1)

        mask_flt = torch.bernoulli(
            torch.full(channel_shape, 1 - p, dtype=x.dtype, device=x.device)
        ).expand_as(x)
        mask = LNSTensor.get_internal_tensor(mask_flt, base)

        result = lns_mul(x_packed, mask)
        return result.to(torch.float64)

    @staticmethod
    def setup_context(ctx, inputs, output):
        _, base, p = inputs
        ctx.save_for_backward(output, base)
        ctx.p = p

    @staticmethod
    def backward(ctx, grad_output):
        output, base = ctx.saved_tensors

        grad_x = torch.where(output == LNS_ZERO, LNS_ZERO, LNSTensor.get_internal_tensor(1 / (1 - ctx.p), base))
        grad_x = lns_mul(grad_output, grad_x)

        return grad_x, None, None

@implements(torch.nn.functional.dropout1d, LNSDropout1dFunction.forward, "default", default=True)
def dropout1d(x, p=0.5, training=True, inplace=False):

    if not training or p == 0.0:
        return x

    if p < 0.0 or p > 1.0:
        raise ValueError(f"Dropout probability p must be in the range [0, 1], but got {p}.")

    if x.dim() < 2 or x.dim() > 3:
        raise ValueError(f"Dropout1d expects a 2D or 3D tensor, but got a tensor with {x.dim()} dimensions.")

    result = LNSDropout1dFunction.apply(x._lns, x.base, p)

    if inplace:
        x._lns = result
        return x

    return lnstensor(result, from_lns=True, b=x.base)

class LNSDropout2dFunction(torch.autograd.Function):

    @staticmethod
    def forward(x, base, p=0.5):
        x_packed = x.to(torch.int64)

        if x.dim() == 3:
            channel_shape = (x.size(0), 1, 1)
        else:
            channel_shape = (x.size(0), x.size(1), 1, 1)

        mask_flt = torch.bernoulli(
            torch.full(channel_shape, 1 - p, dtype=x.dtype, device=x.device)
        ).expand_as(x)
        mask = LNSTensor.get_internal_tensor(mask_flt, base)

        result = lns_mul(x_packed, mask)
        return result.to(torch.float64)

    @staticmethod
    def setup_context(ctx, inputs, output):
        _, base, p = inputs
        ctx.save_for_backward(output, base)
        ctx.p = p

    @staticmethod
    def backward(ctx, grad_output):
        output, base = ctx.saved_tensors

        grad_x = torch.where(output == LNS_ZERO, LNS_ZERO, LNSTensor.get_internal_tensor(1 / (1 - ctx.p), base))
        grad_x = lns_mul(grad_output, grad_x)

        return grad_x, None, None

@implements(torch.nn.functional.dropout2d, LNSDropout2dFunction.forward, "default", default=True)
def dropout2d(x, p=0.5, training=True, inplace=False):

    if not training or p == 0.0:
        return x

    if p < 0.0 or p > 1.0:
        raise ValueError(f"Dropout probability p must be in the range [0, 1], but got {p}.")

    if x.dim() < 3 or x.dim() > 4:
        raise ValueError(f"Dropout2d expects a 3D or 4D tensor, but got a tensor with {x.dim()} dimensions.")
    elif x.dim() == 3:
        # Warning from PyTorch. Copied for now.
        warnings.warn("UserWarning: dropout2d: Received a 3D input to dropout2d and assuming that channel-wise"
                      "1D dropout behavior is desired - input is interpreted as shape (N, C, L), where C is the"
                      "channel dim. This behavior will change in a future release to interpret the input as one"
                      "without a batch dimension, i.e. shape (C, H, W). To maintain the 1D channel-wise dropout"
                      "behavior, please switch to using dropout1d instead.")
        return torch.nn.functional.dropout1d(x, p=p, training=training, inplace=inplace)

    result = LNSDropout2dFunction.apply(x._lns, x.base, p)

    if inplace:
        x._lns = result
        return x

    return lnstensor(result, from_lns=True, b=x.base)

class LNSDropout3dFunction(torch.autograd.Function):

    @staticmethod
    def forward(x, base, p=0.5):
        x_packed = x.to(torch.int64)

        if x.dim() == 4:
            channel_shape = (x.size(0), 1, 1, 1)
        else:
            channel_shape = (x.size(0), x.size(1), 1, 1, 1)

        mask_flt = torch.bernoulli(
            torch.full(channel_shape, 1 - p, dtype=x.dtype, device=x.device)
        ).expand_as(x)
        mask = LNSTensor.get_internal_tensor(mask_flt, base)

        result = lns_mul(x_packed, mask)
        return result.to(torch.float64)

    @staticmethod
    def setup_context(ctx, inputs, output):
        _, base, p = inputs
        ctx.save_for_backward(output, base)
        ctx.p = p

    @staticmethod
    def backward(ctx, grad_output):
        output, base = ctx.saved_tensors

        grad_x = torch.where(output == LNS_ZERO, LNS_ZERO, LNSTensor.get_internal_tensor(1 / (1 - ctx.p), base))
        grad_x = lns_mul(grad_output, grad_x)

        return grad_x, None, None

@implements(torch.nn.functional.dropout3d, LNSDropout3dFunction.forward, "default", default=True)
def dropout3d(x, p=0.5, training=True, inplace=False):

    if not training or p == 0.0:
        return x

    if p < 0.0 or p > 1.0:
        raise ValueError(f"Dropout probability p must be in the range [0, 1], but got {p}.")

    if x.dim() < 4 or x.dim() > 5:
        raise ValueError(f"Dropout3d expects a 4D or 5D tensor, but got a tensor with {x.dim()} dimensions.")

    result = LNSDropout3dFunction.apply(x._lns, x.base, p)

    if inplace:
        x._lns = result
        return x

    return lnstensor(result, from_lns=True, b=x.base)
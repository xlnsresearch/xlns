import warnings

import torch
from .. import LNS_ZERO, LNSTensor, lnstensor, format_lnstensor_operands, implements, zeros, zeros_like
from . import (
    lns_mul,
    lns_sum,
    lns_add,
    lns_matmul,
)

class LNSLinearFunction(torch.autograd.Function):
    """
    Linear transformation is implemented using matrix
    multiplication followed by addition of a bias term.

    Gradients are computed as follows:
    d/dx(x @ A^T + b) = A
    d/dA(x @ A^T + b) = x^T
    d/db(x @ A^T + b) = 1
    """

    @staticmethod
    def forward(x, A, base, bias=None):
        x_packed, A_packed = x.to(torch.int64), A.to(torch.int64)

        output = lns_matmul(x_packed, A_packed.transpose(-2, -1), base)
        if bias is not None:
            output = lns_add(output, bias, base)

        return output

    @staticmethod
    def setup_context(ctx, inputs, output):
        x, A, base, bias = inputs
        ctx.biased = True if bias is not None else False
        ctx.save_for_backward(x, A, base)

    @staticmethod
    def backward(ctx, grad_output):
        x, A, base = ctx.saved_tensors

        grad_x = lns_matmul(grad_output, A, base)

        out_features = A.shape[0]
        in_features = A.shape[1]
        *batch_dims, _ = grad_output.shape

        grad_output_2d = grad_output.reshape(-1, out_features)
        x_2d = x.reshape(-1, in_features)
        grad_output_T = grad_output_2d.transpose(0, 1)
        grad_A = lns_matmul(grad_output_T, x_2d, base)

        if ctx.biased:
            if grad_output.dim() == 1:
                grad_bias = grad_output
            else:
                grad_bias = lns_sum(grad_output, base, dim=tuple(range(grad_output.dim() - 1)))
        else:
            grad_bias = None

        return grad_x, grad_A, None, grad_bias

@implements(torch.nn.functional.linear, LNSLinearFunction.forward, key='default', default=True)
def linear(x, weight, bias=None):

    if bias is not None:
        x, weight, bias = format_lnstensor_operands(x, weight, bias)
        bias_lns = bias._lns
    else:
        x, weight = format_lnstensor_operands(x, weight)
        bias_lns = None

    result = LNSLinearFunction.apply(x._lns, weight._lns, x.base, bias_lns)

    return lnstensor(result, from_lns=True, b=x.base)

class LNSBilinearFunction(torch.autograd.Function):
    """
    Linear transformation is implemented using matrix
    multiplication followed by addition of a bias term.

    Gradients are computed as follows:
    d/dx(x^T @ A @ y + b) = A @ y
    d/dA(x^T @ A @ y + b) = x @ y^T
    d/dy(x^T @ A @ y + b) = A^T @ x
    d/db(x^T @ A @ y + b) = 1
    """

    @staticmethod
    def forward(x, y, A, base, bias=None):
        x_packed, y_packed, A_packed = x.to(torch.int64), y.to(torch.int64), A.to(torch.int64)

        tmp = lns_matmul(A_packed, y_packed.unsqueeze(-1), base).squeeze(-1)
        if tmp.dim() == 1:
            tmp = tmp.unsqueeze(-2)
        output = lns_matmul(x_packed.unsqueeze(-2), tmp.transpose(-2, -1), base).squeeze(-2)

        if bias is not None:
            output = lns_add(output, bias, base)

        return output

    @staticmethod
    def setup_context(ctx, inputs, output):
        x, y, A, base, bias = inputs
        ctx.biased = True if bias is not None else False
        ctx.save_for_backward(x, y, A, base)

    @staticmethod
    def backward(ctx, grad_output):
        x, y, A, base = ctx.saved_tensors

        Ay = lns_matmul(A, y.unsqueeze(-1), base).squeeze(-1)
        grad_x = lns_matmul(grad_output.unsqueeze(-2), Ay, base).squeeze(-2)

        ATx = lns_matmul(A.transpose(-2, -1), x.unsqueeze(-1), base).squeeze(-1)
        grad_y = lns_matmul(grad_output.unsqueeze(-2), ATx, base).squeeze(-2)

        if ctx.biased:
            if grad_output.dim() == 1:
                grad_bias = grad_output
            else:
                grad_bias = lns_sum(grad_output, base, dim=tuple(range(grad_output.dim() - 1)))
        else:
            grad_bias = None

        return grad_x, grad_y, None, None, grad_bias # todo: grad_A requires einsum

@implements(torch.nn.functional.bilinear, LNSBilinearFunction.forward, key='default', default=True)
def bilinear(x, y, weight, bias=None):

    if bias is not None:
        x, y, weight, bias = format_lnstensor_operands(x, y, weight, bias)
        bias_lns = bias._lns
    else:
        x, y, weight = format_lnstensor_operands(x, y, weight)
        bias_lns = None

    result = LNSBilinearFunction.apply(x._lns, y._lns, weight._lns, x.base, bias_lns)

    return lnstensor(result, from_lns=True, b=x.base)

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

class LNSConv1dFunction(torch.autograd.Function):

    @staticmethod
    def forward(x, weight, bias, base, stride=1, padding=0, dilation=1, groups=1):
        x_packed, weight_packed = x.to(torch.int64), weight.to(torch.int64)

        # add batch dimension if needed
        squeeze_batch = False
        if x.dim() == 2:
            squeeze_batch = True
            x_packed = x_packed.unsqueeze(0)

        N, C_in, L_in = x_packed.shape # Batch size, input channels, input length
        C_out, C_w, K = weight.shape   # Output channels, weight channels per group, kernel size

        # Basic checks for grouped conv: channels must be divisible by groups
        assert C_in % groups == 0, f"C_in must be divisible by groups ({C_in} % {groups})"
        assert C_out % groups == 0, f"C_out must be divisible by groups ({C_out} % {groups})"
        assert C_w == C_in // groups, f"Weight shape mismatch: {C_w} vs {C_in // groups}"

        g_Cin = C_in // groups
        g_Cout = C_out // groups

        if padding > 0:
            x_padded = torch.nn.functional.pad(x, (padding, padding), value=LNS_ZERO.item())
        else:
            x_padded = x_packed

        # Output length calculation based on kernel parameters (same as PyTorch)
        L_out = (L_in + 2 * padding - dilation * (K - 1) - 1) // stride + 1
        out = zeros(N, C_out, L_out, device=x.device, b=base)._lns

        for n in range(N):
            for g in range(groups):
                # Select this group's input channels
                inp_group = x_padded[n, g * g_Cin : (g+1) * g_Cin]
                #  Iterate over this group's output channels
                for c_out in range(g * g_Cout, (g + 1) * g_Cout):
                    for l in range(L_out):
                        start = l * stride
                        end = start + K * dilation
                        inp_slice = inp_group[:, start:end:dilation]
                        # Element-wise multiply and sum across in_channels and kernel size
                        out[n, c_out, l] = lns_sum(lns_mul(inp_slice, weight_packed[c_out]), base)
                        if bias is not None:
                            out[n, c_out, l] = lns_add(out[n, c_out, l], bias[c_out], base)

        # If batch dimension was added, remove before returning
        if squeeze_batch:
            out = out.squeeze(0)

        return out.to(torch.float64)

    @staticmethod
    def setup_context(ctx, inputs, output):
        x, weight, bias, base, stride, padding, dilation, groups = inputs
        ctx.save_for_backward(x, weight, bias, base)
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.groups = groups

    @staticmethod
    def backward(ctx, grad_output):
        x, weight, bias, base = ctx.saved_tensors

        # Add batch dimension to grad_output if not present
        squeeze_batch = False
        if grad_output.dim() == 2:
            grad_output = grad_output.unsqueeze(0)
            squeeze_batch = True

        N, C_out, L_out = grad_output.shape
        C_in = weight.shape[1] * ctx.groups
        K = weight.shape[2]
        g_Cin = C_in // ctx.groups
        g_Cout = C_out // ctx.groups

        # Pad input as in forward for correct alignment
        if ctx.padding > 0:
            x_padded = torch.nn.functional.pad(x, (ctx.padding, ctx.padding))
        else:
            x_padded = x

        grad_x_padded = zeros(N, C_in, x.shape[-1] + 2 * ctx.padding, device=grad_output.device, b=base)._lns
        grad_weight = zeros_like(weight, b=base)._lns

        # Compute input gradient: for each padded input element, sum contributions from output grads via chain rule
        for n in range(N):
            for g in range(ctx.groups):
                in_start = g * g_Cin
                in_end = (g + 1) * g_Cin
                out_start = g * g_Cout
                out_end = (g + 1) * g_Cout

                for c_in in range(g_Cin):
                    for l_in in range(x.shape[-1] + 2*ctx.padding):
                        grad = LNSTensor.get_internal_tensor(0, base)
                        # Accumulate gradient over all relevant output channels and positions
                        for c_out in range(out_start, out_end):
                            w = weight[c_out, c_in, :]
                            for k in range(K):
                                # Compute if this input position is in the receptive field of this output
                                l_out_nom = l_in - k * ctx.dilation
                                if l_out_nom % ctx.stride == 0:
                                    l_out = l_out_nom // ctx.stride
                                    if l_out >= 0 and l_out < L_out:
                                        # Chain rule for gradients through conv
                                        grad = lns_add(grad, lns_mul(grad_output[n, c_out, l_out], w[k]), base)
                        grad_x_padded[n, in_start + c_in, l_in] = grad

        # Remove padding to match input shape, as in forward
        if ctx.padding > 0:
            grad_x = grad_x_padded[:, :, ctx.padding:-ctx.padding]
        else:
            grad_x = grad_x_padded

        # Compute weight gradients: correlate grad_output with input windows
        for g in range(ctx.groups):
            in_start = g * g_Cin
            in_end = (g + 1) * g_Cin
            out_start = g * g_Cout
            out_end = (g + 1) * g_Cout

            for c_out in range(out_start, out_end):
                for c_in in range(g_Cin):
                    for k in range(K):
                        grad = LNSTensor.get_internal_tensor(0, base)
                        # Sum over all samples and locations
                        for n in range(N):
                            for l_out in range(L_out):
                                l_in = l_out * ctx.stride + k * ctx.dilation
                                inp_padded = x_padded[n, in_start + c_in, :]
                                if ctx.padding <= l_in < inp_padded.size(0) - ctx.padding:
                                    grad = lns_add(grad, lns_mul(grad_output[n, c_out, l_out], inp_padded[l_in]), base)
                        grad_weight[c_out, c_in, k] = grad

        # Compute bias gradient by summing grad_output along batch and time (output length)
        if bias is not None:
            grad_bias = lns_sum(grad_output, base, dim=[0, 2])
        else:
            grad_bias = None

        # Remove batch dim if input originally had none
        if squeeze_batch:
            grad_x = grad_x.squeeze(0)

        return grad_x, grad_weight, grad_bias, None, None, None, None, None

@implements(torch.nn.functional.conv1d, LNSConv1dFunction.forward, "default", default=True)
def conv1d(x, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):

    if bias is not None:
        x, weight, bias = format_lnstensor_operands(x, weight, bias)
        bias_lns = bias._lns
    else:
        x, weight = format_lnstensor_operands(x, weight)
        bias_lns = None

    result = LNSConv1dFunction.apply(x._lns, weight._lns, bias_lns, x.base,
                                     stride, padding, dilation, groups)

    return lnstensor(result, from_lns=True, b=x.base)

class LNSConv2dFunction(torch.autograd.Function):

    @staticmethod
    def forward(x, weight, bias, base, stride=1, padding=0, dilation=1, groups=1):
        # Handle stride, padding, dilation as tuples
        if isinstance(stride, int):
            stride = (stride, stride)
        if isinstance(padding, int):
            padding = (padding, padding)
        if isinstance(dilation, int):
            dilation = (dilation, dilation)

        stride_h, stride_w = stride
        pad_h, pad_w = padding
        dil_h, dil_w = dilation

        x_packed, weight_packed = x.to(torch.int64), weight.to(torch.int64)

        # Add batch dimension if needed
        squeeze_batch = False
        if x.dim() == 3:
            squeeze_batch = True
            x_packed = x_packed.unsqueeze(0)

        N, C_in, H_in, W_in = x_packed.shape # [batch, in_channels, height, width]
        C_out, C_w, K_H, K_W = weight.shape  # [out_channels, in_per_group, kh, kw]
        assert C_in % groups == 0
        assert C_out % groups == 0
        assert C_w == C_in // groups

        g_Cin = C_in // groups
        g_Cout = C_out // groups

        # Pad input
        if pad_h > 0 or pad_w > 0:
            x_padded = torch.nn.functional.pad(
                x_packed, (pad_w, pad_w, pad_h, pad_h), value=LNS_ZERO.item()
            )
        else:
            x_padded = x_packed

        H_pad, W_pad = x_padded.shape[2:]
        # Output shape calculation as per PyTorch
        H_out = (H_in + 2 * pad_h - dil_h * (K_H - 1) - 1) // stride_h + 1
        W_out = (W_in + 2 * pad_w - dil_w * (K_W - 1) - 1) // stride_w + 1
        out = zeros(N, C_out, H_out, W_out, device=x.device, b=base)._lns

        for n in range(N):
            for g in range(groups):
                inp_group = x_padded[n, g * g_Cin : (g + 1) * g_Cin] # [g_Cin, H_pad, W_pad]
                for c_out in range(g * g_Cout, (g + 1) * g_Cout):
                    for h in range(H_out):
                        for w in range(W_out):
                            h_start = h * stride_h
                            w_start = w * stride_w
                            h_end = h_start + K_H * dil_h
                            w_end = w_start + K_W * dil_w
                            # Extract appropriate input window
                            inp_slice = inp_group[:, h_start:h_end:dil_h, w_start:w_end:dil_w] # shape [g_Cin, K_H, K_W]
                            out[n, c_out, h, w] = lns_sum(lns_mul(inp_slice, weight_packed[c_out]), base)
                            if bias is not None:
                                out[n, c_out, h, w] = lns_add(out[n, c_out, h, w], bias[c_out], base)

        if squeeze_batch:
            out = out.squeeze(0)

        return out.to(torch.float64)

    @staticmethod
    def setup_context(ctx, inputs, output):
        x, weight, bias, base, stride, padding, dilation, groups = inputs
        ctx.save_for_backward(x, weight, bias, base)
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.groups = groups

    @staticmethod
    def backward(ctx, grad_output):
        x, weight, bias, base = ctx.saved_tensors

        # Canonicalize params
        stride = ctx.stride if isinstance(ctx.stride, tuple) else (ctx.stride, ctx.stride)
        padding = ctx.padding if isinstance(ctx.padding, tuple) else (ctx.padding, ctx.padding)
        dilation = ctx.dilation if isinstance(ctx.dilation, tuple) else (ctx.dilation, ctx.dilation)
        stride_h, stride_w = stride
        pad_h, pad_w = padding
        dil_h, dil_w = dilation
        groups = ctx.groups

        squeeze_batch = (x.dim() == 3)
        if grad_output.dim() == 3:
            grad_output = grad_output.unsqueeze(0)
            squeeze_batch = True

        N, C_out, H_out, W_out = grad_output.shape
        C_in = weight.shape[1] * groups
        K_H, K_W = weight.shape[2:]
        g_Cin = C_in // groups
        g_Cout = C_out // groups

        # Pad input for correct alignment
        if pad_h > 0 or pad_w > 0:
            x_padded = torch.nn.functional.pad(x, (pad_w, pad_w, pad_h, pad_h), value=LNS_ZERO.item())
        else:
            x_padded = x

        H_pad = x_padded.shape[2]
        W_pad = x_padded.shape[3]

        grad_x_padded = zeros(N, C_in, H_pad, W_pad, device=grad_output.device, b=base)._lns
        grad_weight = zeros_like(weight, b=base)._lns

        # Input gradient: for each pixel in input, sum all contributions from grad_output via the receptive field
        for n in range(N):
            for g in range(groups):
                in_start = g * g_Cin
                in_end = (g + 1) * g_Cin
                out_start = g * g_Cout
                out_end = (g + 1) * g_Cout
                for c_in in range(g_Cin):
                    for h_in in range(H_pad):
                        for w_in in range(W_pad):
                            grad = LNSTensor.get_internal_tensor(0, base)
                            for c_out in range(out_start, out_end):
                                w = weight[c_out, c_in, :, :]
                                for k_h in range(K_H):
                                    for k_w in range(K_W):
                                        # Figure out which output position this input contributes to
                                        h_out_nom = h_in - k_h * dil_h
                                        w_out_nom = w_in - k_w * dil_w
                                        if h_out_nom % stride_h == 0 and w_out_nom % stride_w == 0:
                                            h_out = h_out_nom // stride_h
                                            w_out = w_out_nom // stride_w
                                            if (0 <= h_out < H_out) and (0 <= w_out < W_out):
                                                grad = lns_add(
                                                    grad,
                                                    lns_mul(grad_output[n, c_out, h_out, w_out], w[k_h, k_w]),
                                                    base,
                                                )
                            grad_x_padded[n, in_start + c_in, h_in, w_in] = grad

        # Remove padding to match original input shape
        if pad_h > 0 or pad_w > 0:
            grad_x = grad_x_padded[:, :, pad_h: H_pad - pad_h, pad_w: W_pad - pad_w]
        else:
            grad_x = grad_x_padded

        # Weight gradient: correlate grad_output windows with input
        for g in range(groups):
            in_start = g * g_Cin
            in_end = (g + 1) * g_Cin
            out_start = g * g_Cout
            out_end = (g + 1) * g_Cout
            for c_out in range(out_start, out_end):
                for c_in in range(g_Cin):
                    for k_h in range(K_H):
                        for k_w in range(K_W):
                            grad = LNSTensor.get_internal_tensor(0, base)
                            for n in range(N):
                                for h_out in range(H_out):
                                    for w_out in range(W_out):
                                        h_in = h_out * stride_h + k_h * dil_h
                                        w_in = w_out * stride_w + k_w * dil_w
                                        inp_padded = x_padded[n, in_start + c_in, :, :]
                                        if (0 <= h_in < inp_padded.size(0)) and (0 <= w_in < inp_padded.size(1)):
                                            grad = lns_add(
                                                grad,
                                                lns_mul(grad_output[n, c_out, h_out, w_out], inp_padded[h_in, w_in]),
                                                base,
                                            )
                            grad_weight[c_out, c_in, k_h, k_w] = grad

        # Bias gradient: sum grad_output over batch, spatial dims
        if bias is not None:
            grad_bias = lns_sum(grad_output, base, dim=[0, 2, 3])
        else:
            grad_bias = None

        if squeeze_batch:
            grad_x = grad_x.squeeze(0)
        return grad_x, grad_weight, grad_bias, None, None, None, None, None

@implements(torch.nn.functional.conv2d, LNSConv2dFunction.forward, "default", default=True)
def conv2d(x, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):

    if bias is not None:
        x, weight, bias = format_lnstensor_operands(x, weight, bias)
        bias_lns = bias._lns
    else:
        x, weight = format_lnstensor_operands(x, weight)
        bias_lns = None

    result = LNSConv2dFunction.apply(x._lns, weight._lns, bias_lns, x.base,
                                     stride, padding, dilation, groups)

    return lnstensor(result, from_lns=True, b=x.base)

class LNSConv3dFunction(torch.autograd.Function):
    @staticmethod
    def forward(x, weight, bias, base, stride=1, padding=0, dilation=1, groups=1):
        # Standardize params to tuples
        if isinstance(stride, int): stride = (stride, stride, stride)
        if isinstance(padding, int): padding = (padding, padding, padding)
        if isinstance(dilation, int): dilation = (dilation, dilation, dilation)
        stride_d, stride_h, stride_w = stride
        pad_d, pad_h, pad_w = padding
        dil_d, dil_h, dil_w = dilation

        x_packed, weight_packed = x.to(torch.int64), weight.to(torch.int64)

        # Add batch dimension if not present (input shape: [N, C_in, D, H, W])
        squeeze_batch = (x.dim() == 4)
        if squeeze_batch:
            x_packed = x_packed.unsqueeze(0)

        N, C_in, D_in, H_in, W_in = x_packed.shape
        C_out, C_w, K_D, K_H, K_W = weight.shape

        assert C_in % groups == 0
        assert C_out % groups == 0
        assert C_w == C_in // groups

        g_Cin = C_in // groups
        g_Cout = C_out // groups

        # Padding
        if pad_d > 0 or pad_h > 0 or pad_w > 0:
            # PyTorch's pad format for 3D: (W_left, W_right, H_left, H_right, D_left, D_right)
            x_padded = torch.nn.functional.pad(
                x_packed,
                (pad_w, pad_w, pad_h, pad_h, pad_d, pad_d),
                value=LNS_ZERO.item()
            )
        else:
            x_padded = x_packed

        D_pad, H_pad, W_pad = x_padded.shape[2:]
        # Output shape from PyTorch:
        D_out = (D_in + 2 * pad_d - dil_d * (K_D - 1) - 1) // stride_d + 1
        H_out = (H_in + 2 * pad_h - dil_h * (K_H - 1) - 1) // stride_h + 1
        W_out = (W_in + 2 * pad_w - dil_w * (K_W - 1) - 1) // stride_w + 1
        out = zeros(N, C_out, D_out, H_out, W_out, device=x.device, b=base)._lns

        for n in range(N):
            for g in range(groups):
                inp_group = x_padded[n, g * g_Cin : (g + 1) * g_Cin]  # [g_Cin, D_pad, H_pad, W_pad]
                for c_out in range(g * g_Cout, (g + 1) * g_Cout):
                    for d in range(D_out):
                        for h in range(H_out):
                            for w in range(W_out):
                                d_start = d * stride_d
                                h_start = h * stride_h
                                w_start = w * stride_w
                                d_end = d_start + K_D * dil_d
                                h_end = h_start + K_H * dil_h
                                w_end = w_start + K_W * dil_w
                                inp_slice = inp_group[:,
                                    d_start:d_end:dil_d,
                                    h_start:h_end:dil_h,
                                    w_start:w_end:dil_w
                                ]  # [g_Cin, K_D, K_H, K_W]
                                out[n, c_out, d, h, w] = lns_sum(
                                    lns_mul(inp_slice, weight_packed[c_out]), base
                                )
                                if bias is not None:
                                    out[n, c_out, d, h, w] = lns_add(
                                        out[n, c_out, d, h, w], bias[c_out], base
                                    )

        if squeeze_batch:
            out = out.squeeze(0)
        return out.to(torch.float64)

    @staticmethod
    def setup_context(ctx, inputs, output):
        x, weight, bias, base, stride, padding, dilation, groups = inputs
        ctx.save_for_backward(x, weight, bias, base)
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.groups = groups

    @staticmethod
    def backward(ctx, grad_output):
        x, weight, bias, base = ctx.saved_tensors

        # Standardize params
        stride = ctx.stride if isinstance(ctx.stride, tuple) else (ctx.stride, ctx.stride, ctx.stride)
        padding = ctx.padding if isinstance(ctx.padding, tuple) else (ctx.padding, ctx.padding, ctx.padding)
        dilation = ctx.dilation if isinstance(ctx.dilation, tuple) else (ctx.dilation, ctx.dilation, ctx.dilation)
        stride_d, stride_h, stride_w = stride
        pad_d, pad_h, pad_w = padding
        dil_d, dil_h, dil_w = dilation
        groups = ctx.groups

        squeeze_batch = (x.dim() == 4)
        if grad_output.dim() == 4:
            grad_output = grad_output.unsqueeze(0)
            squeeze_batch = True

        N, C_out, D_out, H_out, W_out = grad_output.shape
        C_in = weight.shape[1] * groups
        K_D, K_H, K_W = weight.shape[2:]
        g_Cin = C_in // groups
        g_Cout = C_out // groups

        # Padding like in forward for alignment
        if pad_d > 0 or pad_h > 0 or pad_w > 0:
            x_padded = torch.nn.functional.pad(x, (pad_w, pad_w, pad_h, pad_h, pad_d, pad_d), value=LNS_ZERO.item())
        else:
            x_padded = x
        D_pad, H_pad, W_pad = x_padded.shape[2:]

        grad_x_padded = zeros(N, C_in, D_pad, H_pad, W_pad, device=grad_output.device, b=base)._lns
        grad_weight = zeros_like(weight, b=base)._lns

        # dL/dx
        for n in range(N):
            for g in range(groups):
                in_start = g * g_Cin
                in_end = (g + 1) * g_Cin
                out_start = g * g_Cout
                out_end = (g + 1) * g_Cout
                for c_in in range(g_Cin):
                    for d_in in range(D_pad):
                        for h_in in range(H_pad):
                            for w_in in range(W_pad):
                                grad = LNSTensor.get_internal_tensor(0, base)
                                for c_out in range(out_start, out_end):
                                    wgt = weight[c_out, c_in, :, :, :]
                                    for k_d in range(K_D):
                                        for k_h in range(K_H):
                                            for k_w in range(K_W):
                                                d_out_nom = d_in - k_d * dil_d
                                                h_out_nom = h_in - k_h * dil_h
                                                w_out_nom = w_in - k_w * dil_w
                                                if (d_out_nom % stride_d == 0 and h_out_nom % stride_h == 0 and w_out_nom % stride_w == 0):
                                                    d_out = d_out_nom // stride_d
                                                    h_out = h_out_nom // stride_h
                                                    w_out = w_out_nom // stride_w
                                                    if (0 <= d_out < D_out) and (0 <= h_out < H_out) and (0 <= w_out < W_out):
                                                        grad = lns_add(
                                                            grad,
                                                            lns_mul(grad_output[n, c_out, d_out, h_out, w_out], wgt[k_d, k_h, k_w]),
                                                            base
                                                        )
                                grad_x_padded[n, in_start + c_in, d_in, h_in, w_in] = grad

        # Remove padding to match input shape
        d0, d1 = pad_d, D_pad - pad_d
        h0, h1 = pad_h, H_pad - pad_h
        w0, w1 = pad_w, W_pad - pad_w
        grad_x = grad_x_padded[:, :, d0:d1, h0:h1, w0:w1] if (pad_d or pad_h or pad_w) else grad_x_padded

        # dL/dw
        for g in range(groups):
            in_start = g * g_Cin
            in_end = (g + 1) * g_Cin
            out_start = g * g_Cout
            out_end = (g + 1) * g_Cout
            for c_out in range(out_start, out_end):
                for c_in in range(g_Cin):
                    for k_d in range(K_D):
                        for k_h in range(K_H):
                            for k_w in range(K_W):
                                grad = LNSTensor.get_internal_tensor(0, base)
                                for n in range(N):
                                    for d_out in range(D_out):
                                        for h_out in range(H_out):
                                            for w_out in range(W_out):
                                                d_in = d_out * stride_d + k_d * dil_d
                                                h_in = h_out * stride_h + k_h * dil_h
                                                w_in = w_out * stride_w + k_w * dil_w
                                                inp_padded = x_padded[n, in_start + c_in, :, :, :]
                                                if (0 <= d_in < D_pad) and (0 <= h_in < H_pad) and (0 <= w_in < W_pad):
                                                    grad = lns_add(
                                                        grad,
                                                        lns_mul(grad_output[n, c_out, d_out, h_out, w_out], inp_padded[d_in, h_in, w_in]),
                                                        base
                                                    )
                                grad_weight[c_out, c_in, k_d, k_h, k_w] = grad

        # Bias gradient: sum grad_output over batch, spatial dims
        if bias is not None:
            grad_bias = lns_sum(grad_output, base, dim=[0, 2, 3, 4])
        else:
            grad_bias = None

        if squeeze_batch:
            grad_x = grad_x.squeeze(0)
        return grad_x, grad_weight, grad_bias, None, None, None, None, None

@implements(torch.nn.functional.conv3d, LNSConv3dFunction.forward, "default", default=True)
def conv3d(x, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):

    if bias is not None:
        x, weight, bias = format_lnstensor_operands(x, weight, bias)
        bias_lns = bias._lns
    else:
        x, weight = format_lnstensor_operands(x, weight)
        bias_lns = None

    result = LNSConv3dFunction.apply(x._lns, weight._lns, bias_lns, x.base,
                                     stride, padding, dilation, groups)

    return lnstensor(result, from_lns=True, b=x.base)
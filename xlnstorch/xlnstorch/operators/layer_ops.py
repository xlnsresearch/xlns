import warnings
import math
import torch
from xlnstorch import LNS_ZERO, LNS_ONE, CSRC_AVAILABLE, LNSTensor, lnstensor, format_lnstensor_operands, implements, zeros, zeros_like
from xlnstorch.autograd import LNSFunction
from . import (
    lns_mul,
    lns_sum,
    lns_add,
    lns_matmul,
    lns_div,
    lns_mean,
    lns_var,
    lns_sub,
    lns_neg,
    lns_sqrt,
    lns_max,
    lns_min,
)

class LNSLinearFunction(LNSFunction):
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
    else:
        x, weight = format_lnstensor_operands(x, weight)

    result = LNSLinearFunction.apply(x, weight, x.base, bias)

    return lnstensor(result, from_lns=True, b=x.base)

class LNSBilinearFunction(LNSFunction):
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
    else:
        x, y, weight = format_lnstensor_operands(x, y, weight)

    result = LNSBilinearFunction.apply(x, y, weight, x.base, bias)

    return lnstensor(result, from_lns=True, b=x.base)

class LNSDropoutFunction(LNSFunction):

    @staticmethod
    def forward(x, base, p=0.5):
        x_packed = x.to(torch.int64)

        mask = LNSTensor.get_internal_tensor(torch.bernoulli(torch.full_like(x, 1 - p)), base)
        result = lns_mul(x_packed, mask, base)

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
        grad_x = lns_mul(grad_output, grad_x, base)

        return grad_x, None, None

@implements(torch.nn.functional.dropout, LNSDropoutFunction.forward, "default", default=True)
def dropout(x, p=0.5, training=True, inplace=False):

    if not training or p == 0.0:
        return x

    if p < 0.0 or p > 1.0:
        raise ValueError(f"Dropout probability p must be in the range [0, 1], but got {p}.")

    result = LNSDropoutFunction.apply(x, x.base, p)

    if inplace:
        return x._inplace_copy(result)

    return lnstensor(result, from_lns=True, b=x.base)

class LNSDropout1dFunction(LNSFunction):

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

        result = lns_mul(x_packed, mask, base)
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
        grad_x = lns_mul(grad_output, grad_x, base)

        return grad_x, None, None

@implements(torch.nn.functional.dropout1d, LNSDropout1dFunction.forward, "default", default=True)
def dropout1d(x, p=0.5, training=True, inplace=False):

    if not training or p == 0.0:
        return x

    if p < 0.0 or p > 1.0:
        raise ValueError(f"Dropout probability p must be in the range [0, 1], but got {p}.")

    if x.dim() < 2 or x.dim() > 3:
        raise ValueError(f"Dropout1d expects a 2D or 3D tensor, but got a tensor with {x.dim()} dimensions.")

    result = LNSDropout1dFunction.apply(x, x.base, p)

    if inplace:
        return x._inplace_copy(result)

    return lnstensor(result, from_lns=True, b=x.base)

class LNSDropout2dFunction(LNSFunction):

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

        result = lns_mul(x_packed, mask, base)
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
        grad_x = lns_mul(grad_output, grad_x, base)

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

    result = LNSDropout2dFunction.apply(x, x.base, p)

    if inplace:
        return x._inplace_copy(result)

    return lnstensor(result, from_lns=True, b=x.base)

class LNSDropout3dFunction(LNSFunction):

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

        result = lns_mul(x_packed, mask, base)
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
        grad_x = lns_mul(grad_output, grad_x, base)

        return grad_x, None, None

@implements(torch.nn.functional.dropout3d, LNSDropout3dFunction.forward, "default", default=True)
def dropout3d(x, p=0.5, training=True, inplace=False):

    if not training or p == 0.0:
        return x

    if p < 0.0 or p > 1.0:
        raise ValueError(f"Dropout probability p must be in the range [0, 1], but got {p}.")

    if x.dim() < 4 or x.dim() > 5:
        raise ValueError(f"Dropout3d expects a 4D or 5D tensor, but got a tensor with {x.dim()} dimensions.")

    result = LNSDropout3dFunction.apply(x, x.base, p)

    if inplace:
        return x._inplace_copy(result)

    return lnstensor(result, from_lns=True, b=x.base)

class LNSConv1dFunction(LNSFunction):

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
                        out[n, c_out, l] = lns_sum(lns_mul(inp_slice, weight_packed[c_out], base), base)
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
                        grad = LNS_ZERO.clone()
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
                                        grad = lns_add(grad, lns_mul(grad_output[n, c_out, l_out], w[k], base), base)
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
                        grad = LNS_ZERO.clone()
                        # Sum over all samples and locations
                        for n in range(N):
                            for l_out in range(L_out):
                                l_in = l_out * ctx.stride + k * ctx.dilation
                                inp_padded = x_padded[n, in_start + c_in, :]
                                if ctx.padding <= l_in < inp_padded.size(0) - ctx.padding:
                                    grad = lns_add(grad, lns_mul(grad_output[n, c_out, l_out], inp_padded[l_in], base), base)
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

@implements(torch.nn.functional.conv1d, LNSConv1dFunction.forward, "default", default=not CSRC_AVAILABLE)
def conv1d(x, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):

    if bias is not None:
        x, weight, bias = format_lnstensor_operands(x, weight, bias)
    else:
        x, weight = format_lnstensor_operands(x, weight)

    result = LNSConv1dFunction.apply(x, weight, bias, x.base, stride,
                                     padding, dilation, groups)

    return lnstensor(result, from_lns=True, b=x.base)

class LNSConv2dFunction(LNSFunction):

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
                            out[n, c_out, h, w] = lns_sum(lns_mul(inp_slice, weight_packed[c_out], base), base)
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
                            grad = LNS_ZERO.clone()
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
                                                    lns_mul(grad_output[n, c_out, h_out, w_out], w[k_h, k_w], base),
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
                            grad = LNS_ZERO.clone()
                            for n in range(N):
                                for h_out in range(H_out):
                                    for w_out in range(W_out):
                                        h_in = h_out * stride_h + k_h * dil_h
                                        w_in = w_out * stride_w + k_w * dil_w
                                        inp_padded = x_padded[n, in_start + c_in, :, :]
                                        if (0 <= h_in < inp_padded.size(0)) and (0 <= w_in < inp_padded.size(1)):
                                            grad = lns_add(
                                                grad,
                                                lns_mul(grad_output[n, c_out, h_out, w_out], inp_padded[h_in, w_in], base),
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

@implements(torch.nn.functional.conv2d, LNSConv2dFunction.forward, "default", default=not CSRC_AVAILABLE)
def conv2d(x, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):

    if bias is not None:
        x, weight, bias = format_lnstensor_operands(x, weight, bias)
    else:
        x, weight = format_lnstensor_operands(x, weight)

    result = LNSConv2dFunction.apply(x, weight, bias, x.base, stride,
                                     padding, dilation, groups)

    return lnstensor(result, from_lns=True, b=x.base)

class LNSConv3dFunction(LNSFunction):
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
                                    lns_mul(inp_slice, weight_packed[c_out], base), base
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
                                grad = LNS_ZERO.clone()
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
                                                            lns_mul(grad_output[n, c_out, d_out, h_out, w_out], wgt[k_d, k_h, k_w], base),
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
                                grad = LNS_ZERO.clone()
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
                                                        lns_mul(grad_output[n, c_out, d_out, h_out, w_out], inp_padded[d_in, h_in, w_in], base),
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

@implements(torch.nn.functional.conv3d, LNSConv3dFunction.forward, "default", default=not CSRC_AVAILABLE)
def conv3d(x, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):

    if bias is not None:
        x, weight, bias = format_lnstensor_operands(x, weight, bias)
    else:
        x, weight = format_lnstensor_operands(x, weight)

    result = LNSConv3dFunction.apply(x, weight, bias, x.base, stride,
                                     padding, dilation, groups)

    return lnstensor(result, from_lns=True, b=x.base)

class LNSAvgPool1dFuncton(LNSFunction):

    @staticmethod
    def forward(x, kernel_size, base, stride=None, padding=0, ceil_mode=False, count_include_pad=True):
        if stride is None:
            stride = kernel_size

        if x.dim() == 2:
            x = x.unsqueeze(0)
            squeeze_batch = True
        else:
            squeeze_batch = False

        x_packed = x.to(torch.int64)
        N, C, L_in = x_packed.shape

        if padding > 0:
            x_padded = torch.nn.functional.pad(x_packed, (padding, padding), value=LNS_ZERO.item())
        else:
            x_padded = x_packed

        if ceil_mode:
            L_out = int(math.ceil((L_in + 2 * padding - kernel_size) / stride)) + 1
        else:
            L_out = (L_in + 2 * padding - kernel_size) // stride + 1

        out = zeros(N, C, L_out, device=x.device, b=base)._lns
        kernel_size_lns = LNSTensor.get_internal_tensor(kernel_size, base)

        for n in range(N):
            for c in range(C):
                for l_out in range(L_out):
                    start = l_out * stride
                    end = start + kernel_size
                    if end > x_padded.size(-1):
                        break

                    window = x_padded[n, c, start:end]
                    sm = lns_sum(window, base)

                    if count_include_pad:
                        divisor = kernel_size_lns
                    else:
                        left_pad = max(0, padding - start)
                        right_pad = max(0, end - (L_in + padding))
                        valid_count = kernel_size - (left_pad + right_pad)
                        divisor = LNSTensor.get_internal_tensor(max(valid_count, 1), base)

                    out[n, c, l_out] = lns_div(sm, divisor, base)

        if squeeze_batch:
            out = out.squeeze(0)

        return out.to(torch.float64)

    @staticmethod
    def setup_context(ctx, inputs, output):
        x, kernel_size, base, stride, padding, ceil_mode, count_include_pad = inputs
        if stride is None:
            stride = kernel_size
        ctx.save_for_backward(x, base)
        ctx.kernel_size = kernel_size
        ctx.stride = stride
        ctx.padding = padding
        ctx.ceil_mode = ceil_mode
        ctx.count_include_pad = count_include_pad

    @staticmethod
    def backward(ctx, grad_output):
        x, base = ctx.saved_tensors

        if x.dim() == 2:
            x = x.unsqueeze(0)
            grad_output = grad_output.unsqueeze(0)
            squeeze_batch = True
        else:
            squeeze_batch = False

        N, C, L_in = x.shape
        L_out = grad_output.size(-1)

        grad_x = zeros_like(x, b=base)._lns
        kernel_size_lns = LNSTensor.get_internal_tensor(ctx.kernel_size, base)

        for n in range(N):
            for c in range(C):
                for l_out in range(L_out):
                    start = l_out * ctx.stride
                    end = start + ctx.kernel_size

                    if start >= L_in + 2 * ctx.padding:
                        break

                    if ctx.count_include_pad:
                        divisor = kernel_size_lns
                    else:
                        left_pad = max(0, ctx.padding - start)
                        right_pad = max(0, end - (L_in + ctx.padding))
                        valid_count = ctx.kernel_size - (left_pad + right_pad)
                        divisor = LNSTensor.get_internal_tensor(max(valid_count, 1), base)

                    grad = grad_output[n, c, l_out]
                    for i in range(ctx.kernel_size):
                        idx = start + i - ctx.padding
                        if 0 <= idx < L_in:
                            grad_x[n, c, idx] = lns_add(grad_x[n, c, idx], lns_div(grad, divisor, base), base)

        if squeeze_batch:
            grad_x = grad_x.squeeze(0)

        return grad_x, None, None, None, None, None, None

@implements(torch.nn.functional.avg_pool1d, LNSAvgPool1dFuncton.forward, "default", default=not CSRC_AVAILABLE)
def avg_pool1d(x, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True):

    kernel_size = kernel_size[0] if isinstance(kernel_size, (list, tuple)) else kernel_size
    stride = stride[0] if isinstance(stride, (list, tuple)) else stride
    padding = padding[0] if isinstance(padding, (list, tuple)) else padding

    result = LNSAvgPool1dFuncton.apply(x, kernel_size, x.base, stride,
                                       padding, ceil_mode, count_include_pad)

    return lnstensor(result, from_lns=True, b=x.base)

class LNSAvgPool2dFuncton(LNSFunction):

    @staticmethod
    def forward(x, kernel_size, base, stride=None, padding=0, ceil_mode=False, count_include_pad=True, divisor_override=None):
        if stride is None:
            stride = kernel_size

        if isinstance(kernel_size, int): kernel_size = (kernel_size, kernel_size)
        if isinstance(stride, int): stride = (stride, stride)
        if isinstance(padding, int): padding = (padding, padding)
        kernel_w, kernel_h = kernel_size
        stride_h, stride_w = stride
        pad_h, pad_w = padding

        if x.dim() == 3:
            x = x.unsqueeze(0)
            squeeze_batch = True
        else:
            squeeze_batch = False

        x_packed = x.to(torch.int64)
        N, C, H_in, W_in = x_packed.shape

        if pad_h > 0 or pad_w > 0:
            x_padded = torch.nn.functional.pad(x_packed, (pad_w, pad_w, pad_h, pad_h), value=LNS_ZERO.item())
        else:
            x_padded = x_packed

        if ceil_mode:
            H_out = int(math.ceil((H_in + 2 * pad_h - kernel_h) / stride_h)) + 1
            W_out = int(math.ceil((W_in + 2 * pad_w - kernel_w) / stride_w)) + 1
        else:
            H_out = (H_in + 2 * pad_h - kernel_h) // stride_h + 1
            W_out = (W_in + 2 * pad_w - kernel_w) // stride_w + 1

        out = zeros(N, C, H_out, W_out, device=x.device, b=base)._lns
        kernel_area_lns = LNSTensor.get_internal_tensor(kernel_h * kernel_w, base)

        for n in range(N):
            for c in range(C):
                for h_out in range(H_out):
                    for w_out in range(W_out):
                        h_start = h_out * stride_h
                        h_end = h_start + kernel_h
                        w_start = w_out * stride_w
                        w_end = w_start + kernel_w
                        if h_end > x_padded.size(-2) or w_end > x_padded.size(-1):
                            continue

                        window = x_padded[n, c, h_start:h_end, w_start:w_end]
                        sm = lns_sum(window, base)

                        if divisor_override is not None:
                            divisor = divisor_override
                        elif count_include_pad:
                            divisor = kernel_area_lns
                        else:
                            left_pad = max(0, pad_w - w_start)
                            right_pad = max(0, w_end - (W_in + pad_w))
                            top_pad = max(0, pad_h - h_start)
                            bot_pad = max(0, h_end - (H_in + pad_h))
                            valid_h = kernel_h - (top_pad + bot_pad)
                            valid_w = kernel_w - (left_pad + right_pad)
                            valid_count = max(valid_h, 0) * max(valid_w, 0)
                            divisor = LNSTensor.get_internal_tensor(max(valid_count, 1), base)

                        out[n, c, h_out, w_out] = lns_div(sm, divisor, base)

        if squeeze_batch:
            out = out.squeeze(0)

        return out.to(torch.float64)

    @staticmethod
    def setup_context(ctx, inputs, output):
        x, kernel_size, base, stride, padding, ceil_mode, count_include_pad, divisor_override = inputs
        if stride is None:
            stride = kernel_size
        if isinstance(kernel_size, int): kernel_size = (kernel_size, kernel_size)
        if isinstance(stride, int): stride = (stride, stride)
        if isinstance(padding, int): padding = (padding, padding)
        if divisor_override is None:
            ctx.divisor_override = False
            ctx.save_for_backward(x, base)
        else:
            ctx.divisor_override = True
            ctx.save_for_backward(x, divisor_override, base)
        ctx.kernel_size = kernel_size
        ctx.stride = stride
        ctx.padding = padding
        ctx.ceil_mode = ceil_mode
        ctx.count_include_pad = count_include_pad

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.divisor_override:
            x, divisor_override, base = ctx.saved_tensors
        else:
            x, base = ctx.saved_tensors

        kernel_h, kernel_w = ctx.kernel_size
        stride_h, stride_w = ctx.stride
        pad_h, pad_w = ctx.padding

        if x.dim() == 3:
            x = x.unsqueeze(0)
            grad_output = grad_output.unsqueeze(0)
            squeeze_batch = True
        else:
            squeeze_batch = False

        N, C, H_in, W_in = x.shape
        H_out, W_out = grad_output.shape[-2:]

        grad_x = zeros_like(x, b=base)._lns
        kernel_area_lns = LNSTensor.get_internal_tensor(kernel_h * kernel_w, base)

        for n in range(N):
            for c in range(C):
                for h_out in range(H_out):
                    for w_out in range(W_out):
                        h_start = h_out * stride_h
                        h_end = h_start + kernel_h
                        w_start = w_out * stride_w
                        w_end = w_start + kernel_w

                        if ctx.divisor_override:
                            divisor = divisor_override
                        elif ctx.count_include_pad:
                            divisor = kernel_area_lns
                        else:
                            left_pad = max(0, pad_w - w_start)
                            right_pad = max(0, w_end - (W_in + pad_w))
                            top_pad = max(0, pad_h - h_start)
                            bot_pad = max(0, h_end - (H_in + pad_h))
                            valid_h = kernel_h - (top_pad + bot_pad)
                            valid_w = kernel_w - (left_pad + right_pad)
                            valid_count = max(valid_h, 0) * max(valid_w, 0)
                            divisor = LNSTensor.get_internal_tensor(max(valid_count, 1), base)

                        grad = grad_output[n, c, h_out, w_out]
                        for i in range(kernel_h):
                            for j in range(kernel_w):
                                h_idx = h_start + i - pad_h
                                w_idx = w_start + j - pad_w
                                if 0 <= h_idx < H_in and 0 <= w_idx < W_in:
                                    grad_x[n, c, h_idx, w_idx] = lns_add(
                                        grad_x[n, c, h_idx, w_idx],
                                        lns_div(grad, divisor, base),
                                        base
                                    )

        if squeeze_batch:
            grad_x = grad_x.squeeze(0)

        return grad_x, None, None, None, None, None, None, None

@implements(torch.nn.functional.avg_pool2d, LNSAvgPool2dFuncton.forward, "default", default=True)
def avg_pool2d(x, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True, divisor_override=None):

    if divisor_override is not None:
        x, divisor_override = format_lnstensor_operands(x, divisor_override)

    result = LNSAvgPool2dFuncton.apply(x, kernel_size, x.base, stride, padding,
                                       ceil_mode, count_include_pad, divisor_override)

    return lnstensor(result, from_lns=True, b=x.base)

class LNSAvgPool3dFuncton(LNSFunction):

    @staticmethod
    def forward(x, kernel_size, base, stride=None, padding=0, ceil_mode=False, count_include_pad=True, divisor_override=None):
        if stride is None:
            stride = kernel_size

        if isinstance(kernel_size, int): kernel_size = (kernel_size, kernel_size, kernel_size)
        if isinstance(stride, int): stride = (stride, stride, stride)
        if isinstance(padding, int): padding = (padding, padding, padding)
        kernel_d, kernel_w, kernel_h = kernel_size
        stride_d, stride_h, stride_w = stride
        pad_d, pad_h, pad_w = padding

        if x.dim() == 4:
            x = x.unsqueeze(0)
            squeeze_batch = True
        else:
            squeeze_batch = False

        x_packed = x.to(torch.int64)
        N, C, D_in, H_in, W_in = x_packed.shape

        if pad_d > 0 or pad_h > 0 or pad_w > 0:
            x_padded = torch.nn.functional.pad(x_packed, (pad_w, pad_w, pad_h, pad_h, pad_d, pad_d), value=LNS_ZERO.item())
        else:
            x_padded = x_packed

        if ceil_mode:
            D_out = int(math.ceil((D_in + 2 * pad_d - kernel_d) / stride_d)) + 1
            H_out = int(math.ceil((H_in + 2 * pad_h - kernel_h) / stride_h)) + 1
            W_out = int(math.ceil((W_in + 2 * pad_w - kernel_w) / stride_w)) + 1
        else:
            D_out = (D_in + 2 * pad_d - kernel_d) // stride_d + 1
            H_out = (H_in + 2 * pad_h - kernel_h) // stride_h + 1
            W_out = (W_in + 2 * pad_w - kernel_w) // stride_w + 1

        out = zeros(N, C, D_out, H_out, W_out, device=x.device, b=base)._lns
        kernel_vol_lns = LNSTensor.get_internal_tensor(kernel_d * kernel_h * kernel_w, base)

        for n in range(N):
            for c in range(C):
                for d_out in range(D_out):
                    for h_out in range(H_out):
                        for w_out in range(W_out):
                            d_start = d_out * stride_d
                            d_end = d_start + kernel_d
                            h_start = h_out * stride_h
                            h_end = h_start + kernel_h
                            w_start = w_out * stride_w
                            w_end = w_start + kernel_w
                            if d_end > x_padded.size(-3) or h_end > x_padded.size(-2) or w_end > x_padded.size(-1):
                                continue

                            window = x_padded[n, c, d_start:d_end, h_start:h_end, w_start:w_end]
                            sm = lns_sum(window, base)

                            if divisor_override is not None:
                                divisor = divisor_override
                            elif count_include_pad:
                                divisor = kernel_vol_lns
                            else:
                                front_pad = max(0, pad_d - d_start)
                                back_pad = max(0, d_end - (D_in + pad_d))
                                top_pad = max(0, pad_h - h_start)
                                bot_pad = max(0, h_end - (H_in + pad_h))
                                left_pad = max(0, pad_w - w_start)
                                right_pad = max(0, w_end - (W_in + pad_w))
                                valid_d = kernel_d - (front_pad + back_pad)
                                valid_h = kernel_h - (top_pad + bot_pad)
                                valid_w = kernel_w - (left_pad + right_pad)
                                valid_count = max(valid_d, 0) * max(valid_h, 0) * max(valid_w, 0)
                                divisor = LNSTensor.get_internal_tensor(max(valid_count, 1), base)

                            out[n, c, d_out, h_out, w_out] = lns_div(sm, divisor, base)

        if squeeze_batch:
            out = out.squeeze(0)

        return out.to(torch.float64)

    @staticmethod
    def setup_context(ctx, inputs, output):
        x, kernel_size, base, stride, padding, ceil_mode, count_include_pad, divisor_override = inputs
        if stride is None:
            stride = kernel_size
        if isinstance(kernel_size, int): kernel_size = (kernel_size, kernel_size, kernel_size)
        if isinstance(stride, int): stride = (stride, stride, stride)
        if isinstance(padding, int): padding = (padding, padding, padding)
        if divisor_override is None:
            ctx.divisor_override = False
            ctx.save_for_backward(x, base)
        else:
            ctx.divisor_override = True
            ctx.save_for_backward(x, divisor_override, base)
        ctx.kernel_size = kernel_size
        ctx.stride = stride
        ctx.padding = padding
        ctx.ceil_mode = ceil_mode
        ctx.count_include_pad = count_include_pad

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.divisor_override:
            x, divisor_override, base = ctx.saved_tensors
        else:
            x, base = ctx.saved_tensors

        kernel_d, kernel_h, kernel_w = ctx.kernel_size
        stride_d, stride_h, stride_w = ctx.stride
        pad_d, pad_h, pad_w = ctx.padding

        if x.dim() == 4:
            x = x.unsqueeze(0)
            grad_output = grad_output.unsqueeze(0)
            squeeze_batch = True
        else:
            squeeze_batch = False

        N, C, D_in, H_in, W_in = x.shape
        D_out, H_out, W_out = grad_output.shape[-3:]

        grad_x = zeros_like(x, b=base)._lns
        kernel_vol_lns = LNSTensor.get_internal_tensor(kernel_d * kernel_h * kernel_w, base)

        for n in range(N):
            for c in range(C):
                for d_out in range(D_out):
                    for h_out in range(H_out):
                        for w_out in range(W_out):
                            d_start = d_out * stride_d
                            d_end = d_start + kernel_d
                            h_start = h_out * stride_h
                            h_end = h_start + kernel_h
                            w_start = w_out * stride_w
                            w_end = w_start + kernel_w

                            if ctx.divisor_override:
                                divisor = divisor_override
                            elif ctx.count_include_pad:
                                divisor = kernel_vol_lns
                            else:
                                front_pad = max(0, pad_d - d_start)
                                back_pad = max(0, d_end - (D_in + pad_d))
                                top_pad = max(0, pad_h - h_start)
                                bot_pad = max(0, h_end - (H_in + pad_h))
                                left_pad = max(0, pad_w - w_start)
                                right_pad = max(0, w_end - (W_in + pad_w))
                                valid_d = kernel_d - (front_pad + back_pad)
                                valid_h = kernel_h - (top_pad + bot_pad)
                                valid_w = kernel_w - (left_pad + right_pad)
                                valid_count = max(valid_d, 0) * max(valid_h, 0) * max(valid_w, 0)
                                divisor = LNSTensor.get_internal_tensor(max(valid_count, 1), base)

                            grad = grad_output[n, c, d_out, h_out, w_out]
                            for di in range(kernel_d):
                                for hi in range(kernel_h):
                                    for wi in range(kernel_w):
                                        d_idx = d_start + di - pad_d
                                        h_idx = h_start + hi - pad_h
                                        w_idx = w_start + wi - pad_w
                                        if 0 <= d_idx < D_in and 0 <= h_idx < H_in and 0 <= w_idx < W_in:
                                            grad_x[n, c, d_idx, h_idx, w_idx] = lns_add(
                                                grad_x[n, c, d_idx, h_idx, w_idx],
                                                lns_div(grad, divisor, base),
                                                base
                                            )

        if squeeze_batch:
            grad_x = grad_x.squeeze(0)

        return grad_x, None, None, None, None, None, None, None, None

@implements(torch.nn.functional.avg_pool3d, LNSAvgPool3dFuncton.forward, "default", default=True)
def avg_pool3d(x, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True, divisor_override=None):

    if divisor_override is not None:
        x, divisor_override = format_lnstensor_operands(x, divisor_override)

    result = LNSAvgPool3dFuncton.apply(x, kernel_size, x.base, stride, padding,
                                       ceil_mode, count_include_pad, divisor_override)

    return lnstensor(result, from_lns=True, b=x.base)

class LNSAdaptiveAvgPool1dFunction(LNSFunction):

    @staticmethod
    def forward(x, output_size, base):
        if isinstance(output_size, int):
            output_size = (output_size,)

        assert len(output_size) == 1, "AdaptiveAvgPool1d only supports a single output length"
        L_out = output_size[0]

        if x.dim() == 2:
            x = x.unsqueeze(0)
            squeeze_batch = True
        else:
            squeeze_batch = False

        x_packed = x.to(torch.int64)
        N, C, L_in = x_packed.shape

        out = zeros(N, C, L_out, device=x.device, b=base)._lns

        for n in range(N):
            for c in range(C):
                for l_out in range(L_out):
                    start = int(math.floor(l_out * L_in / L_out))
                    end = int(math.ceil((l_out + 1) * L_in / L_out))
                    window = x_packed[n, c, start:end]

                    sm = lns_sum(window, base)
                    divisor = LNSTensor.get_internal_tensor(max(end - start, 1), base)

                    out[n, c, l_out] = lns_div(sm, divisor, base)

        if squeeze_batch:
            out = out.squeeze(0)

        return out.to(torch.float64)

    @staticmethod
    def setup_context(ctx, inputs, output):
        x, output_size, base = inputs
        ctx.save_for_backward(x, base)
        ctx.output_size = output_size

    @staticmethod
    def backward(ctx, grad_output):
        x, base = ctx.saved_tensors
        if isinstance(ctx.output_size, int):
            L_out = ctx.output_size
        else:
            L_out = ctx.output_size[0]

        if x.dim() == 2:
            x = x.unsqueeze(0)
            grad_output = grad_output.unsqueeze(0)
            squeeze_batch = True
        else:
            squeeze_batch = False

        N, C, L_in = x.shape
        grad_x = zeros_like(x, b=base)._lns

        for n in range(N):
            for c in range(C):
                for l_out in range(L_out):
                    start = int(math.floor(l_out * L_in / L_out))
                    end = int(math.ceil((l_out + 1) * L_in / L_out))
                    divisor = LNSTensor.get_internal_tensor(max(end - start, 1), base)
                    grad = lns_div(grad_output[n, c, l_out], divisor, base)
                    for idx in range(start, end):
                        grad_x[n, c, idx] = lns_add(grad_x[n, c, idx], grad, base)

        if squeeze_batch:
            grad_x = grad_x.squeeze(0)

        return grad_x, None, None

@implements(torch.nn.functional.adaptive_avg_pool1d, LNSAdaptiveAvgPool1dFunction.forward, "default", default=True)
def adaptive_avg_pool1d(x, output_size):

    result = LNSAdaptiveAvgPool1dFunction.apply(x, output_size, x.base)
    return lnstensor(result, from_lns=True, b=x.base)

class LNSAdaptiveAvgPool2dFunction(LNSFunction):

    @staticmethod
    def forward(x, output_size, base):
        if isinstance(output_size, int):
            H_out, W_out = output_size, output_size
        else:
            assert len(output_size) == 2, "output_size must be int or tuple of length 2"
            H_out, W_out = output_size

        if x.dim() == 3:
            x = x.unsqueeze(0)
            squeeze_batch = True
        else:
            squeeze_batch = False

        x_packed = x.to(torch.int64)
        N, C, H_in, W_in = x_packed.shape
        out = zeros(N, C, H_out, W_out, device=x.device, b=base)._lns

        for n in range(N):
            for c in range(C):
                for h_out in range(H_out):
                    h_start = int(math.floor(h_out * H_in / H_out))
                    h_end = int(math.ceil((h_out + 1) * H_in / H_out))
                    for w_out in range(W_out):
                        w_start = int(math.floor(w_out * W_in / W_out))
                        w_end = int(math.ceil((w_out + 1) * W_in / W_out))

                        window = x_packed[n, c, h_start:h_end, w_start:w_end]
                        sm = lns_sum(window, base)
                        divisor = LNSTensor.get_internal_tensor(
                            max((h_end - h_start) * (w_end - w_start), 1), base
                        )

                        out[n, c, h_out, w_out] = lns_div(sm, divisor, base)

        if squeeze_batch:
            out = out.squeeze(0)

        return out.to(torch.float64)

    @staticmethod
    def setup_context(ctx, inputs, output):
        x, output_size, base = inputs
        ctx.save_for_backward(x, base)
        ctx.output_size = output_size

    @staticmethod
    def backward(ctx, grad_output):
        x, base = ctx.saved_tensors
        if isinstance(ctx.output_size, int):
            H_out, W_out = ctx.output_size, ctx.output_size
        else:
            H_out, W_out = ctx.output_size

        if x.dim() == 3:
            x = x.unsqueeze(0)
            grad_output = grad_output.unsqueeze(0)
            squeeze_batch = True
        else:
            squeeze_batch = False

        N, C, H_in, W_in = x.shape
        grad_x = zeros_like(x, b=base)._lns

        for n in range(N):
            for c in range(C):
                for h_out in range(H_out):
                    h_start = int(math.floor(h_out * H_in / H_out))
                    h_end = int(math.ceil((h_out + 1) * H_in / H_out))
                    for w_out in range(W_out):
                        w_start = int(math.floor(w_out * W_in / W_out))
                        w_end = int(math.ceil((w_out + 1) * W_in / W_out))

                        divisor = LNSTensor.get_internal_tensor(
                            max((h_end - h_start) * (w_end - w_start), 1), base
                        )
                        grad = lns_div(grad_output[n, c, h_out, w_out], divisor, base)

                        for i in range(h_start, h_end):
                            for j in range(w_start, w_end):
                                grad_x[n, c, i, j] = lns_add(grad_x[n, c, i, j], grad, base)

        if squeeze_batch:
            grad_x = grad_x.squeeze(0)

        return grad_x, None, None

@implements(torch.nn.functional.adaptive_avg_pool2d, LNSAdaptiveAvgPool2dFunction.forward, "default", default=True)
def adaptive_avg_pool2d(x, output_size):

    result = LNSAdaptiveAvgPool2dFunction.apply(x, output_size, x.base)
    return lnstensor(result, from_lns=True, b=x.base)

class LNSAdaptiveAvgPool3dFunction(LNSFunction):

    @staticmethod
    def forward(x, output_size, base):
        if isinstance(output_size, int):
            D_out, H_out, W_out = output_size, output_size, output_size
        else:
            assert len(output_size) == 3, "output_size must be int or tuple of length 3"
            D_out, H_out, W_out = output_size

        if x.dim() == 4:
            x = x.unsqueeze(0)
            squeeze_batch = True
        else:
            squeeze_batch = False

        x_packed = x.to(torch.int64)
        N, C, D_in, H_in, W_in = x_packed.shape
        out = zeros(N, C, D_out, H_out, W_out, device=x.device, b=base)._lns

        for n in range(N):
            for c in range(C):
                for d_out in range(D_out):
                    d_start = int(math.floor(d_out * D_in / D_out))
                    d_end = int(math.ceil((d_out + 1) * D_in / D_out))
                    for h_out in range(H_out):
                        h_start = int(math.floor(h_out * H_in / H_out))
                        h_end = int(math.ceil((h_out + 1) * H_in / H_out))
                        for w_out in range(W_out):
                            w_start = int(math.floor(w_out * W_in / W_out))
                            w_end = int(math.ceil((w_out + 1) * W_in / W_out))

                            window = x_packed[n, c, d_start:d_end, h_start:h_end, w_start:w_end]
                            sm = lns_sum(window, base)
                            divisor = LNSTensor.get_internal_tensor(
                                max((d_end - d_start) * (h_end - h_start) * (w_end - w_start), 1), base
                            )
                            out[n, c, d_out, h_out, w_out] = lns_div(sm, divisor, base)

        if squeeze_batch:
            out = out.squeeze(0)

        return out.to(torch.float64)

    @staticmethod
    def setup_context(ctx, inputs, output):
        x, output_size, base = inputs
        ctx.save_for_backward(x, base)
        ctx.output_size = output_size

    @staticmethod
    def backward(ctx, grad_output):
        x, base = ctx.saved_tensors
        if isinstance(ctx.output_size, int):
            D_out, H_out, W_out = ctx.output_size, ctx.output_size, ctx.output_size
        else:
            D_out, H_out, W_out = ctx.output_size

        if x.dim() == 4:
            x = x.unsqueeze(0)
            grad_output = grad_output.unsqueeze(0)
            squeeze_batch = True
        else:
            squeeze_batch = False

        N, C, D_in, H_in, W_in = x.shape
        grad_x = zeros_like(x, b=base)._lns

        for n in range(N):
            for c in range(C):
                for d_out in range(D_out):
                    d_start = int(math.floor(d_out * D_in / D_out))
                    d_end = int(math.ceil((d_out + 1) * D_in / D_out))
                    for h_out in range(H_out):
                        h_start = int(math.floor(h_out * H_in / H_out))
                        h_end = int(math.ceil((h_out + 1) * H_in / H_out))
                        for w_out in range(W_out):
                            w_start = int(math.floor(w_out * W_in / W_out))
                            w_end = int(math.ceil((w_out + 1) * W_in / W_out))

                            divisor = LNSTensor.get_internal_tensor(
                                max((d_end - d_start) * (h_end - h_start) * (w_end - w_start), 1), base
                            )
                            grad = lns_div(grad_output[n, c, d_out, h_out, w_out], divisor, base)

                            for di in range(d_start, d_end):
                                for hi in range(h_start, h_end):
                                    for wi in range(w_start, w_end):
                                        grad_x[n, c, di, hi, wi] = lns_add(
                                            grad_x[n, c, di, hi, wi], grad, base
                                        )

        if squeeze_batch:
            grad_x = grad_x.squeeze(0)

        return grad_x, None, None

@implements(torch.nn.functional.adaptive_avg_pool3d, LNSAdaptiveAvgPool3dFunction.forward, "default", default=True)
def adaptive_avg_pool3d(x, output_size):

    result = LNSAdaptiveAvgPool3dFunction.apply(x, output_size, x.base)
    return lnstensor(result, from_lns=True, b=x.base)

class LNSBatchNormFunction(LNSFunction):

    @staticmethod
    def forward(x, running_mean, running_var, momentum, eps, base, weight=None, bias=None, training=False):

        red_dims = tuple(i for i in range(x.dim()) if i != 1)

        if training:
            batch_mean = lns_mean(x, base, dim=red_dims, keepdim=True)
            batch_var = lns_var(x, base, LNS_ZERO, dim=red_dims, keepdim=True)
            batch_var_corrected = lns_var(x, base, LNS_ONE, dim=red_dims, keepdim=True)

            with torch.no_grad():

                one_minus_momentum = lns_sub(LNS_ONE, momentum, base)

                new_running_mean = lns_add(
                    lns_mul(one_minus_momentum, running_mean, base),
                    lns_mul(momentum, batch_mean.squeeze(), base), base)
                running_mean.copy_(new_running_mean)

                new_running_var = lns_add(
                    lns_mul(one_minus_momentum, running_var, base),
                    lns_mul(momentum, batch_var_corrected.squeeze(), base), base)
                running_var.copy_(new_running_var)

            mean = batch_mean
            var = batch_var

        else:

            mean = running_mean.view(1, -1, *([1] * (x.dim() - 2)))
            var = running_var.view(1, -1, *([1] * (x.dim() - 2)))

        var_eps = lns_add(var, eps, base)
        inv_std = lns_div(LNS_ONE, lns_sqrt(var_eps, base), base)

        x_centered = lns_sub(x, mean, base)
        x_hat = lns_mul(x_centered, inv_std, base)

        if weight is not None:
            y = lns_mul(x_hat, weight.view(1, -1, *([1] * (x.dim() - 2))), base)
        else:
            y = x_hat

        if bias is not None:
            y = lns_add(y, bias.view(1, -1, *([1] * (x.dim() - 2))), base)

        return y.to(torch.float64)

    @staticmethod
    def setup_context(ctx, inputs, output):
        x, running_mean, running_var, _, eps, base, weight, bias, training = inputs

        ctx.red_dims = tuple(i for i in range(x.dim()) if i != 1)
        ctx.training = training

        if training:
            mean = lns_mean(x, base, dim=ctx.red_dims, keepdim=True)
            var = lns_var(x, base, LNS_ONE, dim=ctx.red_dims, keepdim=True)

        else:
            mean = running_mean.view(1, -1, *([1] * (x.dim() - 2)))
            var = running_var.view(1, -1, *([1] * (x.dim() - 2)))

        ctx.save_for_backward(x, weight, bias, mean, var, eps, base)

    @staticmethod
    def backward(ctx, grad_output):
        x, weight, bias, mean, var, eps, base = ctx.saved_tensors

        var_eps = lns_add(var, eps, base)
        inv_std = lns_div(LNS_ONE, lns_sqrt(var_eps, base), base)
        x_centered = lns_sub(x, mean, base)
        x_hat = lns_mul(x_centered, inv_std, base)

        grad_x = grad_weight = grad_bias = None

        if bias is not None:
            grad_bias = lns_sum(grad_output, base, dim=ctx.red_dims, keepdim=False)

        if weight is not None:
            grad_y_wrt_x_hat = lns_mul(grad_output, weight.view(1, -1, *([1] * (x.dim() - 2))), base)
            grad_weight = lns_sum(lns_mul(grad_output, x_hat, base),
                                  base, dim=ctx.red_dims, keepdim=False)
        else:
            grad_y_wrt_x_hat = grad_output

        if ctx.training:
            N = 1
            for dim in ctx.red_dims:
                N *= x.shape[dim]
            n_elems = LNSTensor.get_internal_tensor(N, base)

            var_eps = lns_add(var, eps, base)
            inv_std_cubed = lns_mul(inv_std, lns_mul(inv_std, inv_std, base), base)
            neg_half = LNSTensor.get_internal_tensor(-0.5, base)

            grad_var = lns_sum(lns_mul(
                lns_mul(grad_y_wrt_x_hat, x_centered, base),
                lns_mul(neg_half, inv_std_cubed, base),
            base), base, dim=ctx.red_dims, keepdim=True)

            neg_inv_std = lns_neg(inv_std)
            grad_mean_term1 = lns_sum(lns_mul(grad_y_wrt_x_hat, neg_inv_std, base),
                                      base, dim=ctx.red_dims, keepdim=True)

            neg_two = LNSTensor.get_internal_tensor(-2.0, base)
            neg_two_over_N = lns_div(neg_two, n_elems, base)
            sum_x_centered = lns_sum(x_centered, base, dim=ctx.red_dims, keepdim=True)
            grad_mean_term2 = lns_mul(lns_mul(grad_var, neg_two_over_N, base),
                                       sum_x_centered, base)

            grad_mean = lns_add(grad_mean_term1, grad_mean_term2, base)

            two = LNSTensor.get_internal_tensor(2.0, base)
            two_over_N = lns_div(two, n_elems, base)
            one_over_N = lns_div(LNS_ONE, n_elems, base)

            grad_x_term1 = lns_mul(grad_y_wrt_x_hat, inv_std, base)
            grad_x_term2 = lns_mul(lns_mul(grad_var, two_over_N, base),
                                   x_centered, base)
            grad_x_term3 = lns_mul(grad_mean, one_over_N, base)

            grad_x = lns_add(grad_x_term1, lns_add(grad_x_term2, grad_x_term3, base), base)

        else:
            grad_x = lns_mul(grad_y_wrt_x_hat, inv_std, base)

        return grad_x.to(torch.float64), None, None, None, None, None, grad_weight, grad_bias, None

@implements(torch.nn.functional.batch_norm, LNSBatchNormFunction.forward, "default", default=True)
def batch_norm(x, running_mean, running_var, weight=None, bias=None, training=False, momentum=0.1, eps=1e-5):

    x, running_mean_cpy, running_var_cpy, weight, bias, momentum, eps = format_lnstensor_operands(x, running_mean, running_var, weight, bias, momentum, eps)

    result = LNSBatchNormFunction.apply(x, running_mean_cpy, running_var_cpy, momentum, eps, x.base, weight, bias, training)

    if training:
        running_mean._inplace_copy(running_mean_cpy._lns)
        running_var._inplace_copy(running_var_cpy._lns)

    return lnstensor(result, from_lns=True, b=x.base)

class LNSLayerNorm(LNSFunction):
    """
    LNS version of nn.LayerNorm (forward pass only).

    x  : input in log–number system
    eps: small constant in the same LNS base that is added for numerical stability
    base: LNS base used by the helper ops
    normalized_shape : tuple or list indicating which (trailing) dims are normalised
                       e.g. for PyTorch’s nn.LayerNorm(normalized_shape=(C,H,W)) you
                       would pass (C, H, W).
    weight, bias (optional): element-wise affine parameters, given in LNS
    """
    @staticmethod
    def forward(x, eps, base, normalized_shape, weight=None, bias=None):
        reduce_dims = tuple(range(x.dim() - len(normalized_shape), x.dim()))

        mean = lns_mean(x, base, dim=reduce_dims, keepdim=True)
        var = lns_var(x, base, LNS_ZERO, dim=reduce_dims, keepdim=True)

        var_eps = lns_add(var, eps, base)
        inv_std = lns_div(LNS_ONE, lns_sqrt(var_eps, base), base)

        x_hat = lns_mul(lns_sub(x, mean, base), inv_std, base)

        if weight is not None:
            shape = [1] * (x.dim() - len(normalized_shape)) + list(normalized_shape)
            x_hat = lns_mul(x_hat, weight.view(*shape), base)

        if bias is not None:
            shape = [1] * (x.dim() - len(normalized_shape)) + list(normalized_shape)
            x_hat = lns_add(x_hat, bias.view(*shape), base)

        return x_hat.to(torch.float64)

    @staticmethod
    def setup_context(ctx, inputs, output):
        x, eps, base, normalized_shape, weight, bias = inputs

        ctx.red_dims = tuple(range(x.dim() - len(normalized_shape), x.dim()))

        mean = lns_mean(x, base, dim=ctx.red_dims, keepdim=True)
        var = lns_var(x, base, LNS_ZERO, dim=ctx.red_dims, keepdim=True)

        ctx.save_for_backward(x, weight, bias, mean, var, eps, base)

    @staticmethod
    def backward(ctx, grad_output):
        x, weight, bias, mean, var, eps, base = ctx.saved_tensors
        red_dims = ctx.red_dims

        var_eps = lns_add(var, eps, base)
        inv_std = lns_div(LNS_ONE, lns_sqrt(var_eps, base), base)
        x_centered = lns_sub(x, mean, base)
        x_hat = lns_mul(x_centered, inv_std, base)

        param_red_dims = tuple(i for i in range(x.dim()) if i not in red_dims)

        grad_bias = None
        if bias is not None:
            grad_bias = lns_sum(grad_output, base, dim=param_red_dims, keepdim=False)

        grad_weight = None
        if weight is not None:

            w_view = [1] * x.dim()
            for d in red_dims:
                w_view[d] = x.size(d)
            weight_b = weight.view(*w_view)

            grad_y_wrt_x_hat = lns_mul(grad_output, weight_b, base)
            grad_weight = lns_sum(
                lns_mul(grad_output, x_hat, base),
                base, dim=param_red_dims, keepdim=False
            )

        else:
            grad_y_wrt_x_hat = grad_output

        N = 1
        for d in red_dims:
            N *= x.shape[d]
        n_elems = LNSTensor.get_internal_tensor(N, base)

        sum_grad = lns_sum(grad_y_wrt_x_hat, base, dim=red_dims, keepdim=True)
        sum_grad_xhat = lns_sum(
            lns_mul(grad_y_wrt_x_hat, x_hat, base),
            base, dim=red_dims, keepdim=True
        )

        Ng = lns_mul(n_elems, grad_y_wrt_x_hat, base)
        term_inner = lns_sub(
            lns_sub(Ng, sum_grad, base),
            lns_mul(x_hat, sum_grad_xhat, base),
            base
        )

        inv_N = lns_div(LNS_ONE, n_elems, base)
        grad_x = lns_mul(
            lns_mul(inv_std, inv_N, base),
            term_inner, base
        )

        return grad_x.to(torch.float64), None, None, None, grad_weight, grad_bias

@implements(torch.nn.functional.layer_norm, LNSLayerNorm.forward, "default", default=True)
def layer_norm(x, normalized_shape, weight=None, bias=None, eps=1e-5):

    x, weight, bias, eps = format_lnstensor_operands(x, weight, bias, eps)
    result = LNSLayerNorm.apply(x, eps, x.base, normalized_shape, weight, bias)

    return lnstensor(result, from_lns=True, b=x.base)

class LNSMaxPool1dFunction(LNSFunction):

    @staticmethod
    def forward(x, kernel_size, base, stride=None, padding=0, dilation=1, ceil_mode=False, return_indices=False):
        if stride is None:
            stride = kernel_size

        if isinstance(kernel_size, (tuple, list)): kernel_size = int(kernel_size[0])
        if isinstance(stride, (tuple, list)): stride = int(stride[0])
        if isinstance(padding, (tuple, list)): padding = int(padding[0])
        if isinstance(dilation, (tuple, list)): dilation = int(dilation[0])

        if x.dim() == 2:
            x = x.unsqueeze(0)
            squeeze_batch = True
        else:
            squeeze_batch = False

        x_packed = x.to(torch.int64)
        N, C, L_in = x_packed.shape

        if padding > 0:
            x_min = lns_min(x_packed, base)
            fill = lns_sub(x_min, LNS_ONE, base).item()
            x_padded = torch.nn.functional.pad(x_packed, (padding, padding), value=fill)
        else:
            x_padded = x_packed

        eff_k = (kernel_size - 1) * dilation + 1

        if ceil_mode:
            L_out = int(math.ceil((L_in + 2 * padding - eff_k) / stride)) + 1
        else:
            L_out = (L_in + 2 * padding - eff_k) // stride + 1

        out_vals = zeros(N, C, L_out, device=x.device, b=base)._lns
        if return_indices:
            out_idx = torch.empty((N, C, L_out), dtype=torch.int64, device=x.device)

        padded_L = x_padded.size(-1)

        for n in range(N):
            for c in range(C):
                for l_out in range(L_out):
                    l_start = l_out * stride
                    l_end = l_start + eff_k

                    if l_start >= padded_L:
                        continue

                    l_end_eff = min(l_end, padded_L)

                    window = x_padded[n, c, l_start:l_end_eff:dilation]
                    window_flat = window.reshape(-1)

                    # pass dim to get indices
                    mx_val, mx_idx = lns_max(window_flat, base, dim=0)
                    out_vals[n, c, l_out] = mx_val

                    if return_indices:
                        i = int(mx_idx)
                        l_in_idx = (l_start + i * dilation) - padding
                        out_idx[n, c, l_out] = l_in_idx

        if squeeze_batch:
            out_vals = out_vals.squeeze(0)
            if return_indices:
                out_idx = out_idx.squeeze(0)

        if return_indices:
            return out_vals.to(torch.float64), out_idx
        else:
            return out_vals.to(torch.float64)

    @staticmethod
    def setup_context(ctx, inputs, output):
        x, kernel_size, base, stride, padding, dilation, ceil_mode, return_indices = inputs

        if stride is None:
            stride = kernel_size

        # Normalize to ints for 1D
        if isinstance(kernel_size, (tuple, list)): kernel_size = int(kernel_size[0])
        if isinstance(stride, (tuple, list)): stride = int(stride[0])
        if isinstance(padding, (tuple, list)): padding = int(padding[0])
        if isinstance(dilation, (tuple, list)): dilation = int(dilation[0])

        ctx.save_for_backward(x, base)
        ctx.kernel_size = int(kernel_size)
        ctx.stride = int(stride)
        ctx.padding = int(padding)
        ctx.dilation = int(dilation)
        ctx.ceil_mode = bool(ceil_mode)
        ctx.return_indices = bool(return_indices)

    @staticmethod
    def backward(ctx, grad_output, grad_out_indices=None):
        x, base = ctx.saved_tensors

        kernel = ctx.kernel_size
        stride = ctx.stride
        pad = ctx.padding
        dil = ctx.dilation

        if x.dim() == 2:
            x = x.unsqueeze(0)
            grad_output = grad_output.unsqueeze(0)
            squeeze_batch = True
        else:
            squeeze_batch = False

        N, C, L_in = x.shape
        x_packed = x.to(torch.int64)

        if pad > 0:
            x_min = lns_min(x, base)
            fill  = lns_sub(x_min, LNS_ONE, base).item()
            x_padded = torch.nn.functional.pad(x_packed, (pad, pad), value=fill)
        else:
            x_padded = x_packed

        padded_L = x_padded.shape[-1]
        eff_k = (kernel - 1) * dil + 1

        if ctx.ceil_mode:
            L_out = int(math.ceil((L_in + 2 * pad - eff_k) / stride)) + 1
        else:
            L_out = (L_in + 2 * pad - eff_k) // stride + 1

        grad_padded = zeros_like(x_padded, b=base)._lns

        for n in range(N):
            for c in range(C):
                for l_out in range(L_out):
                    l_start = l_out * stride
                    l_end = l_start + eff_k

                    if l_start >= padded_L:
                        continue

                    l_end_eff = min(l_end, padded_L)

                    window = x_padded[n, c, l_start:l_end_eff:dil]
                    window_flat = window.reshape(-1)

                    _, mx_idx = lns_max(window_flat, base, dim=0)

                    i = int(mx_idx)
                    l_in_idx = l_start + i * dil

                    grad_padded[n, c, l_in_idx] = lns_add(
                        grad_padded[n, c, l_in_idx],
                        grad_output[n, c, l_out],
                        base
                    )

        if pad > 0:
            grad_padded = grad_padded[:, :, pad:pad + L_in]

        grad_x = grad_padded.squeeze(0) if squeeze_batch else grad_padded
        return grad_x, None, None, None, None, None, None, None

@implements(torch.nn.functional.max_pool1d_with_indices, LNSMaxPool1dFunction.forward, "default", default=True)
@implements(torch.nn.functional.max_pool1d, LNSMaxPool1dFunction.forward, "default", default=True)
def max_pool1d(x, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False, return_indices=False):

    result = LNSMaxPool1dFunction.apply(x, kernel_size, x.base, stride, padding, dilation, ceil_mode, return_indices)

    if return_indices:
        return lnstensor(result[0], from_lns=True, b=x.base), result[1]
    else:
        return lnstensor(result, from_lns=True, b=x.base)

class LNSMaxPool2dFunction(LNSFunction):

    @staticmethod
    def forward(x, kernel_size, base, stride=None, padding=0, dilation=1, ceil_mode=False, return_indices=False):
        if stride is None:
            stride = kernel_size

        if isinstance(kernel_size, int): kernel_size = (kernel_size, kernel_size)
        if isinstance(stride, int): stride = (stride, stride)
        if isinstance(padding, int): padding = (padding, padding)
        if isinstance(dilation, int): dilation = (dilation, dilation)
        kernel_h, kernel_w = kernel_size
        stride_h, stride_w = stride
        pad_h, pad_w = padding
        dil_h, dil_w = dilation

        if x.dim() == 3:
            x = x.unsqueeze(0)
            squeeze_batch = True
        else:
            squeeze_batch = False

        x_packed = x.to(torch.int64)
        N, C, H_in, W_in = x_packed.shape

        if pad_h > 0 or pad_w > 0:
            x_min = lns_min(x_packed, base)
            fill = lns_sub(x_min, LNS_ONE, base).item()
            x_padded = torch.nn.functional.pad(x_packed, (pad_w, pad_w, pad_h, pad_h), value=fill)
        else:
            x_padded = x_packed

        eff_kh = (kernel_h - 1) * dil_h + 1
        eff_kw = (kernel_w - 1) * dil_w + 1

        if ceil_mode:
            H_out = int(math.ceil((H_in + 2 * pad_h - eff_kh) / stride_h)) + 1
            W_out = int(math.ceil((W_in + 2 * pad_w - eff_kw) / stride_w)) + 1
        else:
            H_out = (H_in + 2 * pad_h - eff_kh) // stride_h + 1
            W_out = (W_in + 2 * pad_w - eff_kw) // stride_w + 1

        out_vals = zeros(N, C, H_out, W_out, device=x.device, b=base)._lns
        if return_indices:
            out_idx = torch.empty((N, C, H_out, W_out), dtype=torch.int64, device=x.device)

        padded_H = x_padded.size(-2)
        padded_W = x_padded.size(-1)

        for n in range(N):
            for c in range(C):
                for h_out in range(H_out):
                    h_start = h_out * stride_h
                    h_end = h_start + eff_kh

                    if h_start >= padded_H:
                        continue

                    for w_out in range(W_out):
                        w_start = w_out * stride_w
                        w_end = w_start + eff_kw

                        if w_start >= padded_W:
                            continue

                        h_end_eff = min(h_end, padded_H)
                        w_end_eff = min(w_end, padded_W)

                        window = x_padded[n, c, h_start:h_end_eff:dil_h, w_start:w_end_eff:dil_w]
                        window_flat = window.reshape(-1)

                         # pass dim to get indices
                        mx_val, mx_idx = lns_max(window_flat, base, dim=0)
                        out_vals[n, c, h_out, w_out] = mx_val

                        len_w = (w_end_eff - w_start + dil_w - 1) // dil_w

                        if return_indices:
                            mx_idx_int = int(mx_idx)
                            i = mx_idx_int // len_w
                            j = mx_idx_int % len_w
                            h_in_idx = (h_start + i * dil_h) - pad_h
                            w_in_idx = (w_start + j * dil_w) - pad_w
                            out_idx[n, c, h_out, w_out] = h_in_idx * W_in + w_in_idx

        if squeeze_batch:
            out_vals = out_vals.squeeze(0)
            if return_indices:
                out_idx = out_idx.squeeze(0)

        if return_indices:
            return out_vals.to(torch.float64), out_idx
        else:
            return out_vals.to(torch.float64)

    @staticmethod
    def setup_context(ctx, inputs, output):
        x, kernel_size, base, stride, padding, dilation, ceil_mode, return_indices = inputs

        if stride is None:
            stride = kernel_size

        if isinstance(kernel_size, int): kernel_size = (kernel_size, kernel_size)
        if isinstance(stride, int): stride = (stride, stride)
        if isinstance(padding, int): padding = (padding, padding)
        if isinstance(dilation, int): dilation = (dilation, dilation)

        ctx.save_for_backward(x, base)
        ctx.kernel_size = kernel_size
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.ceil_mode = ceil_mode
        ctx.return_indices = return_indices

    @staticmethod
    def backward(ctx, grad_output, grad_out_indices=None):
        x, base = ctx.saved_tensors

        kernel_h, kernel_w = ctx.kernel_size
        stride_h, stride_w = ctx.stride
        pad_h, pad_w = ctx.padding
        dil_h, dil_w = ctx.dilation

        if x.dim() == 3:
            x = x.unsqueeze(0)
            grad_output = grad_output.unsqueeze(0)
            squeeze_batch = True
        else:
            squeeze_batch = False

        N, C, H_in, W_in = x.shape
        x_packed = x.to(torch.int64)

        if pad_h > 0 or pad_w > 0:
            x_min = lns_min(x, base)
            fill  = lns_sub(x_min, LNS_ONE, base).item()
            x_padded = torch.nn.functional.pad(x_packed, (pad_w, pad_w, pad_h, pad_h), value=fill)

        else:
            x_padded = x_packed

        padded_H, padded_W = x_padded.shape[-2:]
        eff_kh = (kernel_h - 1) * dil_h + 1
        eff_kw = (kernel_w - 1) * dil_w + 1

        if ctx.ceil_mode:
            H_out = int(math.ceil((H_in + 2 * pad_h - eff_kh) / stride_h)) + 1
            W_out = int(math.ceil((W_in + 2 * pad_w - eff_kw) / stride_w)) + 1
        else:
            H_out = (H_in + 2 * pad_h - eff_kh) // stride_h + 1
            W_out = (W_in + 2 * pad_w - eff_kw) // stride_w + 1

        grad_padded = zeros_like(x_padded, b=base)._lns
        for n in range(N):
            for c in range(C):
                for h_out in range(H_out):
                    h_start = h_out * stride_h
                    h_end   = h_start + eff_kh

                    if h_start >= padded_H:
                        continue

                    for w_out in range(W_out):
                        w_start = w_out * stride_w
                        w_end   = w_start + eff_kw

                        if w_start >= padded_W:
                            continue

                        h_end_eff = min(h_end, padded_H)
                        w_end_eff = min(w_end, padded_W)

                        len_h = (h_end_eff - h_start + dil_h - 1) // dil_h
                        len_w = (w_end_eff - w_start + dil_w - 1) // dil_w
                        if len_h <= 0 or len_w <= 0:
                            continue

                        window = x_padded[n, c, h_start:h_end_eff:dil_h, w_start:w_end_eff:dil_w]
                        window_flat = window.reshape(-1)

                        _, mx_idx = lns_max(window_flat, base, dim=0)

                        mx_idx = int(mx_idx)
                        i = mx_idx // len_w
                        j = mx_idx %  len_w

                        h_in_idx = h_start + i * dil_h
                        w_in_idx = w_start + j * dil_w

                        grad_padded[n, c, h_in_idx, w_in_idx] = lns_add(grad_padded[n, c, h_in_idx, w_in_idx],
                                                                        grad_output[n, c, h_out, w_out],
                                                                        base)

        if pad_h > 0 or pad_w > 0:
            grad_padded = grad_padded[:, :, pad_h:pad_h + H_in, pad_w:pad_w + W_in]

        grad_x = grad_padded.squeeze(0) if squeeze_batch else grad_padded
        return grad_x, None, None, None, None, None, None, None

@implements(torch.nn.functional.max_pool2d_with_indices, LNSMaxPool2dFunction.forward, "default", default=True)
@implements(torch.nn.functional.max_pool2d, LNSMaxPool2dFunction.forward, "default", default=True)
def max_pool2d(x, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False, return_indices=False):

    result = LNSMaxPool2dFunction.apply(x, kernel_size, x.base, stride, padding, dilation, ceil_mode, return_indices)

    if return_indices:
        return lnstensor(result[0], from_lns=True, b=x.base), result[1]
    else:
        return lnstensor(result, from_lns=True, b=x.base)

class LNSMaxPool3dFunction(LNSFunction):

    @staticmethod
    def forward(x, kernel_size, base, stride=None, padding=0, dilation=1, ceil_mode=False, return_indices=False):
        if stride is None:
            stride = kernel_size

        if isinstance(kernel_size, int): kernel_size = (kernel_size, kernel_size, kernel_size)
        if isinstance(stride, int): stride = (stride, stride, stride)
        if isinstance(padding, int): padding = (padding, padding, padding)
        if isinstance(dilation, int): dilation = (dilation, dilation, dilation)

        kernel_d, kernel_h, kernel_w = kernel_size
        stride_d, stride_h, stride_w = stride
        pad_d, pad_h, pad_w = padding
        dil_d, dil_h, dil_w = dilation

        if x.dim() == 4:
            x = x.unsqueeze(0)
            squeeze_batch = True
        else:
            squeeze_batch = False

        x_packed = x.to(torch.int64)
        N, C, D_in, H_in, W_in = x_packed.shape

        if pad_d > 0 or pad_h > 0 or pad_w > 0:
            x_min = lns_min(x_packed, base)
            fill = lns_sub(x_min, LNS_ONE, base).item()
            x_padded = torch.nn.functional.pad(
                x_packed,
                (pad_w, pad_w, pad_h, pad_h, pad_d, pad_d),
                value=fill
            )
        else:
            x_padded = x_packed

        eff_kd = (kernel_d - 1) * dil_d + 1
        eff_kh = (kernel_h - 1) * dil_h + 1
        eff_kw = (kernel_w - 1) * dil_w + 1

        if ceil_mode:
            D_out = int(math.ceil((D_in + 2 * pad_d - eff_kd) / stride_d)) + 1
            H_out = int(math.ceil((H_in + 2 * pad_h - eff_kh) / stride_h)) + 1
            W_out = int(math.ceil((W_in + 2 * pad_w - eff_kw) / stride_w)) + 1
        else:
            D_out = (D_in + 2 * pad_d - eff_kd) // stride_d + 1
            H_out = (H_in + 2 * pad_h - eff_kh) // stride_h + 1
            W_out = (W_in + 2 * pad_w - eff_kw) // stride_w + 1

        out_vals = zeros(N, C, D_out, H_out, W_out, device=x.device, b=base)._lns
        if return_indices:
            out_idx = torch.empty((N, C, D_out, H_out, W_out), dtype=torch.int64, device=x.device)

        padded_D = x_padded.size(-3)
        padded_H = x_padded.size(-2)
        padded_W = x_padded.size(-1)

        for n in range(N):
            for c in range(C):
                for d_out in range(D_out):
                    d_start = d_out * stride_d
                    d_end = d_start + eff_kd

                    if d_start >= padded_D:
                        continue

                    for h_out in range(H_out):
                        h_start = h_out * stride_h
                        h_end = h_start + eff_kh

                        if h_start >= padded_H:
                            continue

                        for w_out in range(W_out):
                            w_start = w_out * stride_w
                            w_end = w_start + eff_kw

                            if w_start >= padded_W:
                                continue

                            d_end_eff = min(d_end, padded_D)
                            h_end_eff = min(h_end, padded_H)
                            w_end_eff = min(w_end, padded_W)

                            window = x_padded[n, c,
                                              d_start:d_end_eff:dil_d,
                                              h_start:h_end_eff:dil_h,
                                              w_start:w_end_eff:dil_w]
                            window_flat = window.reshape(-1)

                            mx_val, mx_idx = lns_max(window_flat, base, dim=0)
                            out_vals[n, c, d_out, h_out, w_out] = mx_val

                            len_h = (h_end_eff - h_start + dil_h - 1) // dil_h
                            len_w = (w_end_eff - w_start + dil_w - 1) // dil_w

                            if return_indices:
                                mx_idx_int = int(mx_idx)
                                dhw = len_h * len_w
                                i_d = mx_idx_int // dhw
                                rem = mx_idx_int % dhw
                                i_h = rem // len_w
                                i_w = rem % len_w

                                d_in_idx = (d_start + i_d * dil_d) - pad_d
                                h_in_idx = (h_start + i_h * dil_h) - pad_h
                                w_in_idx = (w_start + i_w * dil_w) - pad_w
                                out_idx[n, c, d_out, h_out, w_out] = (d_in_idx * H_in + h_in_idx) * W_in + w_in_idx

        if squeeze_batch:
            out_vals = out_vals.squeeze(0)
            if return_indices:
                out_idx = out_idx.squeeze(0)

        if return_indices:
            return out_vals.to(torch.float64), out_idx
        else:
            return out_vals.to(torch.float64)

    @staticmethod
    def setup_context(ctx, inputs, output):
        x, kernel_size, base, stride, padding, dilation, ceil_mode, return_indices = inputs

        if stride is None:
            stride = kernel_size

        if isinstance(kernel_size, int): kernel_size = (kernel_size, kernel_size, kernel_size)
        if isinstance(stride, int): stride = (stride, stride, stride)
        if isinstance(padding, int): padding = (padding, padding, padding)
        if isinstance(dilation, int): dilation = (dilation, dilation, dilation)

        ctx.save_for_backward(x, base)
        ctx.kernel_size = kernel_size
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.ceil_mode = ceil_mode
        ctx.return_indices = return_indices

    @staticmethod
    def backward(ctx, grad_output, grad_out_indices=None):
        x, base = ctx.saved_tensors

        kernel_d, kernel_h, kernel_w = ctx.kernel_size
        stride_d, stride_h, stride_w = ctx.stride
        pad_d, pad_h, pad_w = ctx.padding
        dil_d, dil_h, dil_w = ctx.dilation

        if x.dim() == 4:
            x = x.unsqueeze(0)
            grad_output = grad_output.unsqueeze(0)
            squeeze_batch = True
        else:
            squeeze_batch = False

        N, C, D_in, H_in, W_in = x.shape
        x_packed = x.to(torch.int64)

        if pad_d > 0 or pad_h > 0 or pad_w > 0:
            x_min = lns_min(x, base)
            fill = lns_sub(x_min, LNS_ONE, base).item()
            x_padded = torch.nn.functional.pad(
                x_packed,
                (pad_w, pad_w, pad_h, pad_h, pad_d, pad_d),
                value=fill
            )
        else:
            x_padded = x_packed

        padded_D = x_padded.shape[-3]
        padded_H = x_padded.shape[-2]
        padded_W = x_padded.shape[-1]

        eff_kd = (kernel_d - 1) * dil_d + 1
        eff_kh = (kernel_h - 1) * dil_h + 1
        eff_kw = (kernel_w - 1) * dil_w + 1

        if ctx.ceil_mode:
            D_out = int(math.ceil((D_in + 2 * pad_d - eff_kd) / stride_d)) + 1
            H_out = int(math.ceil((H_in + 2 * pad_h - eff_kh) / stride_h)) + 1
            W_out = int(math.ceil((W_in + 2 * pad_w - eff_kw) / stride_w)) + 1
        else:
            D_out = (D_in + 2 * pad_d - eff_kd) // stride_d + 1
            H_out = (H_in + 2 * pad_h - eff_kh) // stride_h + 1
            W_out = (W_in + 2 * pad_w - eff_kw) // stride_w + 1

        grad_padded = zeros_like(x_padded, b=base)._lns

        for n in range(N):
            for c in range(C):
                for d_out in range(D_out):
                    d_start = d_out * stride_d
                    d_end = d_start + eff_kd

                    if d_start >= padded_D:
                        continue

                    for h_out in range(H_out):
                        h_start = h_out * stride_h
                        h_end = h_start + eff_kh

                        if h_start >= padded_H:
                            continue

                        for w_out in range(W_out):
                            w_start = w_out * stride_w
                            w_end = w_start + eff_kw

                            if w_start >= padded_W:
                                continue

                            d_end_eff = min(d_end, padded_D)
                            h_end_eff = min(h_end, padded_H)
                            w_end_eff = min(w_end, padded_W)

                            len_d = (d_end_eff - d_start + dil_d - 1) // dil_d
                            len_h = (h_end_eff - h_start + dil_h - 1) // dil_h
                            len_w = (w_end_eff - w_start + dil_w - 1) // dil_w
                            if len_d <= 0 or len_h <= 0 or len_w <= 0:
                                continue

                            window = x_padded[n, c,
                                              d_start:d_end_eff:dil_d,
                                              h_start:h_end_eff:dil_h,
                                              w_start:w_end_eff:dil_w]
                            window_flat = window.reshape(-1)

                            _, mx_idx = lns_max(window_flat, base, dim=0)

                            mx_idx = int(mx_idx)
                            dhw = len_h * len_w
                            i_d = mx_idx // dhw
                            rem = mx_idx % dhw
                            i_h = rem // len_w
                            i_w = rem % len_w

                            d_in_idx = d_start + i_d * dil_d
                            h_in_idx = h_start + i_h * dil_h
                            w_in_idx = w_start + i_w * dil_w

                            grad_padded[n, c, d_in_idx, h_in_idx, w_in_idx] = lns_add(
                                grad_padded[n, c, d_in_idx, h_in_idx, w_in_idx],
                                grad_output[n, c, d_out, h_out, w_out],
                                base
                            )

        if pad_d > 0 or pad_h > 0 or pad_w > 0:
            grad_padded = grad_padded[:, :,
                                      pad_d:pad_d + D_in,
                                      pad_h:pad_h + H_in,
                                      pad_w:pad_w + W_in]

        grad_x = grad_padded.squeeze(0) if squeeze_batch else grad_padded
        return grad_x, None, None, None, None, None, None, None

@implements(torch.nn.functional.max_pool3d_with_indices, LNSMaxPool3dFunction.forward, "default", default=True)
@implements(torch.nn.functional.max_pool3d, LNSMaxPool3dFunction.forward, "default", default=True)
def max_pool3d(x, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False, return_indices=False):

    result = LNSMaxPool3dFunction.apply(x, kernel_size, x.base, stride, padding, dilation, ceil_mode, return_indices)

    if return_indices:
        return lnstensor(result[0], from_lns=True, b=x.base), result[1]
    else:
        return lnstensor(result, from_lns=True, b=x.base)
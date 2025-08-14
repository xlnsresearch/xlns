import torch
from xlnstorch import LNS_ZERO, LNS_ONE, LNSTensor, lnstensor, format_lnstensor_operands, implements, zeros, ones
from xlnstorch.autograd import LNSFunction
from . import (
    lns_sub,
    lns_abs,
    lns_add,
    lns_mul,
    lns_le,
    lns_isclose,
    lns_gt,
    lns_lt,
    lns_eq,
    lns_div,
    lns_sum_to_size,
)

def _lns_equal(x, y):
    return torch.equal(x, y)

@implements(torch.equal, _lns_equal, "default", default=True)
def equal(x, y):
    x, y = format_lnstensor_operands(x, y)
    return _lns_equal(x._lns, y._lns)

def _lns_eq(x, y):
    return torch.eq(x, y)

@implements(torch.eq, _lns_eq, "default", default=True)
def eq(x, y, *, out=None):
    x, y = format_lnstensor_operands(x, y)
    y = y.broadcast_to(x.shape)
    result = _lns_eq(x._lns, y._lns)

    if out is not None:
        out.copy_(result)

    return result

def _lns_ne(x, y):
    return torch.ne(x, y)

@implements(torch.ne, _lns_ne, "default", default=True)
def ne(x, y, *, out=None):
    x, y = format_lnstensor_operands(x, y)
    y = y.broadcast_to(x.shape)
    result = _lns_ne(x._lns, y._lns)

    if out is not None:
        out.copy_(result)

    return result

def _lns_ge(x, y):
    x_packed, y_packed = x.to(torch.int64), y.to(torch.int64)
    x_packed_log, y_packed_log = x_packed >> 1, y_packed >> 1
    x_packed_sign, y_packed_sign = x_packed & 1, y_packed & 1

    both_pos = (x_packed_sign == 0) & (y_packed_sign == 0)
    result_both_pos = torch.ge(x_packed_log, y_packed_log)

    x_pos_y_neg = (x_packed_sign == 0) & (y_packed_sign == 1)
    result_x_pos = torch.ones_like(x_packed_sign, dtype=torch.bool)

    x_neg_y_pos = (x_packed_sign == 1) & (y_packed_sign == 0)
    result_x_neg = torch.zeros_like(x_packed_sign, dtype=torch.bool)

    # no need to check explicitly for both negative case, as it's the final case
    result_both_neg = torch.ge(y_packed_log, x_packed_log)

    return torch.where(both_pos, result_both_pos,
           torch.where(x_pos_y_neg, result_x_pos,
           torch.where(x_neg_y_pos, result_x_neg, result_both_neg)))

@implements(torch.ge, _lns_ge, "default", default=True)
def ge(x, y, *, out=None):
    x, y = format_lnstensor_operands(x, y)
    y = y.broadcast_to(x.shape)
    result = _lns_ge(x._lns, y._lns)

    if out is not None:
        out.copy_(result)

    return result

def _lns_gt(x, y):
    x_packed, y_packed = x.to(torch.int64), y.to(torch.int64)
    x_packed_log, y_packed_log = x_packed >> 1, y_packed >> 1
    x_packed_sign, y_packed_sign = x_packed & 1, y_packed & 1

    both_pos = (x_packed_sign == 0) & (y_packed_sign == 0)
    result_both_pos = torch.gt(x_packed_log, y_packed_log)

    x_pos_y_neg = (x_packed_sign == 0) & (y_packed_sign == 1)
    result_x_pos = torch.ones_like(x_packed_sign, dtype=torch.bool)

    x_neg_y_pos = (x_packed_sign == 1) & (y_packed_sign == 0)
    result_x_neg = torch.zeros_like(x_packed_sign, dtype=torch.bool)

    # no need to check explicitly for both negative case, as it's the final case
    result_both_neg = torch.gt(y_packed_log, x_packed_log)

    return torch.where(both_pos, result_both_pos,
           torch.where(x_pos_y_neg, result_x_pos,
           torch.where(x_neg_y_pos, result_x_neg, result_both_neg)))

@implements(torch.gt, _lns_gt, "default", default=True)
def gt(x, y, *, out=None):
    x, y = format_lnstensor_operands(x, y)
    y = y.broadcast_to(x.shape)
    result = _lns_gt(x._lns, y._lns)

    if out is not None:
        out.copy_(result)

    return result

def _lns_le(x, y):

    x_packed, y_packed = x.to(torch.int64), y.to(torch.int64)
    x_packed_log, y_packed_log = x_packed >> 1, y_packed >> 1
    x_packed_sign, y_packed_sign = x_packed & 1, y_packed & 1

    both_pos = (x_packed_sign == 0) & (y_packed_sign == 0)
    result_both_pos = torch.le(x_packed_log, y_packed_log)

    x_pos_y_neg = (x_packed_sign == 0) & (y_packed_sign == 1)
    result_x_pos = torch.zeros_like(x_packed_sign, dtype=torch.bool)

    x_neg_y_pos = (x_packed_sign == 1) & (y_packed_sign == 0)
    result_x_neg = torch.ones_like(x_packed_sign, dtype=torch.bool)

    # no need to check explicitly for both negative case, as it's the final case
    result_both_neg = torch.le(y_packed_log, x_packed_log)

    return torch.where(both_pos, result_both_pos,
           torch.where(x_pos_y_neg, result_x_pos,
           torch.where(x_neg_y_pos, result_x_neg, result_both_neg)))

@implements(torch.le, _lns_le, "default", default=True)
def le(x, y, *, out=None):
    x, y = format_lnstensor_operands(x, y)
    y = y.broadcast_to(x.shape)
    result = _lns_le(x._lns, y._lns)

    if out is not None:
        out.copy_(result)

    return result

def _lns_lt(x, y):
    x_packed, y_packed = x.to(torch.int64), y.to(torch.int64)
    x_packed_log, y_packed_log = x_packed >> 1, y_packed >> 1
    x_packed_sign, y_packed_sign = x_packed & 1, y_packed & 1

    both_pos = (x_packed_sign == 0) & (y_packed_sign == 0)
    result_both_pos = torch.lt(x_packed_log, y_packed_log)

    x_pos_y_neg = (x_packed_sign == 0) & (y_packed_sign == 1)
    result_x_pos = torch.zeros_like(x_packed_sign, dtype=torch.bool)

    x_neg_y_pos = (x_packed_sign == 1) & (y_packed_sign == 0)
    result_x_neg = torch.ones_like(x_packed_sign, dtype=torch.bool)

    # no need to check explicitly for both negative case, as it's the final case
    result_both_neg = torch.lt(y_packed_log, x_packed_log)

    return torch.where(both_pos, result_both_pos,
           torch.where(x_pos_y_neg, result_x_pos,
           torch.where(x_neg_y_pos, result_x_neg, result_both_neg)))

@implements(torch.lt, _lns_lt, "default", default=True)
def lt(x, y, *, out=None):
    x, y = format_lnstensor_operands(x, y)
    y = y.broadcast_to(x.shape)
    result = _lns_lt(x._lns, y._lns)

    if out is not None:
        out.copy_(result)

    return result

def _lns_isclose(x, y, base, rtol, atol):
    abs_diff = lns_abs(lns_sub(x, y, base))
    eps = lns_add(atol, lns_mul(rtol, lns_abs(y), base), base)
    return lns_le(abs_diff, eps)

@implements(torch.isclose, _lns_isclose, "default", default=True)
def isclose(x, y, rtol=1e-05, atol=1e-08, equal_nan=False): # equal_nan is not supported for now
    x, y, rtol, atol = format_lnstensor_operands(x, y, rtol, atol)
    return _lns_isclose(x._lns, y._lns, x.base, rtol._lns, atol._lns)

def _lns_allclose(x, y, base, rtol, atol):
    return torch.all(lns_isclose(x, y, base, rtol, atol))

@implements(torch.allclose, _lns_allclose, "default", default=True)
def allclose(x, y, rtol=1e-05, atol=1e-08, equal_nan=False): # equal_nan is not supported for now
    x, y, rtol, atol = format_lnstensor_operands(x, y, rtol, atol)
    return _lns_allclose(x._lns, y._lns, x.base, rtol._lns, atol._lns)

def _lns_any(x, dim=None, keepdim=False):
    x_packed = x.to(torch.int64)
    return torch.any(torch.ne(x_packed | 1, LNS_ZERO), dim=dim, keepdim=keepdim)

@implements(torch.any, _lns_any, "default", default=True)
def any(x, dim=None, keepdim=False, *, out=None):
    result = _lns_any(x._lns, dim, keepdim)

    if out is not None:
        out.copy_(result)

    return result

def _lns_all(x, dim=None, keepdim=False):
    x_packed = x.to(torch.int64)
    return torch.all(torch.ne(x_packed | 1, LNS_ZERO), dim=dim, keepdim=keepdim)

@implements(torch.all, _lns_all, "default", default=True)
def all(x, dim=None, keepdim=False, *, out=None):
    result = _lns_all(x._lns, dim, keepdim)

    if out is not None:
        out.copy_(result)

    return result

def _lns_isin(x, y, assume_unique=False, invert=False):
    return torch.isin(x, y, assume_unique=assume_unique, invert=invert)

@implements(torch.isin, _lns_isin, "default", default=True)
def isin(x, y, *, assume_unique=False, invert=False):
    x, y = format_lnstensor_operands(x, y)
    result = torch.isin(x._lns, y._lns, assume_unique=assume_unique, invert=invert)

    return result

class LNSSortFunction(LNSFunction):

    @staticmethod
    def forward(x, dim=-1, descending=False, stable=False):
        x_packed = x.to(torch.int64)
        x_packed_log = x_packed >> 1
        x_packed_sign = x_packed & 1

        offset = 2 * (torch.max(torch.abs(x_packed_log)) + 1)
        x_packed_logsign = torch.where(x_packed_sign == 1, -offset-x_packed_log, x_packed_log)
        indices = torch.argsort(x_packed_logsign, dim=dim, descending=descending, stable=stable)

        return torch.return_types.sort((torch.gather(x, dim, indices), indices))

    @staticmethod
    def setup_context(ctx, inputs, output):
        _, indices = output
        ctx.save_for_backward(indices)

    @staticmethod
    def backward(ctx, grad_output):
        indices, = ctx.saved_tensors

        grad_x = grad_output.clone()
        grad_x[indices] = grad_output

        return grad_x, None

@implements(torch.sort, LNSSortFunction.forward, "default", default=True)
def sort(x, dim=-1, descending=False, stable=False, *, out=None):
    result = LNSSortFunction.apply(x, dim, descending, stable)

    if out is not None:
        return out._inplace_copy(result)

    return torch.return_types.sort((lnstensor(result[0], from_lns=True, b=x.base), result[1]))

def _lns_argsort(x, dim=-1, descending=False, stable=False):
    x_packed = x.to(torch.int64)
    x_packed_log = x_packed >> 1
    x_packed_sign = x_packed & 1

    offset = 2 * (torch.max(torch.abs(x_packed_log)) + 1)
    x_packed_logsign = torch.where(x_packed_sign == 1, -offset-x_packed_log, x_packed_log)
    return torch.argsort(x_packed_logsign, dim=dim, descending=descending, stable=stable)

@implements(torch.argsort, _lns_argsort, "default", default=True)
def argsort(x, dim=-1, descending=False, stable=False, *, out=None):
    result = _lns_argsort(x._lns, dim, descending, stable)

    if out is not None:
        out.copy_(result)

    return result

class LNSKthvalueFunction(LNSFunction):

    @staticmethod
    def forward(x, k, dim=-1, keepdim=False):
        x_packed = x.to(torch.int64)
        x_packed_log = x_packed >> 1
        x_packed_sign = x_packed & 1

        offset = 2 * (torch.max(torch.abs(x_packed_log)) + 1)
        x_packed_logsign = torch.where(x_packed_sign == 1, -offset-x_packed_log, x_packed_log)
        _, indices = torch.kthvalue(x_packed_logsign, k, dim=dim, keepdim=keepdim)

        if not keepdim:
            indices = indices.unsqueeze(dim)
        x = torch.take_along_dim(x, indices, dim)

        if not keepdim:
            x = x.squeeze(dim)
            indices = indices.squeeze(dim)

        return torch.return_types.kthvalue((x, indices))

    @staticmethod
    def setup_context(ctx, inputs, output):
        x, _, dim, keepdim = inputs
        _, indices = output
        ctx.save_for_backward(x, indices)
        ctx.dim = dim
        ctx.keepdim = keepdim

    @staticmethod
    def backward(ctx, grad_output_values, grad_output_indices):
        x, indices = ctx.saved_tensors

        if not ctx.keepdim:
            indices = indices.unsqueeze(ctx.dim)
            grad_output_values = grad_output_values.unsqueeze(ctx.dim)

        grad_x = torch.full_like(x, LNS_ZERO)
        grad_x = grad_x.scatter_(ctx.dim, indices, grad_output_values)

        return grad_x, None, None, None

@implements(torch.kthvalue, LNSKthvalueFunction.forward, "default", default=True)
def kthvalue(x, k, dim=-1, keepdim=False, *, out=None):
    result = LNSKthvalueFunction.apply(x, k, dim, keepdim)

    if out is not None:
        return out._inplace_copy(result[0])

    return torch.return_types.sort((lnstensor(result[0], from_lns=True, b=x.base), result[1]))

class LNSMaximumFunction(LNSFunction):

    @staticmethod
    def forward(x, y, base):
        x_packed, y_packed = x.to(torch.int64), y.to(torch.int64)
        x_packed_larger = lns_gt(x_packed, y_packed)

        return torch.where(x_packed_larger, x, y)

    @staticmethod
    def setup_context(ctx, inputs, output):
        x, y, base = inputs
        ctx.save_for_backward(x, y, base)

    @staticmethod
    def backward(ctx, grad_output):
        x, y, base = ctx.saved_tensors
        x_packed, y_packed = x.to(torch.int64), y.to(torch.int64)

        x_y_equal = lns_eq(x_packed, y_packed)
        half_grad_output = lns_mul(grad_output, LNSTensor.get_internal_tensor(0.5, base), base)

        grad_x = torch.where(x_y_equal, half_grad_output, torch.where(
            lns_gt(x_packed, y_packed), grad_output, LNS_ZERO
        ))
        grad_y = torch.where(x_y_equal, half_grad_output, torch.where(
            lns_gt(y_packed, x_packed), grad_output, LNS_ZERO
        ))

        grad_x = lns_sum_to_size(grad_x, base, x.shape)
        grad_y = lns_sum_to_size(grad_y, base, y.shape)

        return grad_x, grad_y, None

@implements(torch.maximum, LNSMaximumFunction.forward, "default", default=True)
def maximum(x, y, *, out=None):
    x, y = format_lnstensor_operands(x, y)
    result = LNSMaximumFunction.apply(x, y, x.base)

    if out is not None:
        return out._inplace_copy(result)

    return lnstensor(result, from_lns=True, b=x.base)

class LNSMinimumFunction(LNSFunction):

    @staticmethod
    def forward(x, y, base):
        x_packed, y_packed = x.to(torch.int64), y.to(torch.int64)
        x_packed_smaller = lns_lt(x_packed, y_packed)

        return torch.where(x_packed_smaller, x, y)

    @staticmethod
    def setup_context(ctx, inputs, output):
        x, y, base = inputs
        ctx.save_for_backward(x, y, base)

    @staticmethod
    def backward(ctx, grad_output):
        x, y, base = ctx.saved_tensors
        x_packed, y_packed = x.to(torch.int64), y.to(torch.int64)

        x_y_equal = lns_eq(x_packed, y_packed)
        half_grad_output = lns_mul(grad_output, LNSTensor.get_internal_tensor(0.5, base), base)

        grad_x = torch.where(x_y_equal, half_grad_output, torch.where(
            lns_lt(x_packed, y_packed), grad_output, LNS_ZERO
        ))
        grad_y = torch.where(x_y_equal, half_grad_output, torch.where(
            lns_lt(y_packed, x_packed), grad_output, LNS_ZERO
        ))

        grad_x = lns_sum_to_size(grad_x, base, x.shape)
        grad_y = lns_sum_to_size(grad_y, base, y.shape)

        return grad_x, grad_y, None

@implements(torch.minimum, LNSMinimumFunction.forward, "default", default=True)
def minimum(x, y, *, out=None):
    x, y = format_lnstensor_operands(x, y)
    result = LNSMinimumFunction.apply(x, y, x.base)

    if out is not None:
        return out._inplace_copy(result)

    return lnstensor(result, from_lns=True, b=x.base)

class LNSMaxFunction(LNSFunction):

    @staticmethod
    def forward(x, base, dim=None, keepdim=False):
        x_packed = x.to(torch.int64)
        x_packed_log = x_packed >> 1
        x_packed_sign = x_packed & 1

        offset = 2 * (torch.max(torch.abs(x_packed_log)) + 1)
        x_packed_logsign = torch.where(x_packed_sign == 1, -offset-x_packed_log, x_packed_log)

        if dim is None:
            flat_indices = torch.argmax(x_packed_logsign)
            result = x.flatten()[flat_indices]

            return result

        indices_kept = torch.argmax(x_packed_logsign, dim=dim, keepdim=True)
        result = torch.gather(x, dim, indices_kept)

        if not keepdim:
            indices = indices_kept.squeeze(dim)
            result = result.squeeze(dim)
        else:
            indices = indices_kept

        return torch.return_types.max((result, indices))

    @staticmethod
    def setup_context(ctx, inputs, output):
        x, base, dim, keepdim = inputs
        if dim is None:
            ctx.save_for_backward(x, output, base)
        else:
            _, indices = output
            ctx.save_for_backward(x, indices)
        ctx.dim = dim
        ctx.keepdim = keepdim

    @staticmethod
    def backward(ctx, grad_output, grad_indicies=None): # grad_indices is not used
        if ctx.dim is None:
            x, result, base = ctx.saved_tensors

            max_values = torch.eq(x, result)
            grad = lns_div(LNS_ONE, LNSTensor.get_internal_tensor(max_values.sum(), base), base)

            return torch.where(max_values, grad, LNS_ZERO), None, None, None

        x, indices = ctx.saved_tensors

        grad_x = zeros(x.shape)._lns
        if ctx.keepdim:
            idx_expanded  = indices
            grad_expanded = grad_output
        else:
            idx_expanded  = indices.unsqueeze(ctx.dim)
            grad_expanded = grad_output.unsqueeze(ctx.dim)

        grad_x.scatter_(ctx.dim,
                        idx_expanded.expand(x.shape),
                        grad_expanded.expand(x.shape))

        return grad_x, None, None, None

@implements(torch.max, LNSMaxFunction.forward, "default", default=True)
def max(x, dim=None, keepdim=False, *, out=None):

    result = LNSMaxFunction.apply(x, x.base, dim, keepdim)

    if out is not None:

        if dim is None:
            return out._inplace_copy(result)

        out[0]._inplace_copy(result[0])
        out[1].copy_(result[1])
        return out

    if dim is None:
        return lnstensor(result, from_lns=True, b=x.base)

    return torch.return_types.sort((lnstensor(result[0], from_lns=True, b=x.base), result[1]))

def _lns_argmax(x, dim=None, keepdim=False):
    x_packed = x.to(torch.int64)
    x_packed_log = x_packed >> 1
    x_packed_sign = x_packed & 1

    offset = 2 * (torch.max(torch.abs(x_packed_log)) + 1)
    x_packed_logsign = torch.where(x_packed_sign == 1, -offset-x_packed_log, x_packed_log)
    return torch.argmax(x_packed_logsign, dim=dim, keepdim=keepdim)

@implements(torch.argmax, _lns_argmax, "default", default=True)
def argmax(x, dim=None, keepdim=False, *, out=None):
    result = _lns_argmax(x._lns, dim, keepdim)

    if out is not None:
        out.copy_(result)

    return result

class LNSMinFunction(LNSFunction):

    @staticmethod
    def forward(x, base, dim=None, keepdim=False):
        x_packed = x.to(torch.int64)
        x_packed_log = x_packed >> 1
        x_packed_sign = x_packed & 1

        offset = 2 * (torch.max(torch.abs(x_packed_log)) + 1)
        x_packed_logsign = torch.where(x_packed_sign == 1, -offset-x_packed_log, x_packed_log)

        if dim is None:
            flat_indices = torch.argmin(x_packed_logsign)
            result = x.flatten()[flat_indices]

            return result

        indices_kept = torch.argmin(x_packed_logsign, dim=dim, keepdim=True)
        result = torch.gather(x, dim, indices_kept)

        if not keepdim:
            indices = indices_kept.squeeze(dim)
            result = result.squeeze(dim)
        else:
            indices = indices_kept

        return torch.return_types.max((result, indices))

    @staticmethod
    def setup_context(ctx, inputs, output):
        x, base, dim, keepdim = inputs
        if dim is None:
            ctx.save_for_backward(x, output, base)
        else:
            _, indices = output
            ctx.save_for_backward(x, indices)
        ctx.dim = dim
        ctx.keepdim = keepdim

    @staticmethod
    def backward(ctx, grad_output, grad_indicies=None): # grad_indices is not used
        if ctx.dim is None:
            x, result, base = ctx.saved_tensors

            min_values = torch.eq(x, result)
            grad = lns_div(LNS_ONE, LNSTensor.get_internal_tensor(min_values.sum(), base), base)

            return torch.where(min_values, grad, LNS_ZERO), None, None, None

        x, indices = ctx.saved_tensors

        grad_x = zeros(x.shape)._lns
        if ctx.keepdim:
            idx_expanded  = indices
            grad_expanded = grad_output
        else:
            idx_expanded  = indices.unsqueeze(ctx.dim)
            grad_expanded = grad_output.unsqueeze(ctx.dim)

        grad_x.scatter_(ctx.dim,
                        idx_expanded.expand(x.shape),
                        grad_expanded.expand(x.shape))

        return grad_x, None, None, None

@implements(torch.min, LNSMinFunction.forward, "default", default=True)
def min(x, dim=None, keepdim=False, *, out=None):

    result = LNSMinFunction.apply(x, x.base, dim, keepdim)

    if out is not None:

        if dim is None:
            return out._inplace_copy(result)

        out[0]._inplace_copy(result[0])
        out[1].copy_(result[1])
        return out

    if dim is None:
        return lnstensor(result, from_lns=True, b=x.base)

    return torch.return_types.sort((lnstensor(result[0], from_lns=True, b=x.base), result[1]))

def _lns_argmin(x, dim=None, keepdim=False):
    x_packed = x.to(torch.int64)
    x_packed_log = x_packed >> 1
    x_packed_sign = x_packed & 1

    offset = 2 * (torch.max(torch.abs(x_packed_log)) + 1)
    x_packed_logsign = torch.where(x_packed_sign == 1, -offset-x_packed_log, x_packed_log)
    return torch.argmin(x_packed_logsign, dim=dim, keepdim=keepdim)

@implements(torch.argmin, _lns_argmin, "default", default=True)
def argmin(x, dim=None, keepdim=False, *, out=None):
    result = _lns_argmin(x._lns, dim, keepdim)

    if out is not None:
        out.copy_(result)

    return result

class LNSClampFunction(LNSFunction):

    @staticmethod
    def forward(x, min=None, max=None):
        x_packed = x.to(torch.int64)
        result = x_packed.clone()

        if min is not None:
            min_packed = min.to(torch.int64)
            lt_mask = lns_lt(result, min_packed)
            result = torch.where(lt_mask, min_packed, result)

        if max is not None:
            max_packed = max.to(torch.int64)
            gt_mask = lns_gt(result, max_packed)
            result = torch.where(gt_mask, max_packed, result)

        return result.to(torch.float64)

    @staticmethod
    def setup_context(ctx, inputs, output):
        x, min, max = inputs
        ctx.save_for_backward(x, min, max)

    @staticmethod
    def backward(ctx, grad_output):
        x, min, max = ctx.saved_tensors
        x_packed = x.to(torch.int64)

        grad_x = grad_output.clone()

        if min is not None:
            min_packed = min.to(torch.int64)
            lt_mask = lns_lt(x_packed, min_packed)
            grad_x = torch.where(lt_mask, LNS_ZERO, grad_x)

        if max is not None:
            max_packed = max.to(torch.int64)
            gt_mask = lns_gt(x_packed, max_packed)
            grad_x = torch.where(gt_mask, LNS_ZERO, grad_x)

        return grad_x, None, None

@implements(torch.clamp, LNSClampFunction.forward, "default", default=True)
def clamp(x, min=None, max=None, *, out=None):

    if min is not None and max is not None:
        x, min, max = format_lnstensor_operands(x, min, max)
    elif min is not None:
        x, min = format_lnstensor_operands(x, min)
    elif max is not None:
        x, max = format_lnstensor_operands(x, max)

    result = LNSClampFunction.apply(x, min, max)

    if out is not None:
        return out._inplace_copy(result)

    return lnstensor(result, from_lns=True, b=x.base)
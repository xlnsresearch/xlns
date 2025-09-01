import torch
from xlnstorch import LNS_ZERO, LNS_ONE, LNSTensor, lnstensor, format_lnstensor_operands, implements, zeros, ones
from xlnstorch.autograd import LNSFunction
from xlnstorch.ops import LNSOps

def _lns_equal(ops, x, y):
    return torch.equal(x, y)

@implements(torch.equal, _lns_equal, "default", default=True)
def equal(x, y):
    x, y = format_lnstensor_operands(x, y)
    return _lns_equal(None, x._lns, y._lns)

def _lns_eq(ops, x, y):
    return torch.eq(x, y)

@implements(torch.eq, _lns_eq, "default", default=True)
def eq(x, y, *, out=None):
    x, y = format_lnstensor_operands(x, y)
    y = y.broadcast_to(x.shape)
    result = _lns_eq(None, x._lns, y._lns)

    if out is not None:
        out.copy_(result)

    return result

def _lns_ne(ops, x, y):
    return torch.ne(x, y)

@implements(torch.ne, _lns_ne, "default", default=True)
def ne(x, y, *, out=None):
    x, y = format_lnstensor_operands(x, y)
    y = y.broadcast_to(x.shape)
    result = _lns_ne(None, x._lns, y._lns)

    if out is not None:
        out.copy_(result)

    return result

def _lns_ge(ops, x, y):
    x_log, y_log = x >> 1, y >> 1
    x_sign, y_sign = x & 1, y & 1

    both_pos = (x_sign == 0) & (y_sign == 0)
    result_both_pos = torch.ge(x_log, y_log)

    x_pos_y_neg = (x_sign == 0) & (y_sign == 1)
    result_x_pos = torch.ones_like(x_sign, dtype=torch.bool)

    x_neg_y_pos = (x_sign == 1) & (y_sign == 0)
    result_x_neg = torch.zeros_like(x_sign, dtype=torch.bool)

    # no need to check explicitly for both negative case, as it's the final case
    result_both_neg = torch.ge(y_log, x_log)

    return torch.where(both_pos, result_both_pos,
           torch.where(x_pos_y_neg, result_x_pos,
           torch.where(x_neg_y_pos, result_x_neg, result_both_neg)))

@implements(torch.ge, _lns_ge, "default", default=True)
def ge(x, y, *, out=None):
    x, y = format_lnstensor_operands(x, y)

    x_lns, y_lns = x._lns.view(torch.int64), y._lns.view(torch.int64)
    result = _lns_ge(None, x_lns, y_lns)

    if out is not None:
        out.copy_(result)

    return result

def _lns_gt(ops, x, y):
    x_log, y_log = x >> 1, y >> 1
    x_sign, y_sign = x & 1, y & 1

    both_pos = (x_sign == 0) & (y_sign == 0)
    result_both_pos = torch.gt(x_log, y_log)

    x_pos_y_neg = (x_sign == 0) & (y_sign == 1)
    result_x_pos = torch.ones_like(x_sign, dtype=torch.bool)

    x_neg_y_pos = (x_sign == 1) & (y_sign == 0)
    result_x_neg = torch.zeros_like(x_sign, dtype=torch.bool)

    # no need to check explicitly for both negative case, as it's the final case
    result_both_neg = torch.gt(y_log, x_log)

    return torch.where(both_pos, result_both_pos,
           torch.where(x_pos_y_neg, result_x_pos,
           torch.where(x_neg_y_pos, result_x_neg, result_both_neg)))

@implements(torch.gt, _lns_gt, "default", default=True)
def gt(x, y, *, out=None):
    x, y = format_lnstensor_operands(x, y)

    x_lns, y_lns = x._lns.view(torch.int64), y._lns.view(torch.int64)
    result = _lns_gt(None, x_lns, y_lns)

    if out is not None:
        out.copy_(result)

    return result

def _lns_le(ops, x, y):

    x_log, y_log = x >> 1, y >> 1
    x_sign, y_sign = x & 1, y & 1

    both_pos = (x_sign == 0) & (y_sign == 0)
    result_both_pos = torch.le(x_log, y_log)

    x_pos_y_neg = (x_sign == 0) & (y_sign == 1)
    result_x_pos = torch.zeros_like(x_sign, dtype=torch.bool)

    x_neg_y_pos = (x_sign == 1) & (y_sign == 0)
    result_x_neg = torch.ones_like(x_sign, dtype=torch.bool)

    # no need to check explicitly for both negative case, as it's the final case
    result_both_neg = torch.le(y_log, x_log)

    return torch.where(both_pos, result_both_pos,
           torch.where(x_pos_y_neg, result_x_pos,
           torch.where(x_neg_y_pos, result_x_neg, result_both_neg)))

@implements(torch.le, _lns_le, "default", default=True)
def le(x, y, *, out=None):
    x, y = format_lnstensor_operands(x, y)

    x_lns, y_lns = x._lns.view(torch.int64), y._lns.view(torch.int64)
    result = _lns_le(None, x_lns, y_lns)

    if out is not None:
        out.copy_(result)

    return result

def _lns_lt(ops, x, y):

    x_log, y_log = x >> 1, y >> 1
    x_sign, y_sign = x & 1, y & 1

    both_pos = (x_sign == 0) & (y_sign == 0)
    result_both_pos = torch.lt(x_log, y_log)

    x_pos_y_neg = (x_sign == 0) & (y_sign == 1)
    result_x_pos = torch.zeros_like(x_sign, dtype=torch.bool)

    x_neg_y_pos = (x_sign == 1) & (y_sign == 0)
    result_x_neg = torch.ones_like(x_sign, dtype=torch.bool)

    # no need to check explicitly for both negative case, as it's the final case
    result_both_neg = torch.lt(y_log, x_log)

    return torch.where(both_pos, result_both_pos,
           torch.where(x_pos_y_neg, result_x_pos,
           torch.where(x_neg_y_pos, result_x_neg, result_both_neg)))

@implements(torch.lt, _lns_lt, "default", default=True)
def lt(x, y, *, out=None):
    x, y = format_lnstensor_operands(x, y)

    x_lns, y_lns = x._lns.view(torch.int64), y._lns.view(torch.int64)
    result = _lns_lt(None, x_lns, y_lns)

    if out is not None:
        out.copy_(result)

    return result

def _lns_isclose(ops, x, y, rtol, atol):
    abs_diff = ops.abs(ops.sub(x, y))
    eps = ops.add(atol, ops.mul(rtol, ops.abs(y)))
    return ops.le(abs_diff, eps)

@implements(torch.isclose, _lns_isclose, "default", default=True)
def isclose(x, y, rtol=1e-05, atol=1e-08, equal_nan=False): # equal_nan is not supported for now
    x, y, rtol, atol = format_lnstensor_operands(x, y, rtol, atol)

    x_lns, y_lns = x._lns.view(torch.int64), y._lns.view(torch.int64)
    rtol_lns, atol_lns = rtol._lns.view(torch.int64), atol._lns.view(torch.int64)

    return _lns_isclose(LNSOps(x.base), x_lns, y_lns, rtol_lns, atol_lns)

def _lns_allclose(ops, x, y, rtol, atol):
    return torch.all(ops.isclose(x, y, rtol, atol))

@implements(torch.allclose, _lns_allclose, "default", default=True)
def allclose(x, y, rtol=1e-05, atol=1e-08, equal_nan=False): # equal_nan is not supported for now
    x, y, rtol, atol = format_lnstensor_operands(x, y, rtol, atol)

    x_lns, y_lns = x._lns.view(torch.int64), y._lns.view(torch.int64)
    rtol_lns, atol_lns = rtol._lns.view(torch.int64), atol._lns.view(torch.int64)

    return _lns_allclose(LNSOps(x.base), x_lns, y_lns, rtol_lns, atol_lns)

def _lns_any(ops, x, dim=None, keepdim=False):
    return torch.any(torch.ne(x | 1, LNS_ZERO), dim=dim, keepdim=keepdim)

@implements(torch.any, _lns_any, "default", default=True)
def any(x, dim=None, keepdim=False, *, out=None):
    result = _lns_any(None, x._lns.view(torch.int64), dim, keepdim)

    if out is not None:
        out.copy_(result)

    return result

def _lns_all(ops, x, dim=None, keepdim=False):
    return torch.all(torch.ne(x | 1, LNS_ZERO), dim=dim, keepdim=keepdim)

@implements(torch.all, _lns_all, "default", default=True)
def all(x, dim=None, keepdim=False, *, out=None):
    result = _lns_all(None, x._lns.view(torch.int64), dim, keepdim)

    if out is not None:
        out.copy_(result)

    return result

def _lns_isin(ops, x, y, assume_unique=False, invert=False):
    return torch.isin(x, y, assume_unique=assume_unique, invert=invert)

@implements(torch.isin, _lns_isin, "default", default=True)
def isin(x, y, *, assume_unique=False, invert=False):
    x, y = format_lnstensor_operands(x, y)
    result = _lns_isin(None, x._lns.view(torch.int64), y._lns.view(torch.int64),
                        assume_unique=assume_unique, invert=invert)

    return result

def _sort(ops, x, dim=-1, descending=False, stable=False):
    x_log = x >> 1
    x_sign = x & 1

    offset = 2 * (torch.max(torch.abs(x_log)) + 1)
    x_logsign = torch.where(x_sign == 1, -offset-x_log, x_log)
    indices = torch.argsort(x_logsign, dim=dim, descending=descending, stable=stable)

    return torch.return_types.sort((torch.gather(x, dim, indices), indices))

class LNSSortFunction(LNSFunction):

    _lnstensor_outputs = [0]

    @staticmethod
    def forward(ops, x, dim=-1, descending=False, stable=False):
        x = x.view(torch.int64)
        result = _sort(ops, x, dim, descending, stable)
        return result[0].view(torch.float64), result[1]

    @staticmethod
    def setup_context(ctx, ops, inputs, output):
        _, indices = output
        ctx.save_for_backward(indices)

    @staticmethod
    def backward(ctx, ops, grad_output):
        indices, = ctx.saved_tensors

        grad_x = grad_output.clone()
        grad_x[indices] = grad_output

        return grad_x, None, None, None

@implements(torch.sort, _sort, "default", default=True)
def sort(x, dim=-1, descending=False, stable=False, *, out=None):
    result = LNSSortFunction.apply(x, dim, descending, stable)

    if out is not None:
        return out._inplace_copy(result)

    return torch.return_types.sort((lnstensor(result[0], from_lns=True, b=x.base), result[1]))

def _lns_argsort(ops, x, dim=-1, descending=False, stable=False):
    x_log = x >> 1
    x_sign = x & 1

    offset = 2 * (torch.max(torch.abs(x_log)) + 1)
    x_logsign = torch.where(x_sign == 1, -offset-x_log, x_log)
    return torch.argsort(x_logsign, dim=dim, descending=descending, stable=stable)

@implements(torch.argsort, _lns_argsort, "default", default=True)
def argsort(x, dim=-1, descending=False, stable=False, *, out=None):
    result = _lns_argsort(None, x._lns.view(torch.int64), dim, descending, stable)

    if out is not None:
        out.copy_(result)

    return result

def _kthvalue(ops, x, k, dim=-1, keepdim=False):
    x_log = x >> 1
    x_sign = x & 1

    offset = 2 * (torch.max(torch.abs(x_log)) + 1)
    x_logsign = torch.where(x_sign == 1, -offset-x_log, x_log)
    _, indices = torch.kthvalue(x_logsign, k, dim=dim, keepdim=keepdim)

    if not keepdim:
        indices = indices.unsqueeze(dim)
    x = torch.take_along_dim(x, indices, dim)

    if not keepdim:
        x = x.squeeze(dim)
        indices = indices.squeeze(dim)

    return torch.return_types.kthvalue((x, indices))

class LNSKthvalueFunction(LNSFunction):

    _lnstensor_outputs = [0]

    @staticmethod
    def forward(ops, x, k, dim=-1, keepdim=False):
        x = x.view(torch.int64)
        result = _kthvalue(ops, x, k, dim, keepdim)
        return result[0].view(torch.float64), result[1]

    @staticmethod
    def setup_context(ctx, ops, inputs, output):
        x, _, dim, keepdim = inputs
        _, indices = output
        ctx.save_for_backward(x, indices)
        ctx.dim = dim
        ctx.keepdim = keepdim

    @staticmethod
    def backward(ctx, ops, grad_output_values, grad_output_indices):
        x, indices = ctx.saved_tensors

        if not ctx.keepdim:
            indices = indices.unsqueeze(ctx.dim)
            grad_output_values = grad_output_values.unsqueeze(ctx.dim)

        grad_x = ops.zeros_like(x).view(torch.float64)
        grad_x = grad_x.scatter_(ctx.dim, indices, grad_output_values)

        return grad_x, None, None, None

@implements(torch.kthvalue, _kthvalue, "default", default=True)
def kthvalue(x, k, dim=-1, keepdim=False, *, out=None):
    result = LNSKthvalueFunction.apply(x, k, dim, keepdim)

    if out is not None:
        return out._inplace_copy(result[0])

    return torch.return_types.sort((lnstensor(result[0], from_lns=True, b=x.base), result[1]))

def _maximum(ops, x, y):
    x_greater = ops.gt(x, y)
    return torch.where(x_greater, x, y)

class LNSMaximumFunction(LNSFunction):

    @staticmethod
    def forward(ops, x, y):
        x, y = x.view(torch.int64), y.view(torch.int64)
        result = _maximum(ops, x, y)
        return result.view(torch.float64)

    @staticmethod
    def setup_context(ctx, ops, inputs, output):
        x, y = inputs
        ctx.save_for_backward(x, y)

    @staticmethod
    def backward(ctx, ops, grad_output):
        x, y = ctx.saved_tensors
        x, y, grad_output = x.view(torch.int64), y.view(torch.int64), grad_output.view(torch.int64)

        x_y_equal = ops.eq(x, y)
        half_grad_output = ops.mul(grad_output, ops.to_lns(0.5))

        grad_x = torch.where(x_y_equal, half_grad_output, torch.where(
            ops.gt(x, y), grad_output, LNS_ZERO
        ))
        grad_y = torch.where(x_y_equal, half_grad_output, torch.where(
            ops.gt(y, x), grad_output, LNS_ZERO
        ))

        grad_x = ops.sum_to_size(grad_x, x.shape)
        grad_y = ops.sum_to_size(grad_y, y.shape)

        return grad_x.view(torch.float64), grad_y.view(torch.float64)

@implements(torch.maximum, _maximum, "default", default=True)
def maximum(x, y, *, out=None):
    x, y = format_lnstensor_operands(x, y)
    result = LNSMaximumFunction.apply(x, y)

    if out is not None:
        return out._inplace_copy(result)

    return lnstensor(result, from_lns=True, b=x.base)

def _minimum(ops, x, y):
    x_smaller = ops.lt(x, y)
    return torch.where(x_smaller, x, y)

class LNSMinimumFunction(LNSFunction):

    @staticmethod
    def forward(ops, x, y):
        x, y = x.view(torch.int64), y.view(torch.int64)
        result = _minimum(ops, x, y)
        return result.view(torch.float64)

    @staticmethod
    def setup_context(ctx, ops, inputs, output):
        x, y = inputs
        ctx.save_for_backward(x, y)

    @staticmethod
    def backward(ctx, ops, grad_output):
        x, y = ctx.saved_tensors
        x, y, grad_output = x.view(torch.int64), y.view(torch.int64), grad_output.view(torch.int64)

        x_y_equal = ops.eq(x, y)
        half_grad_output = ops.mul(grad_output, ops.to_lns(0.5))

        grad_x = torch.where(x_y_equal, half_grad_output, torch.where(
            ops.lt(x, y), grad_output, LNS_ZERO
        ))
        grad_y = torch.where(x_y_equal, half_grad_output, torch.where(
            ops.lt(y, x), grad_output, LNS_ZERO
        ))

        grad_x = ops.sum_to_size(grad_x, x.shape)
        grad_y = ops.sum_to_size(grad_y, y.shape)

        return grad_x.view(torch.float64), grad_y.view(torch.float64)

@implements(torch.minimum, _minimum, "default", default=True)
def minimum(x, y, *, out=None):
    x, y = format_lnstensor_operands(x, y)
    result = LNSMinimumFunction.apply(x, y)

    if out is not None:
        return out._inplace_copy(result)

    return lnstensor(result, from_lns=True, b=x.base)

def _max(ops, x, dim=None, keepdim=False):
    x_log = x >> 1
    x_sign = x & 1

    offset = 2 * (torch.max(torch.abs(x_log)) + 1)
    x_logsign = torch.where(x_sign == 1, -offset - x_log, x_log)

    if dim is None:
        flat_indices = torch.argmax(x_logsign)
        result = x.flatten()[flat_indices]

        return result

    indices_kept = torch.argmax(x_logsign, dim=dim, keepdim=True)
    result = torch.gather(x, dim, indices_kept)

    if not keepdim:
        indices = indices_kept.squeeze(dim)
        result = result.squeeze(dim)
    else:
        indices = indices_kept

    return torch.return_types.max((result, indices))

class LNSMaxFunction(LNSFunction):

    _lnstensor_outputs = [0]

    @staticmethod
    def forward(ops, x, dim=None, keepdim=False):
        x = x.view(torch.int64)
        result = _sort(ops, x, dim, descending=True, stable=True)

        if len(result) == 1:
            return result.view(torch.float64)

        return result[0].view(torch.float64), result[1]

    @staticmethod
    def setup_context(ctx, ops, inputs, output):
        x, dim, keepdim = inputs
        if dim is None:
            ctx.save_for_backward(x, output)
        else:
            _, indices = output
            ctx.save_for_backward(x, indices)
        ctx.dim = dim
        ctx.keepdim = keepdim

    @staticmethod
    def backward(ctx, ops, grad_output, grad_indicies=None): # grad_indices is not used
        grad_output = grad_output.view(torch.int64)

        if ctx.dim is None:
            x, result = ctx.saved_tensors

            max_values = torch.eq(x, result)
            grad_x = ops.div(grad_output, ops.to_lns(max_values.sum()))

            return torch.where(max_values, grad_x, LNS_ZERO).view(torch.float64), None, None, None

        x, indices = ctx.saved_tensors

        grad_x = ops.zeros_like(x)
        if ctx.keepdim:
            idx_expanded  = indices
            grad_expanded = grad_output
        else:
            idx_expanded  = indices.unsqueeze(ctx.dim)
            grad_expanded = grad_output.unsqueeze(ctx.dim)

        grad_x.scatter_(ctx.dim,
                        idx_expanded.expand(x.shape),
                        grad_expanded.expand(x.shape))

        return grad_x.view(torch.float64), None, None

@implements(torch.max, _max, "default", default=True)
def max(x, dim=None, keepdim=False, *, out=None):

    result = LNSMaxFunction.apply(x, dim, keepdim)

    if out is not None:

        if dim is None:
            return out._inplace_copy(result)

        out[0]._inplace_copy(result[0])
        out[1].copy_(result[1])
        return out

    if dim is None:
        return lnstensor(result, from_lns=True, b=x.base)

    return torch.return_types.sort((lnstensor(result[0], from_lns=True, b=x.base), result[1]))

def _lns_argmax(ops, x, dim=None, keepdim=False):
    x_log = x >> 1
    x_sign = x & 1

    offset = 2 * (torch.max(torch.abs(x_log)) + 1)
    x_logsign = torch.where(x_sign == 1, -offset - x_log, x_log)
    return torch.argmax(x_logsign, dim=dim, keepdim=keepdim)

@implements(torch.argmax, _lns_argmax, "default", default=True)
def argmax(x, dim=None, keepdim=False, *, out=None):
    result = _lns_argmax(None, x._lns.view(torch.float64), dim, keepdim)

    if out is not None:
        out.copy_(result)

    return result

def _min(ops, x, dim=None, keepdim=False):
    x_log = x >> 1
    x_sign = x & 1

    offset = 2 * (torch.max(torch.abs(x_log)) + 1)
    x_logsign = torch.where(x_sign == 1, -offset - x_log, x_log)

    if dim is None:
        flat_indices = torch.argmin(x_logsign)
        result = x.flatten()[flat_indices]

        return result

    indices_kept = torch.argmin(x_logsign, dim=dim, keepdim=True)
    result = torch.gather(x, dim, indices_kept)

    if not keepdim:
        indices = indices_kept.squeeze(dim)
        result = result.squeeze(dim)
    else:
        indices = indices_kept

    return torch.return_types.max((result, indices))

class LNSMinFunction(LNSFunction):

    _lnstensor_outputs = [0]

    @staticmethod
    def forward(ops, x, dim=None, keepdim=False):
        x = x.view(torch.int64)
        result = _min(ops, x, dim, keepdim)

        if len(result) == 1:
            return result.view(torch.float64)

        return result[0].view(torch.float64), result[1]

    @staticmethod
    def setup_context(ctx, ops, inputs, output):
        x, dim, keepdim = inputs
        if dim is None:
            ctx.save_for_backward(x, output)
        else:
            _, indices = output
            ctx.save_for_backward(x, indices)
        ctx.dim = dim
        ctx.keepdim = keepdim

    @staticmethod
    def backward(ctx, ops, grad_output, grad_indicies=None): # grad_indices is not used
        grad_output = grad_output.view(torch.int64)

        if ctx.dim is None:
            x, result = ctx.saved_tensors

            min_values = torch.eq(x, result)
            grad_x = ops.div(grad_output, ops.to_lns(min_values.sum()))

            return torch.where(min_values, grad_x, LNS_ZERO).view(torch.float64), None, None, None

        x, indices = ctx.saved_tensors

        grad_x = ops.zeros_like(x)
        if ctx.keepdim:
            idx_expanded  = indices
            grad_expanded = grad_output
        else:
            idx_expanded  = indices.unsqueeze(ctx.dim)
            grad_expanded = grad_output.unsqueeze(ctx.dim)

        grad_x.scatter_(ctx.dim,
                        idx_expanded.expand(x.shape),
                        grad_expanded.expand(x.shape))

        return grad_x.view(torch.float64), None, None

@implements(torch.min, _min, "default", default=True)
def min(x, dim=None, keepdim=False, *, out=None):

    result = LNSMinFunction.apply(x, dim, keepdim)

    if out is not None:

        if dim is None:
            return out._inplace_copy(result)

        out[0]._inplace_copy(result[0])
        out[1].copy_(result[1])
        return out

    if dim is None:
        return lnstensor(result, from_lns=True, b=x.base)

    return torch.return_types.sort((lnstensor(result[0], from_lns=True, b=x.base), result[1]))

def _lns_argmin(ops, x, dim=None, keepdim=False):
    x_log = x >> 1
    x_sign = x & 1

    offset = 2 * (torch.max(torch.abs(x_log)) + 1)
    x_logsign = torch.where(x_sign == 1, -offset-x_log, x_log)
    return torch.argmin(x_logsign, dim=dim, keepdim=keepdim)

@implements(torch.argmin, _lns_argmin, "default", default=True)
def argmin(x, dim=None, keepdim=False, *, out=None):
    result = _lns_argmin(None, x._lns.view(torch.float64), dim, keepdim)

    if out is not None:
        out.copy_(result)

    return result

def _clamp(ops, x, min=None, max=None):
    result = x.clone()

    if min is not None:
        lt_mask = ops.lt(result, min)
        result = torch.where(lt_mask, min, result)

    if max is not None:
        gt_mask = ops.gt(result, max)
        result = torch.where(gt_mask, max, result)

    return result

class LNSClampFunction(LNSFunction):

    @staticmethod
    def forward(ops, x, min=None, max=None):
        x = x.view(torch.int64)
        min = min.view(torch.int64) if min is not None else None
        max = max.view(torch.int64) if max is not None else None

        result = _clamp(ops, x, min, max)
        return result.view(torch.float64)

    @staticmethod
    def setup_context(ctx, ops, inputs, output):
        x, min, max = inputs
        ctx.save_for_backward(x, min, max)

    @staticmethod
    def backward(ctx, ops, grad_output):
        x, min, max = ctx.saved_tensors
        x = x.view(torch.int64)
        min = min.view(torch.int64) if min is not None else None
        max = max.view(torch.int64) if max is not None else None

        grad_x = grad_output.view(torch.int64)

        if min is not None:
            lt_mask = ops.lt(x, min)
            grad_x = torch.where(lt_mask, LNS_ZERO, grad_x)

        if max is not None:
            gt_mask = ops.gt(x, max)
            grad_x = torch.where(gt_mask, LNS_ZERO, grad_x)

        return grad_x.view(torch.float64), None, None

@implements(torch.clamp, _clamp, "default", default=True)
def clamp(x, min=None, max=None, *, out=None):

    x, min, max = format_lnstensor_operands(x, min, max)
    result = LNSClampFunction.apply(x, min, max)

    if out is not None:
        return out._inplace_copy(result)

    return lnstensor(result, from_lns=True, b=x.base)
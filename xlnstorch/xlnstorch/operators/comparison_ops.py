import torch
from xlnstorch import LNS_ZERO, format_lnstensor_operands, implements
from xlnstorch.autograd import LNSFunction, LNSNonDifferentiableFunction

class LNSEqualFunction(LNSNonDifferentiableFunction):

    _lnstensor_outputs = []

    @staticmethod
    def forward(ops, x, y):
        return torch.equal(x, y)

@implements(torch.equal, LNSEqualFunction.forward, "default", default=True)
def equal(x, y):
    x, y = format_lnstensor_operands(x, y)
    return LNSEqualFunction.apply(x, y)

class LNSEqFunction(LNSNonDifferentiableFunction):

    _lnstensor_outputs = []

    @staticmethod
    def forward(ops, x, y):
        return torch.eq(x, y)

@implements(torch.eq, LNSEqFunction.forward, "default", default=True)
def eq(x, y, *, out=None):

    x, y = format_lnstensor_operands(x, y)
    result = LNSEqFunction.apply(x, y)

    if out is not None:
        out.copy_(result)

    return result

class LNSNeFunction(LNSNonDifferentiableFunction):

    _lnstensor_outputs = []

    @staticmethod
    def forward(ops, x, y):
        return torch.ne(x, y)

@implements(torch.ne, LNSNeFunction.forward, "default", default=True)
def ne(x, y, *, out=None):
    x, y = format_lnstensor_operands(x, y)
    result = LNSNeFunction.apply(x, y)

    if out is not None:
        out.copy_(result)

    return result

class LNSGeFunction(LNSNonDifferentiableFunction):

    _lnstensor_outputs = []

    @staticmethod
    def forward(ops, x, y):
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

@implements(torch.ge, LNSGeFunction.forward, "default", default=True)
def ge(x, y, *, out=None):
    x, y = format_lnstensor_operands(x, y)
    result = LNSGeFunction.apply(x, y)

    if out is not None:
        out.copy_(result)

    return result

class LNSGtFunction(LNSNonDifferentiableFunction):

    _lnstensor_outputs = []

    @staticmethod
    def forward(ops, x, y):
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

@implements(torch.gt, LNSGtFunction.forward, "default", default=True)
def gt(x, y, *, out=None):
    x, y = format_lnstensor_operands(x, y)
    result = LNSGtFunction.apply(x, y)

    if out is not None:
        out.copy_(result)

    return result

class LNSLeFunction(LNSNonDifferentiableFunction):

    _lnstensor_outputs = []

    @staticmethod
    def forward(ops, x, y):
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

@implements(torch.le, LNSLeFunction.forward, "default", default=True)
def le(x, y, *, out=None):
    x, y = format_lnstensor_operands(x, y)
    result = LNSLeFunction.apply(x, y)

    if out is not None:
        out.copy_(result)

    return result

class LNSLtFunction(LNSNonDifferentiableFunction):

    _lnstensor_outputs = []

    @staticmethod
    def forward(ops, x, y):
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

@implements(torch.lt, LNSLtFunction.forward, "default", default=True)
def lt(x, y, *, out=None):
    x, y = format_lnstensor_operands(x, y)
    result = LNSLtFunction.apply(x, y)

    if out is not None:
        out.copy_(result)

    return result

class LNSIscloseFunction(LNSNonDifferentiableFunction):

    _lnstensor_outputs = []

    @staticmethod
    def forward(ops, x, y, atol, rtol):
        abs_diff = ops.abs(ops.sub(x, y))
        eps = ops.add(atol, ops.mul(rtol, ops.abs(y)))
        return ops.le(abs_diff, eps)

@implements(torch.isclose, LNSIscloseFunction.forward, "default", default=True)
def isclose(x, y, rtol=1e-05, atol=1e-08, equal_nan=False): # equal_nan is not supported for now
    x, y, rtol, atol = format_lnstensor_operands(x, y, rtol, atol)
    return LNSIscloseFunction.apply(x, y, atol, rtol)

class LNSAllcloseFunction(LNSNonDifferentiableFunction):

    _lnstensor_outputs = []

    @staticmethod
    def forward(ops, x, y, atol, rtol):
        return torch.all(ops.isclose(x, y, rtol, atol))

@implements(torch.allclose, LNSAllcloseFunction.forward, "default", default=True)
def allclose(x, y, rtol=1e-05, atol=1e-08, equal_nan=False): # equal_nan is not supported for now
    x, y, rtol, atol = format_lnstensor_operands(x, y, rtol, atol)
    return LNSAllcloseFunction.apply(x, y, atol, rtol)

class LNSAnyFunction(LNSNonDifferentiableFunction):

    _lnstensor_outputs = []

    @staticmethod
    def forward(ops, x, dim=None, keepdim=False):
        return torch.any(torch.ne(x | 1, LNS_ZERO), dim=dim, keepdim=keepdim)

@implements(torch.any, LNSAnyFunction.forward, "default", default=True)
def any(x, dim=None, keepdim=False, *, out=None):
    result = LNSAnyFunction.apply(x, dim, keepdim)

    if out is not None:
        out.copy_(result)

    return result

class LNSAllFunction(LNSNonDifferentiableFunction):

    _lnstensor_outputs = []

    @staticmethod
    def forward(ops, x, dim=None, keepdim=False):
        return torch.all(torch.ne(x | 1, LNS_ZERO), dim=dim, keepdim=keepdim)

@implements(torch.all, LNSAllFunction.forward, "default", default=True)
def all(x, dim=None, keepdim=False, *, out=None):
    result = LNSAllFunction.apply(x, dim, keepdim)

    if out is not None:
        out.copy_(result)

    return result

class LNSIsinFunction(LNSNonDifferentiableFunction):

    _lnstensor_outputs = []

    @staticmethod
    def forward(ops, x, y, assume_unique=False, invert=False):
        return torch.isin(x, y, assume_unique=assume_unique, invert=invert)

@implements(torch.isin, LNSIsinFunction.forward, "default", default=True)
def isin(x, y, *, assume_unique=False, invert=False):
    x, y = format_lnstensor_operands(x, y)
    return LNSIsinFunction.apply(x, y, assume_unique, invert)

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
        return _sort(ops, x, dim, descending, stable)

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
        out[0]._inplace_copy(result[0])
        out[1].copy_(result[1])
        return out

    return result

class LNSArgsortFunction(LNSNonDifferentiableFunction):

    _lnstensor_outputs = []

    @staticmethod
    def forward(ops, x, dim=-1, descending=False, stable=False):
        x_log = x >> 1
        x_sign = x & 1

        offset = 2 * (torch.max(torch.abs(x_log)) + 1)
        x_logsign = torch.where(x_sign == 1, -offset-x_log, x_log)
        return torch.argsort(x_logsign, dim=dim, descending=descending, stable=stable)

@implements(torch.argsort, LNSArgsortFunction.forward, "default", default=True)
def argsort(x, dim=-1, descending=False, stable=False, *, out=None):
    result = LNSArgsortFunction.apply(x, dim, descending, stable)

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
        return _kthvalue(ops, x, k, dim, keepdim)

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

        grad_x = ops.zeros_like(x)
        grad_x = grad_x.scatter_(ctx.dim, indices, grad_output_values)

        return grad_x, None, None, None

@implements(torch.kthvalue, _kthvalue, "default", default=True)
def kthvalue(x, k, dim=-1, keepdim=False, *, out=None):
    result = LNSKthvalueFunction.apply(x, k, dim, keepdim)

    if out is not None:
        out[0]._inplace_copy(result[0])
        out[1].copy_(result[1])
        return out

    return result

def _maximum(ops, x, y):
    x_greater = ops.gt(x, y)
    return torch.where(x_greater, x, y)

class LNSMaximumFunction(LNSFunction):

    @staticmethod
    def forward(ops, x, y):
        return _maximum(ops, x, y)

    @staticmethod
    def setup_context(ctx, ops, inputs, output):
        x, y = inputs
        ctx.save_for_backward(x, y)

    @staticmethod
    def backward(ctx, ops, grad_output):
        x, y = ctx.saved_tensors

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

        return grad_x, grad_y

@implements(torch.maximum, _maximum, "default", default=True)
def maximum(x, y, *, out=None):
    x, y = format_lnstensor_operands(x, y)
    result = LNSMaximumFunction.apply(x, y)

    if out is not None:
        return out._inplace_copy(result)

    return result

def _minimum(ops, x, y):
    x_smaller = ops.lt(x, y)
    return torch.where(x_smaller, x, y)

class LNSMinimumFunction(LNSFunction):

    @staticmethod
    def forward(ops, x, y):
        return _minimum(ops, x, y)

    @staticmethod
    def setup_context(ctx, ops, inputs, output):
        x, y = inputs
        ctx.save_for_backward(x, y)

    @staticmethod
    def backward(ctx, ops, grad_output):
        x, y = ctx.saved_tensors

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

        return grad_x, grad_y

@implements(torch.minimum, _minimum, "default", default=True)
def minimum(x, y, *, out=None):
    x, y = format_lnstensor_operands(x, y)
    result = LNSMinimumFunction.apply(x, y)

    if out is not None:
        return out._inplace_copy(result)

    return result

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
        return _sort(ops, x, dim, descending=True, stable=True)

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

        if ctx.dim is None:
            x, result = ctx.saved_tensors

            max_values = torch.eq(x, result)
            grad_x = ops.div(grad_output, ops.to_lns(max_values.sum()))

            return torch.where(max_values, grad_x, LNS_ZERO), None, None, None

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

        return grad_x, None, None

@implements(torch.max, _max, "default", default=True)
def max(x, dim=None, keepdim=False, *, out=None):
    result = LNSMaxFunction.apply(x, dim, keepdim)

    if out is not None:
        if dim is None:
            return out._inplace_copy(result)

        out[0]._inplace_copy(result[0])
        out[1].copy_(result[1])
        return out

    return result

class LNSArgmaxFunction(LNSNonDifferentiableFunction):

    _lnstensor_outputs = []

    @staticmethod
    def forward(ops, x, dim=None, keepdim=False):
        x_log = x >> 1
        x_sign = x & 1

        offset = 2 * (torch.max(torch.abs(x_log)) + 1)
        x_logsign = torch.where(x_sign == 1, -offset - x_log, x_log)
        return torch.argmax(x_logsign, dim=dim, keepdim=keepdim)

@implements(torch.argmax, LNSArgmaxFunction.forward, "default", default=True)
def argmax(x, dim=None, keepdim=False, *, out=None):
    result = LNSArgmaxFunction.apply(x, dim, keepdim)

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
        return _min(ops, x, dim, keepdim)

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

        if ctx.dim is None:
            x, result = ctx.saved_tensors

            min_values = torch.eq(x, result)
            grad_x = ops.div(grad_output, ops.to_lns(min_values.sum()))

            return torch.where(min_values, grad_x, LNS_ZERO), None, None, None

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

        return grad_x, None, None

@implements(torch.min, _min, "default", default=True)
def min(x, dim=None, keepdim=False, *, out=None):
    result = LNSMinFunction.apply(x, dim, keepdim)

    if out is not None:
        if dim is None:
            return out._inplace_copy(result)

        out[0]._inplace_copy(result[0])
        out[1].copy_(result[1])
        return out

    return result

class LNSArgminFunction(LNSNonDifferentiableFunction):

    _lnstensor_outputs = []

    @staticmethod
    def forward(ops, x, dim=None, keepdim=False):
        x_log = x >> 1
        x_sign = x & 1

        offset = 2 * (torch.max(torch.abs(x_log)) + 1)
        x_logsign = torch.where(x_sign == 1, -offset-x_log, x_log)
        return torch.argmin(x_logsign, dim=dim, keepdim=keepdim)

@implements(torch.argmin, LNSArgminFunction.forward, "default", default=True)
def argmin(x, dim=None, keepdim=False, *, out=None):
    result = LNSArgminFunction.apply(x, dim, keepdim)

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
        return _clamp(ops, x, min, max)

    @staticmethod
    def setup_context(ctx, ops, inputs, output):
        x, min, max = inputs
        ctx.save_for_backward(x, min, max)

    @staticmethod
    def backward(ctx, ops, grad_output):
        x, min, max = ctx.saved_tensors

        grad_x = grad_output.clone()
        if min is not None:
            lt_mask = ops.lt(x, min)
            grad_x = torch.where(lt_mask, LNS_ZERO, grad_x)

        if max is not None:
            gt_mask = ops.gt(x, max)
            grad_x = torch.where(gt_mask, LNS_ZERO, grad_x)

        return grad_x, None, None

@implements(torch.clamp, _clamp, "default", default=True)
def clamp(x, min=None, max=None, *, out=None):
    x, min, max = format_lnstensor_operands(x, min, max)
    result = LNSClampFunction.apply(x, min, max)

    if out is not None:
        return out._inplace_copy(result)

    return result
import torch
from .. import LNS_ZERO, LNSTensor, lnstensor, format_lnstensor_operands, implements, zeros_like
from . import(
    lns_sub,
    lns_mul,
    lns_div,
    lns_neg,
    lns_sum,
    lns_square,
    lns_abs,
    lns_sign,
    lns_log,
    lns_add,
)

class LNSMSELossFunction(torch.autograd.Function):

    @staticmethod
    def forward(x, y, base, size_average=None, reduce=None, reduction='mean', weight=None):
        x_packed, y_packed = x.to(torch.int64), y.to(torch.int64)

        errors = lns_sub(x_packed, y_packed, base=base)
        squared_errors = lns_square(errors, base)

        if weight is not None:
            weight = weight.to(torch.int64)
            squared_errors = lns_mul(squared_errors, weight)

        if reduction == 'none':
            return squared_errors.to(torch.float64)

        elif reduction == 'sum':
            squared_error_sum = lns_sum(squared_errors, base)
            return squared_error_sum.to(torch.float64)

        elif reduction == 'mean':
            squared_error_sum = lns_sum(squared_errors, base)

            if weight is not None:
                weight_sum = lns_sum(weight, base)
                weighted_mean = lns_div(squared_error_sum, weight_sum, base)
                return weighted_mean.to(torch.float64)

            else:
                num_elements = x.numel()
                mean = lns_div(squared_error_sum, LNSTensor.get_internal_tensor(num_elements, base), base)
                return mean.to(torch.float64)

    @staticmethod
    def setup_context(ctx, inputs, output):
        x, y, base, _, _, reduction, weight = inputs
        ctx.reduction = reduction
        ctx.weighted = False if weight is None else True
        if ctx.weighted:
            ctx.save_for_backward(x, y, base, weight)
        else:
            ctx.save_for_backward(x, y, base)

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.weighted:
            x, y, base, weight = ctx.saved_tensors
            x_packed, y_packed = x.to(torch.int64), y.to(torch.int64)

            grad = lns_sub(x_packed, y_packed, base=base)
            grad = lns_mul(grad, LNSTensor.get_internal_tensor(2.0, base))
            grad = lns_mul(grad, weight)

            if ctx.reduction == 'mean':
                weight_sum = lns_sum(weight, base)
                grad = lns_div(grad, weight_sum, base)

        else:
            x, y, base = ctx.saved_tensors
            x_packed, y_packed = x.to(torch.int64), y.to(torch.int64)

            grad = lns_sub(x_packed, y_packed, base=base)
            grad = lns_mul(grad, LNSTensor.get_internal_tensor(2.0, base))

            if ctx.reduction == 'mean':
                num_elements = x.numel()
                grad = lns_div(grad, LNSTensor.get_internal_tensor(num_elements, base), base)

        return grad.to(torch.float64), lns_neg(grad).to(torch.float64), None, None, None, None, None

@implements(torch.nn.functional.mse_loss, LNSMSELossFunction.forward, key="default", default=True)
def mse_loss(x, y, size_average=None, reduce=None, reduction='mean', weight=None):

    if weight is None:
        x, y = format_lnstensor_operands(x, y)
        weight_lns = None
    else:
        x, y, weight = format_lnstensor_operands(x, y, weight)
        weight_lns = weight._lns

    result = LNSMSELossFunction.apply(x._lns, y._lns, x.base, size_average, reduce, reduction, weight_lns)

    return lnstensor(result, from_lns=True, b=x.base)

class LNSL1LossFunction(torch.autograd.Function):

    @staticmethod
    def forward(x, y, base, size_average=None, reduce=None, reduction='mean'):
        x_packed, y_packed = x.to(torch.int64), y.to(torch.int64)

        errors = lns_sub(x_packed, y_packed, base=base)
        abs_errors = lns_abs(errors)

        if reduction == 'none':
            return abs_errors.to(torch.float64)

        elif reduction == 'sum':
            abs_error_sum = lns_sum(abs_errors, base)
            return abs_error_sum.to(torch.float64)

        elif reduction == 'mean':
            abs_error_sum = lns_sum(abs_errors, base)
            num_elements = x.numel()
            abs_error_mean = lns_div(abs_error_sum, LNSTensor.get_internal_tensor(num_elements, base), base)
            return abs_error_mean.to(torch.float64)

    @staticmethod
    def setup_context(ctx, inputs, output):
        x, y, base, _, _, reduction = inputs
        ctx.save_for_backward(x, y, base)
        ctx.reduction = reduction

    @staticmethod
    def backward(ctx, grad_output):
        x, y, base = ctx.saved_tensors
        x_packed, y_packed = x.to(torch.int64), y.to(torch.int64)

        grad = lns_sub(x_packed, y_packed, base=base)
        grad = lns_sign(grad, base)

        if ctx.reduction == 'mean':
            num_elements = x.numel()
            grad = lns_div(grad, LNSTensor.get_internal_tensor(num_elements, base), base)

        return grad.to(torch.float64), lns_neg(grad).to(torch.float64), None, None, None, None

@implements(torch.nn.functional.l1_loss, LNSL1LossFunction.forward, key="default", default=True)
def l1_loss(x, y, size_average=None, reduce=None, reduction='mean'):

    x, y = format_lnstensor_operands(x, y)
    result = LNSL1LossFunction.apply(x._lns, y._lns, x.base, size_average, reduce, reduction)

    return lnstensor(result, from_lns=True, b=x.base)

class LNSBCELossFunction(torch.autograd.Function):

    @staticmethod
    def forward(x, y, base, weight=None, size_average=None, reduce=None, reduction='mean'):
        x_packed, y_packed = x.to(torch.int64), y.to(torch.int64)

        log_x = lns_log(x_packed, base)
        pos_log_prob = lns_mul(y_packed, log_x)

        x2 = lns_sub(LNSTensor.get_internal_tensor(1.0, base), x_packed, base)
        log_x2 = lns_log(x2, base)
        y2 = lns_sub(LNSTensor.get_internal_tensor(1.0, base), y_packed, base)
        neg_log_prob = lns_mul(y2, log_x2)

        loss = lns_add(pos_log_prob, neg_log_prob, base)
        if weight is not None:
            weight = weight.to(torch.int64)
            loss = lns_mul(loss, weight)
        loss = lns_neg(loss)

        if reduction == 'none':
            return loss.to(torch.float64)

        elif reduction == 'sum':
            loss_sum = lns_sum(loss, base)
            return loss_sum.to(torch.float64)

        elif reduction == 'mean':
            loss_sum = lns_sum(loss, base)

            if weight is not None:
                weight_sum = lns_sum(weight, base)
                weighted_mean = lns_div(loss_sum, weight_sum, base)
                return weighted_mean.to(torch.float64)

            else:
                num_elements = x.numel()
                mean = lns_div(loss_sum, LNSTensor.get_internal_tensor(num_elements, base), base)
                return mean.to(torch.float64)

    @staticmethod
    def setup_context(ctx, inputs, output):
        x, y, base, weight, _, _, reduction = inputs
        ctx.reduction = reduction
        ctx.weighted = False if weight is None else True
        if ctx.weighted:
            ctx.save_for_backward(x, y, base, weight)
        else:
            ctx.save_for_backward(x, y, base)

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.weighted:
            x, y, base, weight = ctx.saved_tensors
            x_packed, y_packed = x.to(torch.int64), y.to(torch.int64)

            one_minus_x = lns_sub(LNSTensor.get_internal_tensor(1.0, base), x_packed, base)
            one_minus_y = lns_sub(LNSTensor.get_internal_tensor(1.0, base), y_packed, base)
            term1 = lns_div(one_minus_y, one_minus_x, base)
            term2 = lns_div(y_packed, x_packed, base)

            grad_x = lns_sub(term1, term2, base)
            grad_x = lns_mul(grad_x, weight)

            grad_y = lns_div(x_packed, one_minus_x, base)
            grad_y = lns_log(grad_y, base)
            grad_y = lns_mul(grad_y, weight)
            grad_y = lns_neg(grad_y)

            if ctx.reduction == 'mean':
                weight_sum = lns_sum(weight, base)
                grad_x = lns_div(grad_x, weight_sum, base)
                grad_y = lns_div(grad_y, weight_sum, base)

        else:
            x, y, base = ctx.saved_tensors
            x_packed, y_packed = x.to(torch.int64), y.to(torch.int64)

            one_minus_x = lns_sub(LNSTensor.get_internal_tensor(1.0, base), x_packed, base)
            one_minus_y = lns_sub(LNSTensor.get_internal_tensor(1.0, base), y_packed, base)
            term1 = lns_div(one_minus_y, one_minus_x, base)
            term2 = lns_div(y_packed, x_packed, base)

            grad_x = lns_sub(term1, term2, base)
            grad_y = lns_div(x_packed, one_minus_x, base)
            grad_y = lns_log(grad_y, base)
            grad_y = lns_neg(grad_y)

            if ctx.reduction == 'mean':
                num_elements = x.numel()
                grad_x = lns_div(grad_x, LNSTensor.get_internal_tensor(num_elements, base), base)
                grad_y = lns_div(grad_y, LNSTensor.get_internal_tensor(num_elements, base), base)

        return grad_x, grad_y, None, None, None, None, None

@implements(torch.nn.functional.binary_cross_entropy, LNSBCELossFunction.forward, key="default", default=True)
def binary_cross_entropy(x, y, weight=None, size_average=None, reduce=None, reduction='mean'):

    if weight is None:
        x, y = format_lnstensor_operands(x, y)
        weight_lns = None
    else:
        x, y, weight = format_lnstensor_operands(x, y, weight)
        weight_lns = weight._lns

    result = LNSBCELossFunction.apply(x._lns, y._lns, x.base, weight_lns, size_average, reduce, reduction)

    return lnstensor(result, from_lns=True, b=x.base)

# currently doesn't support ignore_index
class LNSNLLLossFunction(torch.autograd.Function):

    @staticmethod
    def forward(x, y, base, weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean'):
        x_packed = x.to(torch.int64)

        if x_packed.dim() == 1:
            nll = x_packed[y]
        else:
            nll = x_packed.gather(1, y.view(-1, 1)).squeeze(1)

        if weight is not None:
            weight = weight.to(torch.int64)
            sample_weights = weight[y]
            nll = lns_mul(nll, sample_weights)

        loss = lns_neg(nll)

        if reduction == 'none':
            return loss.to(torch.float64)
        
        elif reduction == 'sum':
            loss_sum = lns_sum(loss, base)
            return loss_sum.to(torch.float64)
        
        elif reduction == 'mean':
            loss_sum = lns_sum(loss, base)

            if weight is not None:
                weight_sum = lns_sum(sample_weights, base)
                weighted_mean = lns_div(loss_sum, weight_sum, base)
                return weighted_mean.to(torch.float64)

            else:
                batch_size = LNSTensor.get_internal_tensor(y.size(0), base)
                mean = lns_div(loss_sum, batch_size, base)
                return mean.to(torch.float64)

    @staticmethod
    def setup_context(ctx, inputs, output):
        x, y, base, weight, _, _, _, reduction = inputs
        ctx.reduction = reduction
        ctx.weighted = True if weight is not None else False
        if ctx.weighted:
            ctx.save_for_backward(x, y, base, weight)
        else:
            ctx.save_for_backward(x, y, base)

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.weighted:
            x, y, base, weight = ctx.saved_tensors

            grad_x = zeros_like(x)._lns
            if grad_x.dim() == 1:
                grad_x[y] = lns_neg(weight[y])
            
            else:
                batch_size = y.size(0)
                indices = torch.arange(batch_size)
                grad_x[indices, y] = lns_neg(weight[y])

            if ctx.reduction == 'mean':
                weight_sum = lns_sum(weight, base)
                grad_x = lns_div(grad_x, weight_sum, base)

        else:
            x, y, base = ctx.saved_tensors

            grad_x = zeros_like(x)._lns
            if grad_x.dim() == 1:
                grad_x[y] = LNSTensor.get_internal_tensor(-1.0, base)
            
            else:
                batch_size = y.size(0)
                indices = torch.arange(batch_size)
                grad_x[indices, y] = LNSTensor.get_internal_tensor(-1.0, base)

            if ctx.reduction == 'mean':
                batch_size = LNSTensor.get_internal_tensor(y.size(0), base)
                grad_x = lns_div(grad_x, batch_size, base)

        if ctx.reduction == 'none':
            if grad_x.dim() == 1:
                grad_x = lns_mul(grad_x, grad_output)

            else:
                batch_size = y.size(0)
                indices = torch.arange(batch_size)
                grad_x[indices, y] = lns_mul(grad_x[indices, y], grad_output)

        else:
            grad_x = lns_mul(grad_x, grad_output)

        return grad_x, None, None, None, None, None, None, None

@implements(torch.nn.functional.nll_loss, LNSNLLLossFunction.forward, key="default", default=True)
def nll_loss(x, y, weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean'):

    if not torch.is_tensor(y) or torch.is_floating_point(y) or torch.is_complex(y):
        raise TypeError("Expected target to be a tensor of type LongTensor, but got: {}".format(type(y)))

    if weight is None:
        weight_lns = None
    else:
        x, weight = format_lnstensor_operands(x, y, weight)
        weight_lns = weight._lns

    result = LNSNLLLossFunction.apply(x._lns, y, x.base, weight_lns, size_average, ignore_index, reduce, reduction)

    return lnstensor(result, from_lns=True, b=x.base)
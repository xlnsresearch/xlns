import math

import torch
from xlnstorch import LNS_ZERO, LNS_ONE, LNS_NEG_ONE, LNSTensor, lnstensor, format_lnstensor_operands, implements, zeros_like
from xlnstorch.autograd import LNSFunction
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
    lns_sigmoid,
    lns_exp,
    lns_gt,
    lns_eq,
    lns_maximum,
    lns_reciprocal,
    lns_lt,
    lns_max,
)

class LNSMSELossFunction(LNSFunction):

    @staticmethod
    def forward(x, y, base, size_average=None, reduce=None, reduction='mean', weight=None):
        x_packed, y_packed = x.to(torch.int64), y.to(torch.int64)

        errors = lns_sub(x_packed, y_packed, base=base)
        squared_errors = lns_square(errors, base)

        if weight is not None:
            weight = weight.to(torch.int64)
            squared_errors = lns_mul(squared_errors, weight, base)

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
            grad = lns_mul(grad, LNSTensor.get_internal_tensor(2.0, base), base)
            grad = lns_mul(grad, weight, base)

            if ctx.reduction == 'mean':
                weight_sum = lns_sum(weight, base)
                grad = lns_div(grad, weight_sum, base)

        else:
            x, y, base = ctx.saved_tensors
            x_packed, y_packed = x.to(torch.int64), y.to(torch.int64)

            grad = lns_sub(x_packed, y_packed, base=base)
            grad = lns_mul(grad, LNSTensor.get_internal_tensor(2.0, base), base)

            if ctx.reduction == 'mean':
                num_elements = x.numel()
                grad = lns_div(grad, LNSTensor.get_internal_tensor(num_elements, base), base)

        return grad.to(torch.float64), lns_neg(grad).to(torch.float64), None, None, None, None, None

@implements(torch.nn.functional.mse_loss, LNSMSELossFunction.forward, key="default", default=True)
def mse_loss(x, y, size_average=None, reduce=None, reduction='mean', weight=None):

    if weight is None:
        x, y = format_lnstensor_operands(x, y)
    else:
        x, y, weight = format_lnstensor_operands(x, y, weight)

    result = LNSMSELossFunction.apply(x, y, x.base, size_average,
                                      reduce, reduction, weight)

    return lnstensor(result, from_lns=True, b=x.base)

class LNSL1LossFunction(LNSFunction):

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
    result = LNSL1LossFunction.apply(x, y, x.base, size_average, reduce, reduction)

    return lnstensor(result, from_lns=True, b=x.base)

class LNSBCELossFunction(LNSFunction):

    @staticmethod
    def forward(x, y, base, weight=None, size_average=None, reduce=None, reduction='mean'):
        x_packed, y_packed = x.to(torch.int64), y.to(torch.int64)

        log_x = lns_log(x_packed, base)
        pos_log_prob = lns_mul(y_packed, log_x, base)

        x2 = lns_sub(LNS_ONE, x_packed, base)
        log_x2 = lns_log(x2, base)
        y2 = lns_sub(LNS_ONE, y_packed, base)
        neg_log_prob = lns_mul(y2, log_x2, base)

        loss = lns_add(pos_log_prob, neg_log_prob, base)
        if weight is not None:
            weight = weight.to(torch.int64)
            loss = lns_mul(loss, weight, base)
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

            one_minus_x = lns_sub(LNS_ONE, x_packed, base)
            one_minus_y = lns_sub(LNS_ONE, y_packed, base)
            term1 = lns_div(one_minus_y, one_minus_x, base)
            term2 = lns_div(y_packed, x_packed, base)

            grad_x = lns_sub(term1, term2, base)
            grad_x = lns_mul(grad_x, weight, base)

            grad_y = lns_div(x_packed, one_minus_x, base)
            grad_y = lns_log(grad_y, base)
            grad_y = lns_mul(grad_y, weight, base)
            grad_y = lns_neg(grad_y)

            if ctx.reduction == 'mean':
                weight_sum = lns_sum(weight, base)
                grad_x = lns_div(grad_x, weight_sum, base)
                grad_y = lns_div(grad_y, weight_sum, base)

        else:
            x, y, base = ctx.saved_tensors
            x_packed, y_packed = x.to(torch.int64), y.to(torch.int64)

            one_minus_x = lns_sub(LNS_ONE, x_packed, base)
            one_minus_y = lns_sub(LNS_ONE, y_packed, base)
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
    else:
        x, y, weight = format_lnstensor_operands(x, y, weight)

    result = LNSBCELossFunction.apply(x, y, x.base, weight, size_average, reduce, reduction)

    return lnstensor(result, from_lns=True, b=x.base)

# doesn't implement pos_weight yet
class LNSBCEWithLogitsLossFunction(LNSFunction):

    @staticmethod
    def forward(x, y, base, weight=None, size_average=None, reduce=None, reduction='mean', pos_weight=None):
        x_packed, y_packed = x.to(torch.int64), y.to(torch.int64)

        sigmoid_x = lns_sigmoid(x_packed, base)
        log_sigmoid_x = lns_log(sigmoid_x, base)
        pos_log_prob = lns_mul(y_packed, log_sigmoid_x, base)

        sigmoid_x2 = lns_sub(LNS_ONE, sigmoid_x, base)
        log_sigmoid_x2 = lns_log(sigmoid_x2, base)
        y2 = lns_sub(LNS_ONE, y_packed, base)
        neg_log_prob = lns_mul(y2, log_sigmoid_x2, base)

        loss = lns_add(pos_log_prob, neg_log_prob, base)
        if weight is not None:
            weight = weight.to(torch.int64)
            loss = lns_mul(loss, weight, base)
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
        x, y, base, weight, _, _, reduction, _ = inputs
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

            sigmoid_x = lns_sigmoid(x_packed, base)
            grad_x = lns_sub(sigmoid_x, y_packed, base)
            grad_x = lns_mul(grad_x, weight, base)

            grad_y = lns_mul(x_packed, weight, base)
            grad_y = lns_neg(grad_y)

            if ctx.reduction == 'mean':
                weight_sum = lns_sum(weight, base)
                grad_x = lns_div(grad_x, weight_sum, base)
                grad_y = lns_div(grad_y, weight_sum, base)

        else:
            x, y, base = ctx.saved_tensors
            x_packed, y_packed = x.to(torch.int64), y.to(torch.int64)

            sigmoid_x = lns_sigmoid(x_packed, base)
            grad_x = lns_sub(sigmoid_x, y_packed, base)
            grad_y = lns_neg(x_packed)

            if ctx.reduction == 'mean':
                num_elements = x.numel()
                grad_x = lns_div(grad_x, LNSTensor.get_internal_tensor(num_elements, base), base)
                grad_y = lns_div(grad_y, LNSTensor.get_internal_tensor(num_elements, base), base)

        return grad_x, grad_y, None, None, None, None, None, None

@implements(torch.nn.functional.binary_cross_entropy_with_logits, LNSBCEWithLogitsLossFunction.forward, key="default", default=True)
def binary_cross_entropy_with_logits(x, y, weight=None, size_average=None, reduce=None, reduction='mean', pos_weight=None):

    if weight is None:
        x, y = format_lnstensor_operands(x, y)
    else:
        x, y, weight = format_lnstensor_operands(x, y, weight)

    result = LNSBCEWithLogitsLossFunction.apply(x, y, x.base, weight, size_average, reduce, reduction)

    return lnstensor(result, from_lns=True, b=x.base)

# currently doesn't support ignore_index
class LNSNLLLossFunction(LNSFunction):

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
            nll = lns_mul(nll, sample_weights, base)

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
                grad_x[y] = LNS_NEG_ONE.clone()
            
            else:
                batch_size = y.size(0)
                indices = torch.arange(batch_size)
                grad_x[indices, y] = LNS_NEG_ONE.clone()

            if ctx.reduction == 'mean':
                batch_size = LNSTensor.get_internal_tensor(y.size(0), base)
                grad_x = lns_div(grad_x, batch_size, base)

        if ctx.reduction == 'none':
            if grad_x.dim() == 1:
                grad_x = lns_mul(grad_x, grad_output, base)

            else:
                batch_size = y.size(0)
                indices = torch.arange(batch_size)
                grad_x[indices, y] = lns_mul(grad_x[indices, y], grad_output, base)

        else:
            grad_x = lns_mul(grad_x, grad_output, base)

        return grad_x, None, None, None, None, None, None, None

@implements(torch.nn.functional.nll_loss, LNSNLLLossFunction.forward, key="default", default=True)
def nll_loss(x, y, weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean'):

    assert isinstance(y, torch.Tensor), "y must be a torch.Tensor"

    if weight is not None:
        x, weight = format_lnstensor_operands(x, weight)

    result = LNSNLLLossFunction.apply(x, y, x.base, weight, size_average, ignore_index, reduce, reduction)

    return lnstensor(result, from_lns=True, b=x.base)

class PoissonNLLLossFunction(LNSFunction):

    @staticmethod
    def forward(x, y, eps, base, log_input=True, full=False, size_average=None, reduce=None, reduction='mean'):
        x_packed, y_packed, eps_packed = x.to(torch.int64), y.to(torch.int64), eps.to(torch.int64)

        if log_input:
            exp_x = lns_exp(x_packed, base)
            loss = lns_sub(exp_x, lns_mul(y_packed, x_packed, base), base)
        else:
            log_x = lns_log(lns_add(x_packed, eps_packed, base), base)
            loss = lns_sub(x_packed, lns_mul(y_packed, log_x, base), base)

        if full:
            y_clamped = torch.where(lns_gt(y_packed, LNS_ONE), y_packed, LNS_ONE)

            two_pi = LNSTensor.get_internal_tensor(math.tau, base)
            stirling_term1 = lns_mul(y_clamped, lns_log(y_clamped, base), base)
            stirling_term3 = lns_mul(lns_log(lns_mul(two_pi, y_clamped, base), base), LNSTensor.get_internal_tensor(0.5, base), base)
            stirling = lns_add(lns_sub(stirling_term1, y_clamped, base), stirling_term3, base)

            loss = lns_add(loss, stirling, base)

        if reduction == 'none':
            return loss.to(torch.float64)

        elif reduction == 'sum':
            loss_sum = lns_sum(loss, base)
            return loss_sum.to(torch.float64)

        elif reduction == 'mean':
            loss_sum = lns_sum(loss, base)
            num_elements = x.numel()
            mean = lns_div(loss_sum, LNSTensor.get_internal_tensor(num_elements, base), base)
            return mean.to(torch.float64)

    @staticmethod
    def setup_context(ctx, inputs, output):
        x, y, eps, base, log_input, full, _, _, reduction = inputs
        ctx.save_for_backward(x, y, eps, base)
        ctx.log_input = log_input
        ctx.full = full
        ctx.reduction = reduction

    @staticmethod
    def backward(ctx, grad_output):
        x, y, eps, base = ctx.saved_tensors
        x_packed, y_packed, eps_packed = x.to(torch.int64), y.to(torch.int64), eps.to(torch.int64)

        if ctx.log_input:
            grad_x = lns_sub(lns_exp(x_packed, base), y_packed, base)
            grad_y = lns_neg(x_packed)

        else:
            grad_x = lns_div(y_packed, lns_add(x_packed, eps_packed, base), base)
            grad_x = lns_sub(LNS_ONE, grad_x, base)
            grad_y = lns_neg(lns_log(lns_add(x_packed, eps_packed, base), base))

        if ctx.full:
            stirling_grad = torch.where(lns_gt(y_packed, LNS_ONE),
                                       lns_add(lns_log(y_packed, base), lns_div(
                                           LNSTensor.get_internal_tensor(0.5, base), y_packed, base), base),
                                       LNS_ZERO)
            grad_y = lns_add(grad_y, stirling_grad, base)

        if ctx.reduction == 'mean':
            num_elements = x.numel()
            grad_x = lns_div(grad_x, LNSTensor.get_internal_tensor(num_elements, base), base)

        grad_x = lns_mul(grad_x, grad_output, base)
        grad_y = lns_mul(grad_y, grad_output, base)

        return grad_x, grad_y, None, None, None, None, None, None, None

@implements(torch.nn.functional.poisson_nll_loss, PoissonNLLLossFunction.forward, key="default", default=True)
def poisson_nll_loss(x, y, log_input=True, full=False, size_average=None, eps=1e-08, reduce=None, reduction='mean'):

    x, y, eps = format_lnstensor_operands(x, y, eps)
    result = PoissonNLLLossFunction.apply(x, y, eps, x.base, log_input, full, size_average, reduce, reduction)

    return lnstensor(result, from_lns=True, b=x.base)

class LNSHingeEmbeddingLossFunction(LNSFunction):

    @staticmethod
    def forward(x, y, margin, base, size_average=None, reduce=None, reduction='mean'):
        x_packed, y_packed, margin_packed = x.to(torch.int64), y.to(torch.int64), margin.to(torch.int64)

        positive_mask = lns_eq(y_packed, LNS_ONE)
        loss = torch.where(positive_mask, x, lns_maximum(LNS_ZERO, lns_sub(margin_packed, x_packed, base), base))

        if reduction == 'none':
            return loss.to(torch.float64)

        elif reduction == 'sum':
            loss_sum = lns_sum(loss, base)
            return loss_sum.to(torch.float64)

        elif reduction == 'mean':
            loss_sum = lns_sum(loss, base)
            num_elements = x.numel()
            mean = lns_div(loss_sum, LNSTensor.get_internal_tensor(num_elements, base), base)
            return mean.to(torch.float64)
        
    @staticmethod
    def setup_context(ctx, inputs, output):
        x, y, margin, base, _, _, reduction = inputs
        ctx.save_for_backward(x, y, margin, base)
        ctx.reduction = reduction

    @staticmethod
    def backward(ctx, grad_output):
        x, y, margin, base = ctx.saved_tensors
        y_packed, margin_packed = y.to(torch.int64), margin.to(torch.int64)

        grad_x = torch.where(lns_eq(y_packed, LNS_ONE),
                             LNS_ONE,
                             torch.where(lns_gt(lns_sub(margin_packed, x, base), LNS_ZERO),
                                         LNS_NEG_ONE, LNS_ZERO))

        if ctx.reduction == 'mean':
            num_elements = x.numel()
            grad_x = lns_div(grad_x, LNSTensor.get_internal_tensor(num_elements, base), base)

        grad_x = lns_mul(grad_x, grad_output, base)

        return grad_x, None, None, None, None, None, None

@implements(torch.nn.functional.hinge_embedding_loss, LNSHingeEmbeddingLossFunction.forward, key="default", default=True)
def hinge_embedding_loss(x, y, margin=1.0, size_average=None, reduce=None, reduction='mean'):

    x, y, margin = format_lnstensor_operands(x, y, margin)
    result = LNSHingeEmbeddingLossFunction.apply(x, y, margin, x.base, size_average, reduce, reduction)

    return lnstensor(result, from_lns=True, b=x.base)

class LNSKLDivLossFunction(LNSFunction):

    @staticmethod
    def forward(x, y, base, size_average=None, reduce=None, reduction='mean', log_target=False):
        x_packed, y_packed = x.to(torch.int64), y.to(torch.int64)

        if log_target:
            loss = lns_mul(lns_exp(y_packed, base), lns_sub(y_packed, x_packed, base), base)
        else:
            loss = lns_mul(y_packed, lns_sub(lns_log(y_packed, base), x_packed, base), base)

        if reduction == 'none':
            return loss.to(torch.float64)

        elif reduction == 'sum':
            loss_sum = lns_sum(loss, base)
            return loss_sum.to(torch.float64)

        elif reduction == 'mean':
            loss_sum = lns_sum(loss, base)
            num_elements = x.numel()
            mean = lns_div(loss_sum, LNSTensor.get_internal_tensor(num_elements, base), base)
            return mean.to(torch.float64)

        elif reduction == 'batchmean':
            loss_sum = lns_sum(loss, base)
            num_elements = x.size(0)
            batch_mean = lns_div(loss_sum, LNSTensor.get_internal_tensor(num_elements, base), base)
            return batch_mean.to(torch.float64)

    @staticmethod
    def setup_context(ctx, inputs, output):
        x, y, base, _, _, reduction, log_target = inputs
        ctx.save_for_backward(x, y, output, base)
        ctx.reduction = reduction
        ctx.log_target = log_target

    @staticmethod
    def backward(ctx, grad_output):
        x, y, output, base = ctx.saved_tensors
        x_packed, y_packed, output_packed = x.to(torch.int64), y.to(torch.int64), output.to(torch.int64)

        if ctx.log_target:
            exp_y = lns_exp(y_packed, base)
            grad_x = lns_neg(exp_y)
            grad_y = lns_mul(exp_y, lns_add(lns_sub(y_packed, x_packed, base), LNS_ONE, base), base)
        else:
            grad_x = lns_neg(y_packed)
            grad_y = lns_add(lns_sub(lns_log(y_packed, base), x_packed, base), LNS_ONE, base)

        if ctx.reduction == 'mean':
            num_elements = LNSTensor.get_internal_tensor(x.numel(), base)
            grad_x = lns_div(grad_x, num_elements, base)
            grad_y = lns_div(grad_y, num_elements, base)

        elif ctx.reduction == 'batchmean':
            num_elements = LNSTensor.get_internal_tensor(x.size(0), base)
            grad_x = lns_div(grad_x, num_elements, base)
            grad_y = lns_div(grad_y, num_elements, base)

        grad_x = lns_mul(grad_x, grad_output, base)
        grad_y = lns_mul(grad_y, grad_output, base)

        return grad_x, grad_y, None, None, None, None, None

@implements(torch.nn.functional.kl_div, LNSKLDivLossFunction.forward, key="default", default=True)
def kl_div(x, y, size_average=None, reduce=None, reduction='mean', log_target=False):

    x, y = format_lnstensor_operands(x, y)
    result = LNSKLDivLossFunction.apply(x, y, x.base, size_average, reduce, reduction, log_target)

    return lnstensor(result, from_lns=True, b=x.base)

class LNSMarginRankingLossFunction(LNSFunction):

    @staticmethod
    def forward(x1, x2, y, margin, base, size_average=None, reduce=None, reduction='mean'):
        x1_packed, x2_packed, y_packed = x1.to(torch.int64), x2.to(torch.int64), y.to(torch.int64)

        loss = lns_sub(x1_packed, x2_packed, base)
        loss = lns_mul(loss, y_packed, base)
        loss = lns_sub(margin, loss, base)
        loss = lns_maximum(LNS_ZERO, loss, base)

        if reduction == 'none':
            return loss.to(torch.float64)

        elif reduction == 'sum':
            loss_sum = lns_sum(loss, base)
            return loss_sum.to(torch.float64)

        elif reduction == 'mean':
            loss_sum = lns_sum(loss, base)
            num_elements = x1.numel()
            mean = lns_div(loss_sum, LNSTensor.get_internal_tensor(num_elements, base), base)
            return mean.to(torch.float64)

    @staticmethod
    def setup_context(ctx, inputs, output):
        x1, x2, y, margin, base, _, _, reduction = inputs
        ctx.save_for_backward(x1, x2, y, margin, base)
        ctx.reduction = reduction

    @staticmethod
    def backward(ctx, grad_output):
        x1, x2, y, margin, base = ctx.saved_tensors
        x1_packed, x2_packed, y_packed, margin_packed = x1.to(torch.int64), x2.to(torch.int64), y.to(torch.int64), margin.to(torch.int64)

        loss = lns_sub(x1_packed, x2_packed, base)
        loss = lns_mul(loss, y_packed, base)
        loss = lns_sub(margin, loss, base)
        gt_zero_mask = lns_gt(loss, LNS_ZERO)

        grad_x1 = torch.where(gt_zero_mask, lns_neg(y), LNS_ZERO)
        grad_x2 = torch.where(gt_zero_mask, y, LNS_ZERO)
        grad_y = torch.where(gt_zero_mask, lns_sub(x2_packed, x1_packed, base), LNS_ZERO)

        if ctx.reduction == 'mean':
            num_elements = LNSTensor.get_internal_tensor(x1.numel(), base)
            grad_x1 = lns_div(grad_x1, num_elements, base)
            grad_x2 = lns_div(grad_x2, num_elements, base)
            grad_y = lns_div(grad_y, num_elements, base)

        grad_x1 = lns_mul(grad_x1, grad_output, base)
        grad_x2 = lns_mul(grad_x2, grad_output, base)
        grad_y = lns_mul(grad_y, grad_output, base)

        return grad_x1, grad_x2, grad_y, None, None, None, None, None

@implements(torch.nn.functional.margin_ranking_loss, LNSMarginRankingLossFunction.forward, key="default", default=True)
def margin_ranking_loss(x1, x2, y, margin=0.0, size_average=None, reduce=None, reduction='mean'):

    x1, x2, y, margin = format_lnstensor_operands(x1, x2, y, margin)
    result = LNSMarginRankingLossFunction.apply(x1, x2, y, margin, x1.base, size_average, reduce, reduction)

    return lnstensor(result, from_lns=True, b=x1.base)

class LNSGaussianNLLLossFunction(LNSFunction):

    @staticmethod
    def forward(x, y, var, eps, base, full=False, reduction='mean'):
        x_packed, y_packed, var_packed, eps_packed = x.to(torch.int64), y.to(torch.int64), var.to(torch.int64), eps.to(torch.int64)

        var_eps = lns_maximum(var_packed, eps_packed, base)
        loss = lns_square(lns_sub(x_packed, y_packed, base), base)
        loss = lns_add(lns_log(var_eps, base), lns_div(loss, var_eps, base), base)

        if full:
            two_pi = LNSTensor.get_internal_tensor(math.tau, base)
            loss = lns_add(loss, lns_log(two_pi, base), base)

        loss = lns_div(loss, LNSTensor.get_internal_tensor(2.0, base), base)

        if reduction == 'none':
            return loss.to(torch.float64)

        elif reduction == 'sum':
            loss_sum = lns_sum(loss, base)
            return loss_sum.to(torch.float64)

        elif reduction == 'mean':
            loss_sum = lns_sum(loss, base)
            num_elements = x.numel()
            mean = lns_div(loss_sum, LNSTensor.get_internal_tensor(num_elements, base), base)
            return mean.to(torch.float64)

    @staticmethod
    def setup_context(ctx, inputs, output):
        x, y, var, eps, base, _, reduction = inputs
        ctx.save_for_backward(x, y, var, eps, base)
        ctx.reduction = reduction

    @staticmethod
    def backward(ctx, grad_output):
        x, y, var, eps, base = ctx.saved_tensors
        x_packed, y_packed, var_packed, eps_packed = x.to(torch.int64), y.to(torch.int64), var.to(torch.int64), eps.to(torch.int64)

        var_eps = lns_maximum(var_packed, eps_packed, base)
        grad_x = lns_div(lns_sub(x_packed, y_packed, base), var_eps, base)
        grad_y = lns_neg(grad_x)

        grad_var = lns_square(lns_div(lns_sub(x_packed, y_packed, base), var_packed, base), base)
        grad_var = lns_sub(lns_reciprocal(var_packed, base), grad_var, base)
        grad_var = lns_div(grad_var, LNSTensor.get_internal_tensor(2.0, base), base)

        if ctx.reduction == 'mean':
            num_elements = LNSTensor.get_internal_tensor(x.numel(), base)
            grad_x = lns_div(grad_x, num_elements, base)
            grad_y = lns_div(grad_y, num_elements, base)
            grad_var = lns_div(grad_var, num_elements, base)

        grad_x = lns_mul(grad_x, grad_output, base)
        grad_y = lns_mul(grad_y, grad_output, base)
        grad_var = lns_mul(grad_var, grad_output, base)

        return grad_x, grad_y, grad_var, None, None, None, None

@implements(torch.nn.functional.gaussian_nll_loss, LNSGaussianNLLLossFunction.forward, key="default", default=True)
def gaussian_nll_loss(x, y, var, full=False, eps=1e-6, reduction='mean'):

    x, y, var, eps = format_lnstensor_operands(x, y, var, eps)
    result = LNSGaussianNLLLossFunction.apply(x, y, var, eps, x.base, full, reduction)

    return lnstensor(result, from_lns=True, b=x.base)

class LNSHuberLossFunction(LNSFunction):

    @staticmethod
    def forward(x, y, delta, base, reduction='mean', weight=None):
        x_packed, y_packed, delta_packed = x.to(torch.int64), y.to(torch.int64), delta.to(torch.int64)

        two = LNSTensor.get_internal_tensor(2.0, base)

        abs_diff = lns_abs(lns_sub(x_packed, y_packed, base))
        l1_term = lns_sub(abs_diff, lns_div(delta_packed, two, base), base)
        l1_term = lns_mul(l1_term, delta_packed, base)

        l2_term = lns_square(lns_sub(x_packed, y_packed, base), base)
        l2_term = lns_div(l2_term, two, base)

        loss = torch.where(lns_lt(abs_diff, delta_packed), l2_term, l1_term)
        if weight is not None:
            weight = weight.to(torch.int64)
            loss = lns_mul(loss, weight, base)

        if reduction == 'none':
            return loss.to(torch.float64)

        elif reduction == 'sum':
            loss_sum = lns_sum(loss, base)
            return loss_sum.to(torch.float64)

        elif reduction == 'mean':
            loss_sum = lns_sum(loss, base)
            num_elements = x.numel()
            mean = lns_div(loss_sum, LNSTensor.get_internal_tensor(num_elements, base), base)
            return mean.to(torch.float64)

    @staticmethod
    def setup_context(ctx, inputs, output):
        x, y, delta, base, reduction, weight = inputs
        ctx.reduction = reduction
        ctx.weighted = False if weight is None else True
        if ctx.weighted:
            ctx.save_for_backward(x, y, delta, base, weight)
        else:
            ctx.save_for_backward(x, y, delta, base)

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.weighted:
            x, y, delta, base, weight = ctx.saved_tensors
        else:
            x, y, delta, base = ctx.saved_tensors

        x_packed, y_packed, delta_packed = x.to(torch.int64), y.to(torch.int64), delta.to(torch.int64)
        two = LNSTensor.get_internal_tensor(2.0, base)

        l2_loss_grad_x = lns_sub(x_packed, y_packed, base)
        l2_loss_grad_y = lns_neg(l2_loss_grad_x)
        l1_loss_grad_x = lns_mul(lns_sign(lns_sub(x_packed, y_packed, base), base), delta_packed, base)
        l1_loss_grad_y = lns_neg(l1_loss_grad_x)

        abs_diff = lns_abs(lns_sub(x_packed, y_packed, base))
        l2_mask = lns_lt(abs_diff, delta_packed)
        grad_x = torch.where(l2_mask, l2_loss_grad_x, l1_loss_grad_x)
        grad_y = torch.where(l2_mask, l2_loss_grad_y, l1_loss_grad_y)

        if ctx.weighted:
            weight = weight.to(torch.int64)

            two = LNSTensor.get_internal_tensor(2.0, base)

            abs_diff = lns_abs(lns_sub(x_packed, y_packed, base))
            l1_term = lns_sub(abs_diff, lns_div(delta_packed, two, base), base)
            l1_term = lns_mul(l1_term, delta_packed, base)

            l2_term = lns_square(lns_sub(x_packed, y_packed, base), base)
            l2_term = lns_div(l2_term, two, base)

            grad_w = torch.where(lns_lt(abs_diff, delta_packed), l2_term, l1_term)
            grad_x = lns_mul(grad_x, weight, base)
            grad_y = lns_mul(grad_y, weight, base)

        else:
            grad_w = None

        if ctx.reduction == 'mean':
            num_elements = x.numel()
            grad_x = lns_div(grad_x, LNSTensor.get_internal_tensor(num_elements, base), base)
            grad_y = lns_div(grad_y, LNSTensor.get_internal_tensor(num_elements, base), base)
            if grad_w is not None:
                grad_w = lns_div(grad_w, LNSTensor.get_internal_tensor(num_elements, base), base)

        grad_x = lns_mul(grad_x, grad_output, base)
        grad_y = lns_mul(grad_y, grad_output, base)

        return grad_x, grad_y, None, None, None, grad_w

@implements(torch.nn.functional.huber_loss, LNSHuberLossFunction.forward, key="default", default=True)
def huber_loss(x, y, delta=1.0, reduction='mean', weight=None):

    if weight is None:
        x, y, delta = format_lnstensor_operands(x, y, delta)
    else:
        x, y, delta, weight = format_lnstensor_operands(x, y, delta, weight)

    result = LNSHuberLossFunction.apply(x, y, delta, x.base, reduction, weight)

    return lnstensor(result, from_lns=True, b=x.base)

class LNSSmoothL1LossFunction(LNSFunction):

    @staticmethod
    def forward(x, y, beta, base, size_average=None, reduce=None, reduction='mean'):
        x_packed, y_packed, beta_packed = x.to(torch.int64), y.to(torch.int64), beta.to(torch.int64)

        two = LNSTensor.get_internal_tensor(2.0, base)

        abs_diff = lns_abs(lns_sub(x_packed, y_packed, base))
        l1_term = lns_sub(abs_diff, lns_div(beta_packed, two, base), base)

        l2_term = lns_square(lns_sub(x_packed, y_packed, base), base)
        l2_term = lns_div(l2_term, lns_mul(two, beta_packed, base), base)

        loss = torch.where(lns_lt(abs_diff, beta_packed), l2_term, l1_term)

        if reduction == 'none':
            return loss.to(torch.float64)

        elif reduction == 'sum':
            loss_sum = lns_sum(loss, base)
            return loss_sum.to(torch.float64)

        elif reduction == 'mean':
            loss_sum = lns_sum(loss, base)
            num_elements = x.numel()
            mean = lns_div(loss_sum, LNSTensor.get_internal_tensor(num_elements, base), base)
            return mean.to(torch.float64)
    
    @staticmethod
    def setup_context(ctx, inputs, output):
        x, y, beta, base, _, _, reduction = inputs
        ctx.save_for_backward(x, y, beta, base)
        ctx.reduction = reduction

    @staticmethod
    def backward(ctx, grad_output):
        x, y, beta, base = ctx.saved_tensors
        x_packed, y_packed, beta_packed = x.to(torch.int64), y.to(torch.int64), beta.to(torch.int64)

        l2_loss_grad_x = lns_div(lns_sub(x_packed, y_packed, base), beta_packed, base)
        l2_loss_grad_y = lns_neg(l2_loss_grad_x)
        l1_loss_grad_x = lns_sign(lns_sub(x_packed, y_packed, base), base)
        l1_loss_grad_y = lns_neg(l1_loss_grad_x)

        abs_diff = lns_abs(lns_sub(x_packed, y_packed, base))
        l2_mask = lns_lt(abs_diff, beta_packed)
        grad_x = torch.where(l2_mask, l2_loss_grad_x, l1_loss_grad_x)
        grad_y = torch.where(l2_mask, l2_loss_grad_y, l1_loss_grad_y)

        if ctx.reduction == 'mean':
            num_elements = LNSTensor.get_internal_tensor(x.numel(), base)
            grad_x = lns_div(grad_x, num_elements, base)
            grad_y = lns_div(grad_y, num_elements, base)

        grad_x = lns_mul(grad_x, grad_output, base)
        grad_y = lns_mul(grad_y, grad_output, base)

        return grad_x, grad_y, None, None, None, None, None
    
@implements(torch.nn.functional.smooth_l1_loss, LNSSmoothL1LossFunction.forward, key="default", default=True)
def smooth_l1_loss(x, y, size_average=None, reduce=None, reduction='mean', beta=1.0):

    x, y, beta = format_lnstensor_operands(x, y, beta)
    result = LNSSmoothL1LossFunction.apply(x, y, beta, x.base, size_average, reduce, reduction)

    return lnstensor(result, from_lns=True, b=x.base)

class LNSCrossEntropyLossFunction(LNSFunction):

    @staticmethod
    def forward(x, y, base, weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean'):
        x_packed = x.to(torch.int64)
        dim = -1 if x_packed.dim() > 1 else 0

        m = lns_max(x_packed, base, dim=dim, keepdim=True)[0]
        x_sub_m = lns_sub(x_packed, m, base)

        exp_x_sub_m = lns_exp(x_sub_m, base)
        sum_exp_x_sub_m = lns_sum(exp_x_sub_m, base, dim=dim, keepdim=True)
        log_sum_exp_x_sub_m = lns_log(sum_exp_x_sub_m, base)

        log_softmax = lns_sub(x_sub_m, log_sum_exp_x_sub_m, base)

        if log_softmax.dim() == 1:
            nll = log_softmax[y]
        else:
            nll = log_softmax.gather(1, y.view(-1, 1)).squeeze(1)

        if weight is not None:
            weight = weight.to(torch.int64)
            sample_weights = weight[y]
            nll = lns_mul(nll, sample_weights, base)

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
            sample_weights = weight[y.to(torch.int64)].to(torch.int64)

        else:
            x, y, base = ctx.saved_tensors
            sample_weights = None

        dim = -1 if x.dim() > 1 else 0
        m = lns_max(x, base, dim=dim, keepdim=True)[0]

        x_sub_m = lns_sub(x, m, base)
        exp_x_sub_m = lns_exp(x_sub_m, base)
        sum_exp_x_sub_m = lns_sum(exp_x_sub_m, base, dim=dim, keepdim=True)
        log_sum_exp_x_sub_m = lns_log(sum_exp_x_sub_m, base)

        log_softmax = lns_sub(x_sub_m, log_sum_exp_x_sub_m, base)
        softmax = lns_exp(log_softmax, base)

        grad_x = softmax.clone()

        if grad_x.dim() == 1:

            if sample_weights is not None:
                grad_x = lns_mul(grad_x, sample_weights, base)
                grad_x[y] = lns_sub(grad_x[y], sample_weights, base)

            else:
                grad_x[y] = lns_sub(grad_x[y], LNS_ONE, base)

        else:
            idx = torch.arange(y.size(0))

            if sample_weights is not None:
                grad_x = lns_mul(grad_x, sample_weights.view(-1, 1), base)
                grad_x[idx, y] = lns_sub(grad_x[idx, y], sample_weights, base)

            else:
                grad_x[idx, y] = lns_sub(grad_x[idx, y], LNS_ONE, base)

        if ctx.reduction == 'mean':

            if sample_weights is not None:
                denom = lns_sum(sample_weights, base)
            else:
                denom = LNSTensor.get_internal_tensor(y.size(0), base)

            grad_x = lns_div(grad_x, denom, base)

        elif ctx.reduction == 'none':

            if grad_x.dim() == 1:
                grad_x = lns_mul(grad_x, grad_output, base)

            else:
                grad_x = lns_mul(grad_x, grad_output.view(-1, 1), base)

        else:
            grad_x = lns_mul(grad_x, grad_output, base)

        return grad_x, None, None, None, None, None, None, None

@implements(torch.nn.functional.cross_entropy, LNSCrossEntropyLossFunction.forward, key="default", default=True)
def cross_entropy(x, y, weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean', label_smoothing=0.0):

    assert isinstance(y, torch.Tensor), "y must be a torch.Tensor"

    if weight is not None:
        x, weight = format_lnstensor_operands(x, weight)

    result = LNSCrossEntropyLossFunction.apply(x, y, x.base, weight, size_average, ignore_index, reduce, reduction)

    return lnstensor(result, from_lns=True, b=x.base)
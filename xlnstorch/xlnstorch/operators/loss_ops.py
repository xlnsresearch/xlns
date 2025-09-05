import math

import torch
from xlnstorch import LNS_ZERO, LNS_ONE, LNS_NEG_ONE, format_lnstensor_operands, implements
from xlnstorch.autograd import LNSFunction

def _mse_loss(ops, x, y, reduction='mean', weight=None):
    errors = ops.sub(x, y)
    squared_errors = ops.square(errors)

    if weight is not None:
        squared_errors = ops.mul(squared_errors, weight)

    if reduction == 'none':
        return squared_errors

    elif reduction == 'sum':
        squared_error_sum = ops.sum(squared_errors)
        return squared_error_sum

    elif reduction == 'mean':
        squared_error_sum = ops.sum(squared_errors)

        if weight is not None:
            weight_sum = ops.sum(weight)
            weighted_mean = ops.div(squared_error_sum, weight_sum)
            return weighted_mean

        else:
            num_elements = x.numel()
            mean = ops.div(squared_error_sum, ops.to_lns(num_elements))
            return mean

class LNSMSELossFunction(LNSFunction):

    @staticmethod
    def forward(ops, x, y, reduction='mean', weight=None):
        return _mse_loss(ops, x, y, reduction, weight)

    @staticmethod
    def setup_context(ctx, ops, inputs, output):
        x, y, reduction, weight = inputs
        ctx.reduction = reduction
        ctx.weighted = False if weight is None else True
        if ctx.weighted:
            ctx.save_for_backward(x, y, weight)
        else:
            ctx.save_for_backward(x, y)

    @staticmethod
    def backward(ctx, ops, grad_output):

        if ctx.weighted:
            x, y, weight = ctx.saved_tensors

            grad = ops.sub(x, y)
            grad = ops.mul(grad, ops.to_lns(2.0))
            grad = ops.mul(grad, weight)

            if ctx.reduction == 'mean':
                weight_sum = ops.sum(weight)
                grad = ops.div(grad, weight_sum)

        else:
            x, y = ctx.saved_tensors

            grad = ops.sub(x, y)
            grad = ops.mul(grad, ops.to_lns(2.0))

            if ctx.reduction == 'mean':
                num_elements = x.numel()
                grad = ops.div(grad, ops.to_lns(num_elements))

        grad_x = ops.mul(grad, grad_output)
        grad_y = ops.neg(grad_x)

        return grad_x, grad_y, None, None

@implements(torch.nn.functional.mse_loss, _mse_loss, key="default", default=True)
def mse_loss(x, y, size_average=None, reduce=None, reduction='mean', weight=None):
    x, y, weight = format_lnstensor_operands(x, y, weight)
    return LNSMSELossFunction.apply(x, y, reduction, weight)

def _l1_loss(ops, x, y, reduction='mean'):
    errors = ops.sub(x, y)
    abs_errors = ops.abs(errors)

    if reduction == 'none':
        return abs_errors

    elif reduction == 'sum':
        abs_error_sum = ops.sum(abs_errors)
        return abs_error_sum

    elif reduction == 'mean':
        abs_error_sum = ops.sum(abs_errors)
        num_elements = x.numel()
        abs_error_mean = ops.div(abs_error_sum, ops.to_lns(num_elements))
        return abs_error_mean

class LNSL1LossFunction(LNSFunction):

    @staticmethod
    def forward(ops, x, y, reduction='mean'):
        return _l1_loss(ops, x, y, reduction)

    @staticmethod
    def setup_context(ctx, ops, inputs, output):
        x, y, reduction = inputs
        ctx.save_for_backward(x, y)
        ctx.reduction = reduction

    @staticmethod
    def backward(ctx, ops, grad_output):
        x, y = ctx.saved_tensors

        grad = ops.sub(x, y)
        grad = ops.sign(grad)

        if ctx.reduction == 'mean':
            num_elements = x.numel()
            grad = ops.div(grad, ops.to_lns(num_elements))

        grad_x = ops.mul(grad, grad_output)
        grad_y = ops.neg(grad_x)

        return grad_x, grad_y, None

@implements(torch.nn.functional.l1_loss, _l1_loss, key="default", default=True)
def l1_loss(x, y, size_average=None, reduce=None, reduction='mean'):
    x, y = format_lnstensor_operands(x, y)
    return LNSL1LossFunction.apply(x, y, reduction)

def _bce_loss(ops, x, y, weight=None, reduction='mean'):

    log_x = ops.log(x)
    pos_log_prob = ops.mul(y, log_x)

    x2 = ops.sub(LNS_ONE, x)
    log_x2 = ops.log(x2)
    y2 = ops.sub(LNS_ONE, y)
    neg_log_prob = ops.mul(y2, log_x2)

    loss = ops.add(pos_log_prob, neg_log_prob)
    if weight is not None:
        loss = ops.mul(loss, weight)
    loss = ops.neg(loss)

    if reduction == 'none':
        return loss

    elif reduction == 'sum':
        loss_sum = ops.sum(loss)
        return loss_sum

    elif reduction == 'mean':
        loss_sum = ops.sum(loss)

        if weight is not None:
            weight_sum = ops.sum(weight)
            weighted_mean = ops.div(loss_sum, weight_sum)
            return weighted_mean

        else:
            num_elements = x.numel()
            mean = ops.div(loss_sum, ops.to_lns(num_elements))
            return mean

class LNSBCELossFunction(LNSFunction):

    @staticmethod
    def forward(ops, x, y, weight=None, reduction='mean'):
        return _bce_loss(ops, x, y, weight, reduction)

    @staticmethod
    def setup_context(ctx, ops, inputs, output):
        x, y, weight, reduction = inputs
        ctx.reduction = reduction
        ctx.weighted = False if weight is None else True
        if ctx.weighted:
            ctx.save_for_backward(x, y, weight)
        else:
            ctx.save_for_backward(x, y)

    @staticmethod
    def backward(ctx, ops, grad_output):

        if ctx.weighted:
            x, y, weight = ctx.saved_tensors

            one_minus_x = ops.sub(LNS_ONE, x)
            one_minus_y = ops.sub(LNS_ONE, y)
            term1 = ops.div(one_minus_y, one_minus_x)
            term2 = ops.div(y, x)

            grad_x = ops.sub(term1, term2)
            grad_x = ops.mul(grad_x, weight)

            grad_y = ops.div(x, one_minus_x)
            grad_y = ops.log(grad_y)
            grad_y = ops.mul(grad_y, weight)
            grad_y = ops.neg(grad_y)

            if ctx.reduction == 'mean':
                weight_sum = ops.sum(weight)
                grad_x = ops.div(grad_x, weight_sum)
                grad_y = ops.div(grad_y, weight_sum)

        else:
            x, y = ctx.saved_tensors

            one_minus_x = ops.sub(LNS_ONE, x)
            one_minus_y = ops.sub(LNS_ONE, y)
            term1 = ops.div(one_minus_y, one_minus_x)
            term2 = ops.div(y, x)

            grad_x = ops.sub(term1, term2)
            grad_y = ops.div(x, one_minus_x)
            grad_y = ops.log(grad_y)
            grad_y = ops.neg(grad_y)

            if ctx.reduction == 'mean':
                num_elements = ops.to_lns(x.numel())
                grad_x = ops.div(grad_x, num_elements)
                grad_y = ops.div(grad_y, num_elements)

        grad_x = ops.mul(grad_x, grad_output)
        grad_y = ops.mul(grad_y, grad_output)

        return grad_x, grad_y, None, None

@implements(torch.nn.functional.binary_cross_entropy, _bce_loss, key="default", default=True)
def binary_cross_entropy(x, y, weight=None, size_average=None, reduce=None, reduction='mean'):
    x, y, weight = format_lnstensor_operands(x, y, weight)
    return LNSBCELossFunction.apply(x, y, weight, reduction)

def _bce_with_logits_loss(ops, x, y, weight=None, reduction='mean'):
    sigmoid_x = ops.sigmoid(x)
    log_sigmoid_x = ops.log(sigmoid_x)
    pos_log_prob = ops.mul(y, log_sigmoid_x)

    sigmoid_x2 = ops.sub(LNS_ONE, sigmoid_x)
    log_sigmoid_x2 = ops.log(sigmoid_x2)
    y2 = ops.sub(LNS_ONE, y)
    neg_log_prob = ops.mul(y2, log_sigmoid_x2)

    loss = ops.add(pos_log_prob, neg_log_prob)
    if weight is not None:
        loss = ops.mul(loss, weight)
    loss = ops.neg(loss)

    if reduction == 'none':
        return loss

    elif reduction == 'sum':
        loss_sum = ops.sum(loss)
        return loss_sum

    elif reduction == 'mean':
        loss_sum = ops.sum(loss)

        if weight is not None:
            weight_sum = ops.sum(weight)
            weighted_mean = ops.div(loss_sum, weight_sum)
            return weighted_mean

        else:
            num_elements = ops.to_lns(x.numel())
            mean = ops.div(loss_sum, num_elements)
            return mean

# doesn't implement pos_weight yet
class LNSBCEWithLogitsLossFunction(LNSFunction):

    @staticmethod
    def forward(ops, x, y, weight=None, reduction='mean'):
        return _bce_with_logits_loss(ops, x, y, weight, reduction)

    @staticmethod
    def setup_context(ctx, ops, inputs, output):
        x, y, weight, reduction = inputs
        ctx.reduction = reduction
        ctx.weighted = False if weight is None else True
        if ctx.weighted:
            ctx.save_for_backward(x, y, weight)
        else:
            ctx.save_for_backward(x, y)

    @staticmethod
    def backward(ctx, ops, grad_output):

        if ctx.weighted:
            x, y, weight = ctx.saved_tensors

            sigmoid_x = ops.sigmoid(x)
            grad_x = ops.sub(sigmoid_x, y)
            grad_x = ops.mul(grad_x, weight)

            grad_y = ops.mul(x, weight)
            grad_y = ops.neg(grad_y)

            if ctx.reduction == 'mean':
                weight_sum = ops.sum(weight)
                grad_x = ops.div(grad_x, weight_sum)
                grad_y = ops.div(grad_y, weight_sum)

        else:
            x, y = ctx.saved_tensors

            sigmoid_x = ops.sigmoid(x)
            grad_x = ops.sub(sigmoid_x, y)
            grad_y = ops.neg(x)

            if ctx.reduction == 'mean':
                num_elements = ops.to_lns(x.numel())
                grad_x = ops.div(grad_x, num_elements)
                grad_y = ops.div(grad_y, num_elements)

        grad_x = ops.mul(grad_x, grad_output)
        grad_y = ops.mul(grad_y, grad_output)

        return grad_x, grad_y, None, None

@implements(torch.nn.functional.binary_cross_entropy_with_logits, _bce_with_logits_loss, key="default", default=True)
def binary_cross_entropy_with_logits(x, y, weight=None, size_average=None, reduce=None, reduction='mean', pos_weight=None):
    if pos_weight is not None:
        raise NotImplementedError("pos_weight is not implemented yet.")

    x, y, weight = format_lnstensor_operands(x, y, weight)
    return LNSBCEWithLogitsLossFunction.apply(x, y, weight, reduction)

def _nll_loss(ops, x, y, weight=None, reduction='mean'):
    if x.dim() == 1:
        nll = x[y]
    else:
        nll = x.gather(1, y.view(-1, 1)).squeeze(1)

    if weight is not None:
        sample_weights = weight[y]
        nll = ops.mul(nll, sample_weights)

    loss = ops.neg(nll)

    if reduction == 'none':
        return loss
    
    elif reduction == 'sum':
        loss_sum = ops.sum(loss)
        return loss_sum
    
    elif reduction == 'mean':
        loss_sum = ops.sum(loss)

        if weight is not None:
            weight_sum = ops.sum(sample_weights)
            weighted_mean = ops.div(loss_sum, weight_sum)
            return weighted_mean

        else:
            batch_size = ops.to_lns(y.size(0))
            mean = ops.div(loss_sum, batch_size)
            return mean

# currently doesn't support ignore_index
class LNSNLLLossFunction(LNSFunction):

    @staticmethod
    def forward(ops, x, y, weight=None, reduction='mean'):
        return _nll_loss(ops, x, y, weight, reduction)

    @staticmethod
    def setup_context(ctx, ops, inputs, output):
        x, y, weight, reduction = inputs
        ctx.reduction = reduction
        ctx.weighted = True if weight is not None else False
        if ctx.weighted:
            ctx.save_for_backward(x, y, weight)
        else:
            ctx.save_for_backward(x, y)

    @staticmethod
    def backward(ctx, ops, grad_output):

        if ctx.weighted:
            x, y, weight = ctx.saved_tensors

            grad_x = ops.zeros_like(x)
            if grad_x.dim() == 1:
                grad_x[y] = ops.neg(weight[y])

            else:
                batch_size = y.size(0)
                indices = torch.arange(batch_size)
                grad_x[indices, y] = ops.neg(weight[y])

            if ctx.reduction == 'mean':
                weight_sum = ops.sum(weight)
                grad_x = ops.div(grad_x, weight_sum)

        else:
            x, y = ctx.saved_tensors

            grad_x = ops.zeros_like(x)
            if grad_x.dim() == 1:
                grad_x[y] = LNS_NEG_ONE.clone()

            else:
                batch_size = y.size(0)
                indices = torch.arange(batch_size)
                grad_x[indices, y] = LNS_NEG_ONE.clone()

            if ctx.reduction == 'mean':
                batch_size = ops.to_lns(y.size(0))
                grad_x = ops.div(grad_x, batch_size)

        if ctx.reduction == 'none':
            if grad_x.dim() == 1:
                grad_x = ops.mul(grad_x, grad_output)

            else:
                batch_size = y.size(0)
                indices = torch.arange(batch_size)
                grad_x[indices, y] = ops.mul(grad_x[indices, y], grad_output)

        else:
            grad_x = ops.mul(grad_x, grad_output)

        return grad_x, None, None, None

@implements(torch.nn.functional.nll_loss, _nll_loss, key="default", default=True)
def nll_loss(x, y, weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean'):
    assert isinstance(y, torch.Tensor), "y must be a torch.Tensor"
    if ignore_index != -100:
        raise NotImplementedError("ignore_index is not implemented yet.")

    x, weight = format_lnstensor_operands(x, weight)
    return LNSNLLLossFunction.apply(x, y, weight, reduction)

def _poisson_nll_loss(ops, x, y, eps, log_input=True, full=False, reduction='mean'):
    if log_input:
        exp_x = ops.exp(x)
        loss = ops.sub(exp_x, ops.mul(y, x))
    else:
        log_x = ops.log(ops.add(x, eps))
        loss = ops.sub(x, ops.mul(y, log_x))

    if full:
        y_clamped = torch.where(ops.gt(y, LNS_ONE), y, LNS_ONE)

        two_pi = ops.to_lns(2.0 * math.pi)
        stirling_term1 = ops.mul(y_clamped, ops.log(y_clamped))
        stirling_term3 = ops.mul(ops.log(ops.mul(two_pi, y_clamped)), ops.to_lns(0.5))
        stirling = ops.add(ops.sub(stirling_term1, y_clamped), stirling_term3)

        loss = ops.add(loss, stirling)

    if reduction == 'none':
        return loss

    elif reduction == 'sum':
        loss_sum = ops.sum(loss)
        return loss_sum

    elif reduction == 'mean':
        loss_sum = ops.sum(loss)
        num_elements = ops.to_lns(x.numel())
        mean = ops.div(loss_sum, num_elements)
        return mean

class PoissonNLLLossFunction(LNSFunction):

    @staticmethod
    def forward(ops, x, y, eps, log_input=True, full=False, reduction='mean'):
        return _poisson_nll_loss(ops, x, y, eps, log_input, full, reduction)

    @staticmethod
    def setup_context(ctx, ops, inputs, output):
        x, y, eps, log_input, full, reduction = inputs
        ctx.save_for_backward(x, y, eps)
        ctx.log_input = log_input
        ctx.full = full
        ctx.reduction = reduction

    @staticmethod
    def backward(ctx, ops, grad_output):
        x, y, eps = ctx.saved_tensors

        if ctx.log_input:
            grad_x = ops.sub(ops.exp(x), y)
            grad_y = ops.neg(x)

        else:
            grad_x = ops.div(y, ops.add(x, eps))
            grad_x = ops.sub(LNS_ONE, grad_x)
            grad_y = ops.neg(ops.log(ops.add(x, eps)))

        if ctx.full:
            stirling_grad = torch.where(ops.gt(y, LNS_ONE),
                                        ops.add(ops.log(y), ops.div(
                                            ops.to_lns(0.5), y)),
                                       LNS_ZERO)
            grad_y = ops.add(grad_y, stirling_grad)

        if ctx.reduction == 'mean':
            num_elements = ops.to_lns(x.numel())
            grad_x = ops.div(grad_x, num_elements)

        grad_x = ops.mul(grad_x, grad_output)
        grad_y = ops.mul(grad_y, grad_output)

        return grad_x, grad_y, None, None, None, None

@implements(torch.nn.functional.poisson_nll_loss, _poisson_nll_loss, key="default", default=True)
def poisson_nll_loss(x, y, log_input=True, full=False, size_average=None, eps=1e-08, reduce=None, reduction='mean'):
    x, y, eps = format_lnstensor_operands(x, y, eps)
    return PoissonNLLLossFunction.apply(x, y, eps, log_input, full, reduction)

def _hinge_embedding_loss(ops, x, y, margin, reduction='mean'):
    positive_mask = ops.eq(y, LNS_ONE)
    loss = torch.where(positive_mask, x, ops.maximum(LNS_ZERO, ops.sub(margin, x)))

    if reduction == 'none':
        return loss

    elif reduction == 'sum':
        loss_sum = ops.sum(loss)
        return loss_sum

    elif reduction == 'mean':
        loss_sum = ops.sum(loss)
        num_elements = ops.to_lns(x.numel())
        mean = ops.div(loss_sum, num_elements)
        return mean

class LNSHingeEmbeddingLossFunction(LNSFunction):

    @staticmethod
    def forward(ops, x, y, margin, reduction='mean'):
        return _hinge_embedding_loss(ops, x, y, margin, reduction)

    @staticmethod
    def setup_context(ctx, ops, inputs, output):
        x, y, margin, reduction = inputs
        ctx.save_for_backward(x, y, margin)
        ctx.reduction = reduction

    @staticmethod
    def backward(ctx, ops, grad_output):
        x, y, margin = ctx.saved_tensors

        grad_x = torch.where(ops.eq(y, LNS_ONE),
                             LNS_ONE,
                             torch.where(ops.gt(ops.sub(margin, x), LNS_ZERO),
                                         LNS_NEG_ONE, LNS_ZERO))

        if ctx.reduction == 'mean':
            num_elements = ops.to_lns(x.numel())
            grad_x = ops.div(grad_x, num_elements)

        grad_x = ops.mul(grad_x, grad_output)

        return grad_x, None, None, None

@implements(torch.nn.functional.hinge_embedding_loss, _hinge_embedding_loss, key="default", default=True)
def hinge_embedding_loss(x, y, margin=1.0, size_average=None, reduce=None, reduction='mean'):
    x, y, margin = format_lnstensor_operands(x, y, margin)
    return LNSHingeEmbeddingLossFunction.apply(x, y, margin, reduction)

def _kl_div_loss(ops, x, y, reduction='mean', log_target=False):
    if log_target:
        loss = ops.mul(ops.exp(y), ops.sub(y, x))
    else:
        loss = ops.mul(y, ops.sub(ops.log(y), x))

    if reduction == 'none':
        return loss

    elif reduction == 'sum':
        loss_sum = ops.sum(loss)
        return loss_sum

    elif reduction == 'mean':
        loss_sum = ops.sum(loss)
        num_elements = ops.to_lns(x.numel())
        mean = ops.div(loss_sum, num_elements)
        return mean

    elif reduction == 'batchmean':
        loss_sum = ops.sum(loss)
        num_elements = ops.to_lns(x.size(0))
        batch_mean = ops.div(loss_sum, num_elements)
        return batch_mean

class LNSKLDivLossFunction(LNSFunction):

    @staticmethod
    def forward(ops, x, y, reduction='mean', log_target=False):
        return _kl_div_loss(ops, x, y, reduction, log_target)

    @staticmethod
    def setup_context(ctx, ops, inputs, output):
        x, y, reduction, log_target = inputs
        ctx.save_for_backward(x, y)
        ctx.reduction = reduction
        ctx.log_target = log_target

    @staticmethod
    def backward(ctx, ops, grad_output):
        x, y = ctx.saved_tensors

        if ctx.log_target:
            exp_y = ops.exp(y)
            grad_x = ops.neg(exp_y)
            grad_y = ops.mul(exp_y, ops.add(ops.sub(y, x), LNS_ONE))
        else:
            grad_x = ops.neg(y)
            grad_y = ops.add(ops.sub(ops.log(y), x), LNS_ONE)

        if ctx.reduction == 'mean':
            num_elements = ops.to_lns(x.numel())
            grad_x = ops.div(grad_x, num_elements)
            grad_y = ops.div(grad_y, num_elements)

        elif ctx.reduction == 'batchmean':
            num_elements = ops.to_lns(x.size(0))
            grad_x = ops.div(grad_x, num_elements)
            grad_y = ops.div(grad_y, num_elements)

        grad_x = ops.mul(grad_x, grad_output)
        grad_y = ops.mul(grad_y, grad_output)

        return grad_x, grad_y, None, None

@implements(torch.nn.functional.kl_div, _kl_div_loss, key="default", default=True)
def kl_div(x, y, size_average=None, reduce=None, reduction='mean', log_target=False):
    x, y = format_lnstensor_operands(x, y)
    return LNSKLDivLossFunction.apply(x, y, reduction, log_target)

def _margin_ranking_loss(ops, x1, x2, y, margin, reduction='mean'):
    loss = ops.sub(x1, x2)
    loss = ops.mul(loss, y)
    loss = ops.sub(margin, loss)
    loss = ops.maximum(LNS_ZERO, loss)

    if reduction == 'none':
        return loss

    elif reduction == 'sum':
        loss_sum = ops.sum(loss)
        return loss_sum

    elif reduction == 'mean':
        loss_sum = ops.sum(loss)
        num_elements = ops.to_lns(x1.numel())
        mean = ops.div(loss_sum, num_elements)
        return mean

class LNSMarginRankingLossFunction(LNSFunction):

    @staticmethod
    def forward(ops, x1, x2, y, margin, reduction='mean'):
        return _margin_ranking_loss(ops, x1, x2, y, margin, reduction)

    @staticmethod
    def setup_context(ctx, ops, inputs, output):
        x1, x2, y, margin, reduction = inputs
        ctx.save_for_backward(x1, x2, y, margin)
        ctx.reduction = reduction

    @staticmethod
    def backward(ctx, ops, grad_output):
        x1, x2, y, margin = ctx.saved_tensors

        loss = ops.sub(x1, x2)
        loss = ops.mul(loss, y)
        loss = ops.sub(margin, loss)
        gt_zero_mask = ops.gt(loss, LNS_ZERO)

        grad_x1 = torch.where(gt_zero_mask, ops.neg(y), LNS_ZERO)
        grad_x2 = torch.where(gt_zero_mask, y, LNS_ZERO)
        grad_y = torch.where(gt_zero_mask, ops.sub(x2, x1), LNS_ZERO)

        if ctx.reduction == 'mean':
            num_elements = ops.to_lns(x1.numel())
            grad_x1 = ops.div(grad_x1, num_elements)
            grad_x2 = ops.div(grad_x2, num_elements)
            grad_y = ops.div(grad_y, num_elements)

        grad_x1 = ops.mul(grad_x1, grad_output)
        grad_x2 = ops.mul(grad_x2, grad_output)
        grad_y = ops.mul(grad_y, grad_output)

        return grad_x1, grad_x2, grad_y, None, None

@implements(torch.nn.functional.margin_ranking_loss, _margin_ranking_loss, key="default", default=True)
def margin_ranking_loss(x1, x2, y, margin=0.0, size_average=None, reduce=None, reduction='mean'):
    x1, x2, y, margin = format_lnstensor_operands(x1, x2, y, margin)
    return LNSMarginRankingLossFunction.apply(x1, x2, y, margin, reduction)

def _gaussian_nll_loss(ops, x, y, var, eps, full=False, reduction='mean'):
    var_eps = ops.maximum(var, eps)
    loss = ops.square(ops.sub(x, y))
    loss = ops.add(ops.log(var_eps), ops.div(loss, var_eps))

    if full:
        two_pi = ops.to_lns(2.0 * math.pi)
        loss = ops.add(loss, ops.log(two_pi))

    loss = ops.div(loss, ops.to_lns(2.0))

    if reduction == 'none':
        return loss

    elif reduction == 'sum':
        loss_sum = ops.sum(loss)
        return loss_sum

    elif reduction == 'mean':
        loss_sum = ops.sum(loss)
        num_elements = x.numel()
        mean = ops.div(loss_sum, ops.to_lns(num_elements))
        return mean

class LNSGaussianNLLLossFunction(LNSFunction):

    @staticmethod
    def forward(ops, x, y, var, eps, full=False, reduction='mean'):
        return _gaussian_nll_loss(ops, x, y, var, eps, full, reduction)

    @staticmethod
    def setup_context(ctx, ops, inputs, output):
        x, y, var, eps, _, reduction = inputs
        ctx.save_for_backward(x, y, var, eps)
        ctx.reduction = reduction

    @staticmethod
    def backward(ctx, ops, grad_output):
        x, y, var, eps = ctx.saved_tensors

        var_eps = ops.maximum(var, eps)
        grad_x = ops.div(ops.sub(x, y), var_eps)
        grad_y = ops.neg(grad_x)

        grad_var = ops.square(ops.div(ops.sub(x, y), var))
        grad_var = ops.sub(ops.reciprocal(var), grad_var)
        grad_var = ops.div(grad_var, ops.to_lns(2.0))

        if ctx.reduction == 'mean':
            num_elements = ops.to_lns(x.numel())
            grad_x = ops.div(grad_x, num_elements)
            grad_y = ops.div(grad_y, num_elements)
            grad_var = ops.div(grad_var, num_elements)

        grad_x = ops.mul(grad_x, grad_output)
        grad_y = ops.mul(grad_y, grad_output)
        grad_var = ops.mul(grad_var, grad_output)

        return grad_x, grad_y, grad_var, None, None, None

@implements(torch.nn.functional.gaussian_nll_loss, _gaussian_nll_loss, key="default", default=True)
def gaussian_nll_loss(x, y, var, full=False, eps=1e-6, reduction='mean'):
    x, y, var, eps = format_lnstensor_operands(x, y, var, eps)
    return LNSGaussianNLLLossFunction.apply(x, y, var, eps, full, reduction)

def _huber_loss(ops, x, y, delta, reduction='mean', weight=None):
    two = ops.to_lns(2.0)

    abs_diff = ops.abs(ops.sub(x, y))
    l1_term = ops.sub(abs_diff, ops.div(delta, two))
    l1_term = ops.mul(l1_term, delta)

    l2_term = ops.square(ops.sub(x, y))
    l2_term = ops.div(l2_term, two)

    loss = torch.where(ops.lt(abs_diff, delta), l2_term, l1_term)
    if weight is not None:
        loss = ops.mul(loss, weight)

    if reduction == 'none':
        return loss

    elif reduction == 'sum':
        loss_sum = ops.sum(loss)
        return loss_sum

    elif reduction == 'mean':
        loss_sum = ops.sum(loss)
        num_elements = ops.to_lns(x.numel())
        mean = ops.div(loss_sum, num_elements)
        return mean

class LNSHuberLossFunction(LNSFunction):

    @staticmethod
    def forward(ops, x, y, delta, reduction='mean', weight=None):
        return _huber_loss(x, y, delta, reduction, weight)

    @staticmethod
    def setup_context(ctx, ops, inputs, output):
        x, y, delta, reduction, weight = inputs
        ctx.reduction = reduction
        ctx.weighted = False if weight is None else True
        if ctx.weighted:
            ctx.save_for_backward(x, y, delta, weight)
        else:
            ctx.save_for_backward(x, y, delta)

    @staticmethod
    def backward(ctx, ops, grad_output):
        if ctx.weighted:
            x, y, delta, weight = ctx.saved_tensors
        else:
            x, y, delta = ctx.saved_tensors

        two = ops.to_lns(2.0)

        l2_loss_grad_x = ops.sub(x, y)
        l2_loss_grad_y = ops.neg(l2_loss_grad_x)
        l1_loss_grad_x = ops.mul(ops.sign(ops.sub(x, y)), delta)
        l1_loss_grad_y = ops.neg(l1_loss_grad_x)

        abs_diff = ops.abs(ops.sub(x, y))
        l2_mask = ops.lt(abs_diff, delta)
        grad_x = torch.where(l2_mask, l2_loss_grad_x, l1_loss_grad_x)
        grad_y = torch.where(l2_mask, l2_loss_grad_y, l1_loss_grad_y)

        if ctx.weighted:

            abs_diff = ops.abs(ops.sub(x, y))
            l1_term = ops.sub(abs_diff, ops.div(delta, two))
            l1_term = ops.mul(l1_term, delta)

            l2_term = ops.square(ops.sub(x, y))
            l2_term = ops.div(l2_term, two)

            grad_w = torch.where(ops.lt(abs_diff, delta), l2_term, l1_term)
            grad_x = ops.mul(grad_x, weight)
            grad_y = ops.mul(grad_y, weight)

        else:
            grad_w = None

        if ctx.reduction == 'mean':
            num_elements = ops.to_lns(x.numel())
            grad_x = ops.div(grad_x, num_elements)
            grad_y = ops.div(grad_y, num_elements)
            if grad_w is not None:
                grad_w = ops.div(grad_w, num_elements)

        grad_x = ops.mul(grad_x, grad_output)
        grad_y = ops.mul(grad_y, grad_output)
        if grad_w is not None:
            grad_w = ops.mul(grad_w, grad_output)

        return grad_x, grad_y, None, None, grad_w

@implements(torch.nn.functional.huber_loss, _huber_loss, key="default", default=True)
def huber_loss(x, y, delta=1.0, reduction='mean', weight=None):
    x, y, delta, weight = format_lnstensor_operands(x, y, delta, weight)
    return LNSHuberLossFunction.apply(x, y, delta, reduction, weight)

def _smooth_l1_loss(ops, x, y, beta, reduction='mean'):
    two = ops.to_lns(2.0)

    abs_diff = ops.abs(ops.sub(x, y))
    l1_term = ops.sub(abs_diff, ops.div(beta, two))

    l2_term = ops.square(ops.sub(x, y))
    l2_term = ops.div(l2_term, ops.mul(two, beta))

    loss = torch.where(ops.lt(abs_diff, beta), l2_term, l1_term)

    if reduction == 'none':
        return loss

    elif reduction == 'sum':
        loss_sum = ops.sum(loss)
        return loss_sum

    elif reduction == 'mean':
        loss_sum = ops.sum(loss)
        num_elements = ops.to_lns(x.numel())
        mean = ops.div(loss_sum, num_elements)
        return mean

class LNSSmoothL1LossFunction(LNSFunction):

    @staticmethod
    def forward(ops, x, y, beta, reduction='mean'):
        return _smooth_l1_loss(ops, x, y, beta, reduction)

    @staticmethod
    def setup_context(ctx, ops, inputs, output):
        x, y, beta, reduction = inputs
        ctx.save_for_backward(x, y, beta)
        ctx.reduction = reduction

    @staticmethod
    def backward(ctx, ops, grad_output):
        x, y, beta = ctx.saved_tensors

        l2_loss_grad_x = ops.div(ops.sub(x, y), beta)
        l2_loss_grad_y = ops.neg(l2_loss_grad_x)
        l1_loss_grad_x = ops.sign(ops.sub(x, y))
        l1_loss_grad_y = ops.neg(l1_loss_grad_x)

        abs_diff = ops.abs(ops.sub(x, y))
        l2_mask = ops.lt(abs_diff, beta)
        grad_x = torch.where(l2_mask, l2_loss_grad_x, l1_loss_grad_x)
        grad_y = torch.where(l2_mask, l2_loss_grad_y, l1_loss_grad_y)

        if ctx.reduction == 'mean':
            num_elements = ops.to_lns(x.numel())
            grad_x = ops.div(grad_x, num_elements)
            grad_y = ops.div(grad_y, num_elements)

        grad_x = ops.mul(grad_x, grad_output)
        grad_y = ops.mul(grad_y, grad_output)

        return grad_x, grad_y, None, None
    
@implements(torch.nn.functional.smooth_l1_loss, _smooth_l1_loss, key="default", default=True)
def smooth_l1_loss(x, y, size_average=None, reduce=None, reduction='mean', beta=1.0):
    x, y, beta = format_lnstensor_operands(x, y, beta)
    return LNSSmoothL1LossFunction.apply(x, y, beta, reduction)

def _cross_entropy(ops, x, y, weight=None, reduction='mean'):
    dim = -1 if x.dim() > 1 else 0

    m = ops.max(x, dim=dim, keepdim=True)[0]
    x_sub_m = ops.sub(x, m)

    exp_x_sub_m = ops.exp(x_sub_m)
    sum_exp_x_sub_m = ops.sum(exp_x_sub_m, dim=dim, keepdim=True)
    log_sum_exp_x_sub_m = ops.log(sum_exp_x_sub_m)

    log_softmax = ops.sub(x_sub_m, log_sum_exp_x_sub_m)

    if log_softmax.dim() == 1:
        nll = log_softmax[y]
    else:
        nll = log_softmax.gather(1, y.view(-1, 1)).squeeze(1)

    if weight is not None:
        sample_weights = weight[y]
        nll = ops.mul(nll, sample_weights)

    loss = ops.neg(nll)

    if reduction == 'none':
        return loss

    elif reduction == 'sum':
        loss_sum = ops.sum(loss)
        return loss_sum

    elif reduction == 'mean':
        loss_sum = ops.sum(loss)

        if weight is not None:
            weight_sum = ops.sum(sample_weights)
            weighted_mean = ops.div(loss_sum, weight_sum)
            return weighted_mean

        else:
            batch_size = ops.to_lns(y.size(0))
            mean = ops.div(loss_sum, batch_size)
            return mean

class LNSCrossEntropyLossFunction(LNSFunction):

    @staticmethod
    def forward(ops, x, y, weight=None, reduction='mean'):
        return _cross_entropy(x, y, weight, reduction)

    @staticmethod
    def setup_context(ctx, ops, inputs, output):
        x, y, weight, reduction = inputs
        ctx.reduction = reduction
        ctx.weighted = True if weight is not None else False

        if ctx.weighted:
            ctx.save_for_backward(x, y, weight)
        else:
            ctx.save_for_backward(x, y)

    @staticmethod
    def backward(ctx, ops, grad_output):
        if ctx.weighted:
            x, y, weight = ctx.saved_tensors
            sample_weights = weight[y]

        else:
            x, y = ctx.saved_tensors
            sample_weights = None

        dim = -1 if x.dim() > 1 else 0
        m = ops.max(x, dim=dim, keepdim=True)[0]

        x_sub_m = ops.sub(x, m)
        exp_x_sub_m = ops.exp(x_sub_m)
        sum_exp_x_sub_m = ops.sum(exp_x_sub_m, dim=dim, keepdim=True)
        log_sum_exp_x_sub_m = ops.log(sum_exp_x_sub_m)

        log_softmax = ops.sub(x_sub_m, log_sum_exp_x_sub_m)
        softmax = ops.exp(log_softmax)

        grad_x = softmax.clone()

        if grad_x.dim() == 1:

            if sample_weights is not None:
                grad_x = ops.mul(grad_x, sample_weights)
                grad_x[y] = ops.sub(grad_x[y], sample_weights)

            else:
                grad_x[y] = ops.sub(grad_x[y], LNS_ONE)

        else:
            idx = torch.arange(y.size(0))

            if sample_weights is not None:
                grad_x = ops.mul(grad_x, sample_weights.view(-1, 1))
                grad_x[idx, y] = ops.sub(grad_x[idx, y], sample_weights)

            else:
                grad_x[idx, y] = ops.sub(grad_x[idx, y], LNS_ONE)

        if ctx.reduction == 'mean':

            if sample_weights is not None:
                denom = ops.sum(sample_weights)
            else:
                denom = ops.to_lns(y.size(0))

            grad_x = ops.div(grad_x, denom)

        elif ctx.reduction == 'none':

            if grad_x.dim() == 1:
                grad_x = ops.mul(grad_x, grad_output)

            else:
                grad_x = ops.mul(grad_x, grad_output.view(-1, 1))

        else:
            grad_x = ops.mul(grad_x, grad_output)

        return grad_x, None, None, None

@implements(torch.nn.functional.cross_entropy, _cross_entropy, key="default", default=True)
def cross_entropy(x, y, weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean', label_smoothing=0.0):
    assert isinstance(y, torch.Tensor), "y must be a torch.Tensor"
    if ignore_index != -100:
        raise NotImplementedError("ignore_index is not implemented yet.")
    if label_smoothing != 0.0:
        raise NotImplementedError("label_smoothing is not implemented yet.")

    x, weight = format_lnstensor_operands(x, weight)
    return LNSCrossEntropyLossFunction.apply(x, y, weight, reduction)

import torch
from .. import LNS_ZERO, LNSTensor, lnstensor, format_lnstensor_operands, implements
from . import(
    lns_sub,
    lns_mul,
    lns_div,
    lns_neg,
    lns_sum,
    lns_square,
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
                mean = lns_div(squared_error_sum, LNSTensor.get_internal_tensor(num_elements, base))
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
                grad = lns_div(grad, LNSTensor.get_internal_tensor(num_elements, base))

        return grad.to(torch.float64), lns_neg(grad).to(torch.float64), None, None, None, None, None, None

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
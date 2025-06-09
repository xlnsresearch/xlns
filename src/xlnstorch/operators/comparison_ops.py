import torch
from .. import LNSTensor, format_lnstensor_operands, implements

def _lns_equal(x, y):
    return torch.equal(x, y)

@implements(torch.equal, _lns_equal, "default", default=True)
def equal(x, y):
    x, y = format_lnstensor_operands(x, y)
    return _lns_equal(x._lns, y._lns)

def _lns_eq(x, y):
    return torch.eq(x, y)

@implements(torch.eq, None, "default", default=True)
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
    x_packed, y_packed = x._lns, y._lns
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
    x_packed, y_packed = x._lns, y._lns
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
    x_packed, y_packed = x._lns, y._lns
    x_packed_log, y_packed_log = x_packed >> 1, y_packed >> 1
    x_packed_sign, y_packed_sign = x_packed & 1, y_packed & 1

    both_pos = (x_packed_sign == 0) & (y_packed_sign == 0)
    result_both_pos = torch.le(x_packed_log, y_packed_log)

    x_pos_y_neg = (x_packed_sign == 0) & (y_packed_sign == 1)
    result_x_pos = torch.ones_like(x_packed_sign, dtype=torch.bool)

    x_neg_y_pos = (x_packed_sign == 1) & (y_packed_sign == 0)
    result_x_neg = torch.zeros_like(x_packed_sign, dtype=torch.bool)

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
    x_packed, y_packed = x._lns, y._lns
    x_packed_log, y_packed_log = x_packed >> 1, y_packed >> 1
    x_packed_sign, y_packed_sign = x_packed & 1, y_packed & 1

    both_pos = (x_packed_sign == 0) & (y_packed_sign == 0)
    result_both_pos = torch.lt(x_packed_log, y_packed_log)

    x_pos_y_neg = (x_packed_sign == 0) & (y_packed_sign == 1)
    result_x_pos = torch.ones_like(x_packed_sign, dtype=torch.bool)

    x_neg_y_pos = (x_packed_sign == 1) & (y_packed_sign == 0)
    result_x_neg = torch.zeros_like(x_packed_sign, dtype=torch.bool)

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
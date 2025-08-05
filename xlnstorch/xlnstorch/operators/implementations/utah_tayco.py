"""
A torch port of the Utah-Tayco ufunc implementation for addition. See

https://github.com/xlnsresearch/xlns/blob/main/src/xlnsconf/utah_tayco_ufunc.py
"""
from enum import Enum
import torch
from xlnstorch.tensor_utils import get_precision_from_base
from xlnstorch import implements_sbdb

class RoundingMode(Enum):
    FLOOR = 'floor'
    CEIL = 'ceil'
    NEAREST = 'nearest'
    FAITHFUL = 'faithful'

def max_err(exact, approx):
    return torch.max(torch.abs(exact - approx))

def avg_abs_err(exact, approx):
    return torch.average(torch.abs(exact - approx))

def avg_err(exact, approx):
    return torch.abs(torch.average(exact - approx))

def fix_rnd(prec, mode=RoundingMode.NEAREST): 
    if mode == RoundingMode.FLOOR:
        return lambda xs: torch.floor(xs * (1 / prec)) * prec
    elif mode == RoundingMode.CEIL:
        return lambda xs: torch.ceil(xs * (1 / prec)) * prec
    elif mode ==  RoundingMode.NEAREST:
        return lambda xs: torch.round(xs * (1 / prec)) * prec
    elif mode == RoundingMode.FAITHFUL:
        # Use floor for the faithful rounding
        return lambda xs: torch.floor(xs * (1 / prec)) * prec
    else:
        raise ValueError(f'fix_rnd: unknown rounding mode {mode}')

def phi_add(xs):
    return torch.log2(1 + 2 ** xs)

def dphi_add(xs):
    return 2 ** xs / (2 ** xs + 1)

def phi_sub(xs):
    return torch.log2(1 - 2 ** xs)

def dphi_sub(xs):
    return 2 ** xs / (2 ** xs - 1)

def taylor_add(delta, xs):
    ns = torch.ceil(xs / delta) * delta
    rs = ns - xs
    return phi_add(ns) - rs * dphi_add(ns)

def taylor_add_rnd(rnd, delta, xs):
    ns = torch.ceil(xs / delta) * delta
    rs = ns - xs
    return rnd(phi_add(ns)) - rnd(rs * rnd(dphi_add(ns)))

def taylor_add_err(i, r):
    return phi_add(i - r) - phi_add(i) + r * dphi_add(i)

def taylor_add_err_rnd(rnd, i, r):
    return phi_add(i - r) - rnd(phi_add(i)) + rnd(r * rnd(dphi_add(i)))

def taylor_add_err_bound(delta):
    return taylor_add_err(0, delta)

def taylor_sub(delta, xs):
    #assume xlns will always provide xs<=0
    #if np.any(xs > -1):
    #    raise ValueError('taylor_sub: xs > -1')
    ns = torch.ceil(xs / delta) * delta
    rs = ns - xs
    return phi_sub(ns) - rs * dphi_sub(ns)

def taylor_sub_rnd(rnd, delta, xs):
    #assume xlns will always provide xs<=0
    #if np.any(xs > -1):
    #    raise ValueError('taylor_sub_rnd: xs > -1')
    ns = torch.ceil(xs / delta) * delta
    rs = ns - xs
    return rnd(phi_sub(ns)) - rnd(rs * rnd(dphi_sub(ns)))

def taylor_sub_err(i, r):
    return -phi_sub(i - r) + phi_sub(i) - r * dphi_sub(i)

def taylor_sub_err_rnd(rnd, i, r):
    return phi_sub(i - r) - rnd(phi_sub(i)) + rnd(r * rnd(dphi_sub(i)))

def taylor_sub_err_bound(delta):
    return taylor_sub_err(-1, delta)

def q_add(delta, i, r):
    return taylor_add_err(i, r) / taylor_add_err(i, delta)

def q_add_lo(delta, r):
    return q_add(delta, 0, r)

def q_add_hi(delta, r):
    return (2 ** -r + r * torch.log(2) - 1) / (2 ** -delta + delta * torch.log(2) - 1)

def r_add_opt(delta):
    x = 2 ** delta
    return torch.log2(x * (2 * torch.log(x + 1) - torch.log(x) - 2 * torch.log(2)) / (-2 * x * (torch.log(x + 1) - torch.log(x) - torch.log(2)) - x + 1))

def q_add_range_bound(delta):
    r = r_add_opt(delta)
    return q_add_hi(delta, r) - q_add_lo(delta, r)

def q_add_approx_bound(delta, delta_p):
    return 1 - q_add_lo(delta, delta - delta_p)

def ec_add_rnd(rnd, delta, delta_p, c, xs):
    ns = torch.ceil(xs / delta) * delta
    rs = ns - xs
    ec = rnd(rnd(taylor_add_err(ns, delta)) * rnd(q_add(delta, c, torch.floor(rs / delta_p) * delta_p)))
    return rnd(phi_add(ns)) - rnd(rs * rnd(dphi_add(ns))) + ec

def ind(delta, xs):
    return (torch.ceil(xs / delta) - 1) * delta

def rem(delta, xs):
    return ind(delta, xs) - xs

def kval(delta, xs):
    return xs - phi_sub(ind(delta, xs)) + phi_sub(rem(delta, xs))

def k_rnd(rnd, delta, xs):
    return xs - rnd(phi_sub(ind(delta, xs))) + rnd(phi_sub(rem(delta, xs)))

def cotrans2(delta, da, xs):
    return phi_sub(ind(da, xs)) + taylor_sub(delta, kval(da, xs))

def cotrans2_rnd(rnd, delta, da, xs):
    return rnd(phi_sub(ind(da, xs))) + taylor_sub_rnd(rnd, delta, k_rnd(rnd, da, xs))

def cotrans3(delta, da, db, xs):
    return phi_sub(ind(db, xs)) + taylor_sub(delta, kval(db, xs))

def cotrans3_rnd(rnd, delta, da, db, xs):
    rab = rem(db, xs)
    res = torch.zeros(xs.shape, dtype=torch.float64)
    special = rab >= -da
    incl = rab < -da
    rab, xs, ys = rab[incl], xs[incl], xs[special]
    rb = ind(da, rab)
    k1 = k_rnd(rnd, da, rab)
    k2 = xs + rnd(phi_sub(rb)) + taylor_sub_rnd(rnd, delta, k1) - rnd(phi_sub(ind(db, xs)))
    res[incl] = rnd(phi_sub(ind(db, xs))) + taylor_sub_rnd(rnd, delta, k2)
    res[special] = cotrans2_rnd(rnd, delta, db, ys)
    return res

@implements_sbdb("utah_tayco")
def sbdb_ufunc_utah_tayco(z, s, base):
    """
    See https://ieeexplore.ieee.org/document/11038317 for the original
    paper describing this implementation.
    """
    precision = get_precision_from_base(base)
    if precision is None:
        raise ValueError(f"sbdb_utah_tayco only allows power-of-two bases, which {base.item()} is not")

    precision = torch.tensor(precision, dtype=torch.float64)
    eps = 2**(-precision)
    rnd = fix_rnd(eps,RoundingMode.FLOOR)
    N = torch.ceil((precision - 3) / 2)

    delta = 2**(-N)
    delta_a = 2**(-torch.floor((precision + 2 * N) / 3))
    delta_b = 2**(-torch.floor((2 * precision + N) / 3))

    z = torch.minimum(-s,z)
    return 2 * torch.floor(
        torch.where(s==0,
                    taylor_add_rnd(rnd, delta, z * eps),
                    torch.where(z > -1 / eps,
                                cotrans3_rnd(rnd,delta, delta_a, delta_b, z*eps),
                                taylor_sub_rnd(rnd,delta,z*eps))) / eps
                          ).to(torch.float64)
 
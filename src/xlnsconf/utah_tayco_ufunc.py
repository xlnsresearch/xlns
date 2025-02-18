"""Taylor interp/Coleman 3-table cotran; mirrors UnivUtah's formal Lean proofs by Nguyen Son, Alexey Solovyev""" 

import numpy as np
from enum import Enum
import xlns as xl

# taken from definitions.py at https://github.com/monadius/notebooks/tree/main/python/lns
# modified to remove typing and work on python 3.8

# Enumeration with rounding modes

class RoundingMode(Enum):
    FLOOR = 'floor'
    CEIL = 'ceil'
    NEAREST = 'nearest'
    FAITHFUL = 'faithful'

def max_err(exact, approx):
    return np.max(np.abs(exact - approx))

def avg_abs_err(exact, approx):
    return np.average(np.abs(exact - approx))

def avg_err(exact, approx):
    return np.abs(np.average(exact - approx))

def fix_rnd(prec, mode=RoundingMode.NEAREST): 
    if   mode == RoundingMode.FLOOR:
            return lambda xs: np.floor(xs * (1 / prec)) * prec
    elif mode == RoundingMode.CEIL:
            return lambda xs: np.ceil(xs * (1 / prec)) * prec
    elif mode ==  RoundingMode.NEAREST:
            return lambda xs: np.round(xs * (1 / prec)) * prec
    elif mode == RoundingMode.FAITHFUL:
            # Use floor for the faithful rounding
            return lambda xs: np.floor(xs * (1 / prec)) * prec
    else:
            raise ValueError(f'fix_rnd: unknown rounding mode {mode}')

# Φp and Φm and their derivatives

def phi_add(xs):
    return np.log2(1 + 2 ** xs)

def dphi_add(xs):
    return 2 ** xs / (2 ** xs + 1)

def phi_sub(xs):
    return np.log2(1 - 2 ** xs)

def dphi_sub(xs):
    return 2 ** xs / (2 ** xs - 1)

# First-order taylor approximations

def taylor_add(delta, xs):
    #assume xlns will always provide xs<=0
    #if np.any(xs > 0):
    #    raise ValueError('taylor_add: xs > 0')
    ns = np.ceil(xs / delta) * delta
    rs = ns - xs
    return phi_add(ns) - rs * dphi_add(ns)

# ΦTp_fix
def taylor_add_rnd(rnd, delta, xs):
    #assume xlns will always provide xs<=0
    #if np.any(xs > 0):
    #    raise ValueError('taylor_add_rnd: xs > 0')
    ns = np.ceil(xs / delta) * delta
    rs = ns - xs
    return rnd(phi_add(ns)) - rnd(rs * rnd(dphi_add(ns)))

# Ep
def taylor_add_err(i, r):
    return phi_add(i - r) - phi_add(i) + r * dphi_add(i)

# Ep_fix
def taylor_add_err_rnd(rnd, i, r):
    return phi_add(i - r) - rnd(phi_add(i)) + rnd(r * rnd(dphi_add(i)))

def taylor_add_err_bound(delta):
    return taylor_add_err(0, delta)

def taylor_sub(delta, xs):
    #assume xlns will always provide xs<=0
    #if np.any(xs > -1):
    #    raise ValueError('taylor_sub: xs > -1')
    ns = np.ceil(xs / delta) * delta
    rs = ns - xs
    return phi_sub(ns) - rs * dphi_sub(ns)

# ΦTm_fix
def taylor_sub_rnd(rnd, delta, xs):
    #assume xlns will always provide xs<=0
    #if np.any(xs > -1):
    #    raise ValueError('taylor_sub_rnd: xs > -1')
    ns = np.ceil(xs / delta) * delta
    rs = ns - xs
    return rnd(phi_sub(ns)) - rnd(rs * rnd(dphi_sub(ns)))

# Em
def taylor_sub_err(i, r):
    return -phi_sub(i - r) + phi_sub(i) - r * dphi_sub(i)

# Em_fix
def taylor_sub_err_rnd(rnd, i, r):
    return phi_sub(i - r) - rnd(phi_sub(i)) + rnd(r * rnd(dphi_sub(i)))

def taylor_sub_err_bound(delta):
    return taylor_sub_err(-1, delta)

# Error-correction techniques

# Qp
def q_add(delta, i, r):
    return taylor_add_err(i, r) / taylor_add_err(i, delta)

# Qp_lo
def q_add_lo(delta, r):
    return q_add(delta, 0, r)

# Qp_hi
def q_add_hi(delta, r):
    return (2 ** -r + r * np.log(2) - 1) / (2 ** -delta + delta * np.log(2) - 1)

# Rp_opt
def r_add_opt(delta):
    x = 2 ** delta
    return np.log2(x * (2 * np.log(x + 1) - np.log(x) - 2 * np.log(2)) / (-2 * x * (np.log(x + 1) - np.log(x) - np.log(2)) - x + 1))

# QRp
def q_add_range_bound(delta):
    r = r_add_opt(delta)
    return q_add_hi(delta, r) - q_add_lo(delta, r)

# QIp
def q_add_approx_bound(delta, delta_p):
    return 1 - q_add_lo(delta, delta - delta_p)

# ΦECp_fix
def ec_add_rnd(rnd, delta, delta_p, c, xs):
    #assume xlns will always provide xs<=0
    #if np.any(xs > 0):
    #    raise ValueError('ec_add_rnd: xs > 0')
    ns = np.ceil(xs / delta) * delta
    rs = ns - xs
    ec = rnd(rnd(taylor_add_err(ns, delta)) * rnd(q_add(delta, c, np.floor(rs / delta_p) * delta_p)))
    return rnd(phi_add(ns)) - rnd(rs * rnd(dphi_add(ns))) + ec

# Co-transformations

def ind(delta, xs):
    return (np.ceil(xs / delta) - 1) * delta

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
    if isinstance(xs, float):
        xs = np.reshape(xs, (1,))
    rab = rem(db, xs)
    #res = np.zeros(len(xs))
    #for xlns, this should allow arbitrary shape input xs
    res = np.zeros(xs.shape)
    #purpose of special and incl unclear to me
    special = rab >= -da
    incl = rab < -da
    rab, xs, ys = rab[incl], xs[incl], xs[special]
    rb = ind(da, rab)
    k1 = k_rnd(rnd, da, rab)
    k2 = xs + rnd(phi_sub(rb)) + taylor_sub_rnd(rnd, delta, k1) - rnd(phi_sub(ind(db, xs)))
    res[incl] = rnd(phi_sub(ind(db, xs))) + taylor_sub_rnd(rnd, delta, k2)
    res[special] = cotrans2_rnd(rnd, delta, db, ys)
    return res


def sbdb_ufunc_utah_tayco(z,s, B=None,F=None):
    """UnivUtah ufunc for linear Taylor interp, 3 table Coleman cotran; doesn't work for xlnsb,xlnsnpb (B**2**F must==2)""" 
    if F==None:
        if B != None:
            raise ValueError(f'sbdb_utah_tayco cannot work with non-power-of-two base')
        F = xl.xlnsF
    if B == None:
        B = xl.xlnsB
    if (B**(2**F-1) > 2.0) or (B**(2**F+1) < 2.0):
            raise ValueError(f'sbdb_utah_tayco only allows power-of-two bases, which {B} is not')
    eps = 2**(-F)
    rnd = fix_rnd(eps,RoundingMode.FLOOR)
    N = np.ceil((F-3)/2)
    delta = 2**(-N)
    delta_a = 2**(-np.floor((  F + 2*N)/3))
    delta_b = 2**(-np.floor((2*F +   N)/3))
    z = np.minimum(-s,z) #like sbdb_ideal, this forces z==0 to be z=-1 for s=-1, otherwise no effect
    #print('delta='+str(delta))
    #print('delta_a='+str(delta_a)+' '+str(2**-14))
    #print('delta_b='+str(delta_b)+' '+str(2**-18))
    res = 2*np.int64(np.where(s==0, 
                               taylor_add_rnd(rnd,delta,z*eps),
                               np.where(z>-1/eps, cotrans3_rnd(rnd,delta, delta_a, delta_b, z*eps),  
                                                  taylor_sub_rnd(rnd,delta,z*eps)))/eps)
    #ideal = xl.sbdb_ufunc_ideal(z,s,B=B,F=F)
    #if abs(ideal-res).max()>65536:
    #    print(abs(ideal-res).max())
    #    print((z.reshape(z.size))[abs(ideal-res).argmax()])
    #    print((s.reshape(s.size))[abs(ideal-res).argmax()])
    #    print("bad utah")

    return res 

xl.sbdb_ufunc = sbdb_ufunc_utah_tayco



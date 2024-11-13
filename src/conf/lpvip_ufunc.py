"""several versions LPVIP (Arnold, PATMOS 2004) using Mitchell's method as Gaussian Log"""


import xlns as xl
import numpy as np

def mitch_ufunc(z,F):
  return (((1<<F)+(z&((1<<F)-1)))>>(-(z>>F)))
 
def dbcor_ufunc(z,F):
   return np.where(z < 1<<(F-4), 4<<F,
   np.where(z < 1<<(F-3), 3<<F,
   np.where(z < 1<<(F-2), 2<<F,
   np.where(z < 1<<(F-1), 1<<F, 0))))

def sbdb_ufunc_lpvip(z,s,B=None,F=None):
  if F == None:
    return np.where(s,-2,2)*(mitch_ufunc(z+(s<<xl.xlnsF),xl.xlnsF)+s*dbcor_ufunc(-z,xl.xlnsF))
  else:
    return np.where(s,-2,2)*(mitch_ufunc(z+(s<<F),F)+s*dbcor_ufunc(-z,F))

def sbdb_ufunc_lpvipAddonly(z,s,B=None,F=None):
  if F == None:
    return np.where(s,xl.sbdb_ufunc_ideal(z,s,B=B,F=F),2*mitch_ufunc(z+(s<<xl.xlnsF),xl.xlnsF))
  else:
    return np.where(s,xl.sbdb_ufunc_ideal(z,s,B=B,F=F),2*mitch_ufunc(z+(s<<F),F))

def sb_ufunc_premit(zi,F):   #was called premitchnpi(zi):
  postcond = np.where(zi <= -(3<<F), 0,
             np.where(zi >= -(3<<(F-2)), -1, +1))
  z = ((zi<<3) + (zi&0xfff^0xfff) + 16)>>3
  return np.where(zi==0,1<<F, (((1<<F)+(z&((1<<F)-1)))>>(-(z>>F)))+postcond )<<1

def db_ufunc_premit(z,F):
  precond = np.where(z < -(2<<F),
                    5<<(F-3),                   #0.625
                    (z >> 2) + (9 << (F-3))) #.25*zr + 9/8
  #return -2*(mitch_ufunc(z+precond,F)+lpvip_ufunc.dbcor_ufunc(-z,F))
  return np.where(-z >= 1<<F, -2*mitch_ufunc(z+precond,F),
                              xl.sbdb_ufunc_ideal(z,1)) #use ideal for singularity

def sbdb_ufunc_premitAddidealSub(z,s,B=None,F=None):
  if F==None:
    return np.where(s,xl.sbdb_ufunc_ideal(z,s),sb_ufunc_premit(z,xl.xlnsF))
  else:
    return np.where(s,xl.sbdb_ufunc_ideal(z,s,B=B,F=F),sb_ufunc_premit(z,F))

def sbdb_ufunc_premitAddlpvipSub(z,s,B=None,F=None):
  if F==None:
    return np.where(s,sbdb_ufunc_lpvip(z,s),sb_ufunc_premit(z,xl.xlnsF))
  else:
    return np.where(s,sbdb_ufunc_lpvip(z,s,B=B,F=F),sb_ufunc_premit(z,F))

def sbdb_ufunc_premitAddpremitSub(z,s,B=None,F=None):
  if F==None:
    return np.where(s,db_ufunc_premit(z,xl.xlnsF),sb_ufunc_premit(z,xl.xlnsF))
  else:
    return np.where(s,db_ufunc_premit(z,F),sb_ufunc_premit(z,F))


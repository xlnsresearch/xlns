'''
faithful test of user-configuration ufunc against ideal LNS 

uses xlnsnpb to trigger F-bit sbdb_ufunc calls to ufunc passed to it
'''

import math
import xlns as xl
import numpy as np
import xlnsconf.interp_cotran_ufunc as interp_cotran_ufunc

def faithful_sb_ufunc(f, ufunc):
    x = xl.xlnsnpb(np.multiply.accumulate(np.array((f*2**f)*[xl.xlns(2**2**-f)])),2**2**-f)
    truncsb = xl.sbdb_ufunc_trunc((1/x).nd//2,0,F=f,B=2**2**-f)//2
    truncdb = xl.sbdb_ufunc_trunc((1/x).nd//2,1,F=f,B=2**2**-f)//2
    xl.sbdb_ufunc =  ufunc
    testufuncsb = (1/x+1).nd//2
    testufuncdb = (1-1/x).nd//2
    xl.sbdb_ufunc =  xl.sbdb_ufunc_ideal
    return (((testufuncsb==truncsb).sum() + (testufuncsb==truncsb+1).sum())/len(x),
            ((testufuncdb==truncdb).sum() + (testufuncdb==truncdb+1).sum())/len(x))

def test_faithful_sb_interp():
for f in [7,9,11,13]:
#for f in [15]:
#for f in [17]:
  print("f="+str(f))
  res = faithful_sb_ufunc(f,xl.sbdb_ufunc_ideal)
  print(str(res)+" ideal sb db")
  res = faithful_sb_ufunc(f,interp_cotran_ufunc.sbdb_ufunc_interpsbcotrdb)
  print(str(res)+" interp sb db")
  res = faithful_sb_ufunc(f,interp_cotran_ufunc.sbdb_ufunc_interpsbcotrdb_g2)
  print(str(res)+" interp_g2 sb db")
  res = faithful_sb_ufunc(f,interp_cotran_ufunc.sbdb_ufunc_interpsbcotrdb_g4)
  print(str(res)+" interp_g4 sb db")
  res = faithful_sb_ufunc(f,interp_cotran_ufunc.sbdb_ufunc_interpsbcotrdb_g6)
  print(str(res)+" interp_g6 sb db")
  res = faithful_sb_ufunc(f,interp_cotran_ufunc.sbdb_ufunc_interpsbcotrdb_g2_faith)
  print(str(res)+" interp_g2 sb db faith")
  res = faithful_sb_ufunc(f,interp_cotran_ufunc.sbdb_ufunc_interpsbcotrdb_g4_faith)
  print(str(res)+" interp_g4 sb db faith")
  res = faithful_sb_ufunc(f,interp_cotran_ufunc.sbdb_ufunc_interpsbcotrdb_g6_faith)
  print(str(res)+" interp_g6 sb db faith")

test_faithful_sb_interp()

#>>> f=8
#>>> x = xl.xlnsnpb(np.multiply.accumulate(np.array((f*2**f)*[xl.xlns(2**2**-f)])),2**2**-f)
#>>> trunc=xl.sbdb_ufunc_trunc((1/x).nd//2,0,F=8,B=2**2**-8)//2
#>>> ideal=(1/x+1).nd//2
#>>> (ideal==trunc).sum()
#1003
#>>> (ideal==trunc+1).sum()
#1045



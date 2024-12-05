'''
exhaustive test of user-configuration ufunc against 64-bit FP

uses xlnsv to trigger f-bit sbdb_ufunc calls to ufunc passed to it
'''

import math
import xlns as xl
import numpy as np
import xlnsconf.interp_cotran_ufunc as interp_cotran_ufunc


def testufunc(f,ufunc):
    xl.sbdb_ufunc =  ufunc
    x=np.multiply.accumulate(np.array(xl.xlnscopy((f*2**f)*[2**2**-f], xl.xlnsv, f)))
    xf=np.multiply.accumulate(np.array(((f*2**f)*[2**2**-f])))
    relerr_sb = (((xf+1)-xl.float64(x+1))/(xf+1))
    relerr_db = (((xf-1)-xl.float64(x-1))/(xf-1))
    for i in range(0,len(relerr_db)):
      if relerr_db[i]>.1:
        print(str(i)+" "+str(relerr_db[i]))
    xl.sbdb_ufunc =  xl.sbdb_ufunc_ideal
    return (relerr_sb.min(), relerr_sb.max(), relerr_db.min(), relerr_db.max())

#f=8
#res = testufunc(f,xlnsconf.interp_cotran_ufunc.sbdb_ufunc_interpsb)
#res = testufunc(f,interp_cotran_ufunc.sbdb_ufunc_interpsb)
#print(res)
#-0.004508454874271358 0.003607134478010154
#print(math.log(abs(res[0]),2))
#print(math.log(abs(res[1]),2))
#print(math.log(abs(res[2]),2))
#print(math.log(abs(res[3]),2))
#-7.793151203102941
#-8.114931075248318
#res = testufunc(f,xlnsconf.interp_cotran_ufunc.sbdb_ufunc_cotrdb)
#res = testufunc(f,interp_cotran_ufunc.sbdb_ufunc_cotrdb)
#print(res)

def testinterpcotran():
 for f in [8,9,11]:
 #for f in [8,9,11,13,15]:
  print("f="+str(f))
  res = testufunc(f,interp_cotran_ufunc.sbdb_ufunc_interpsb)
  print(str(res)+" interp sb only")
  res = testufunc(f,interp_cotran_ufunc.sbdb_ufunc_cotrdb)
  print(str(res)+" cotran db only")
  res = testufunc(f,interp_cotran_ufunc.sbdb_ufunc_interpsbcotrdb)
  print(str(res)+" interp sb+cotran db")

testinterpcotran()

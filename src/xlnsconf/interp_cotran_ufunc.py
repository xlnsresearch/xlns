import xlns as xl
import numpy as np

def sbdb_ufunc_interpsb(z,s,B=None,F=None):
  if B == None:
    B = xl.xlnsB
  if F == None:
    F = xl.xlnsF
  N = (F-5)//2
  J = F - N
  #print("N="+str(N))
  #print("J="+str(J))
  zhmask = -(1<<J)
  zlmask = -1 - zhmask
  #print("zhmask="+str(bin(zhmask)))
  #print("zlmask="+str(bin(zlmask)))
  #print(zlmask)
  zh = z & zhmask
  zl = z & zlmask
  #print("zh="+str(bin(zh)))
  #print("zl="+str(bin(zl)))
  szh = xl.sbdb_ufunc_ideal(zh,0,B=B,F=F)//2
  szhp1 = xl.sbdb_ufunc_ideal(zh-zhmask,0,B=B,F=F)//2
  #print("szh="+str(bin(szh)))
  #print("szhp1="+str(bin(szhp1)))
  res = ((szhp1 - szh)* zl)>>J
  #print("res="+str(bin(res)))
  res += szh
  return np.where(s,xl.sbdb_ufunc_ideal(z,s,B=B,F=F), 2*res)


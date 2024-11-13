""" 
implements Dally et al. bin-based LNS summation 

this code for academic non-commercial use only
may be covered by U.S. Patent Application Publication US20210056446A1
to use:
   import xlns as xl
   import xlnsconf.dally
   xl.xlnsnp.sum = xlnsconf.dally.sum_xlnsnp
   xl.xlnsnpv.sum = xlnsconf.dally.sum_xlnsnpv
to restore default:
   xl.xlnsnp.sum = xl.xlnsnp.sum_default
   xl.xlnsnpv.sum = xl.xlnsnpv.sum_default

an example:
>>> import xlns as xl
>>> import xlnsconf.dally
>>> x=xl.xlnsnpv([1,2,3,4.],8)
>>> x.sum()
xlnsnpv(10.015890222930839 f=8)
>>> xl.xlnsnpv.sum = xlnscoonf.dally.sum_xlnsnpv
>>> x.sum()
xlnsnpv(9.988807817513946 f=8)
>>> x=xl.xlnsnpv([1,2,3,4.,5,6,7,11],8)
>>> x.sum()
xlnsnpv(38.99335135447898 f=8)
>>> xl.xlnsnpv.sum = xl.xlnsnpv.sum_default
>>> x.sum()
xlnsnpv(38.99335135447898 f=8)
"""


import numpy as np
import xlns as xl

print("this code for academic non-commercial use only")
print("may be covered by U.S. Patent Application Publication US20210056446A1")

def sum_xlnsnp(self,axis=None):
  if axis == None:
    #print("case axis=None")
    a = 1.0*xl.xlnsnp.ravel(self)
  elif axis == 1:
    #print("case axis=1")
    a = 1.0*xl.xlnsnp.transpose(self)
  else:
    #print("case axis=0")
    a = 1.0*self
  if len(a)==1:
    return a
  mm = 2**(xl.xlnsF+1)-1
  binweight = 2**xl.xlnsF*[1,-1]*xl.xlnsB**(np.arange(0,mm+1)//2)
  if len(a.shape())>1:
    binweight = np.array([binweight]).T
  possmat=(np.reshape(xl.xlnsnp.size(a)*[range(0,mm+1)],
           (xl.xlnsnp.shape(a)[::-1]+(mm+1,))).T)
  binmask = ((a.nd&mm)==possmat)
  vbin = binmask<<(a.nd>>(xl.xlnsF+1))
  bincount = vbin.sum(axis=1)
  return xl.xlnsnp((bincount*binweight).sum(axis=0))

def sum_xlnsnpv(self,axis=None):
  if axis == None:
    #print("case axis=None")
    a = 1.0*xl.xlnsnpv.ravel(self)
  elif axis == 1:
    #print("case axis=1")
    a = 1.0*xl.xlnsnpv.transpose(self)
  else:
    #print("case axis=0")
    a = 1.0*self
  if len(a)==1:
    return a
  shift = xl.xlnsF-self.f
  mm = 2**(self.f+1)-1
  binweight = 2**self.f*[1,-1]*(2**2**(-self.f))**(np.arange(0,mm+1)//2)
  if len(a.shape())>1:
    binweight = np.array([binweight]).T
  possmat=(np.reshape(xl.xlnsnpv.size(a)*[range(0,mm+1)],
           (xl.xlnsnpv.shape(a)[::-1]+(mm+1,))).T)
  binmask = (((a.nd>>shift)&mm)==possmat)
  vbin = binmask<<(a.nd>>(xl.xlnsF+1))
  bincount = vbin.sum(axis=1)
  return xl.xlnsnpv((bincount*binweight).sum(axis=0),self.f)


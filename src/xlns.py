"""
 for eXperimentation with Logarithmic Number System  

 classes offer:
  scalar (xlns,xlnsr,xlnsv,xlnsb and xlnsud) 
  array (xlnsnp, xlnsnpr, xlnsnpv and xlnsnpb)
  non-redundant (xlns,xlnsv,xlnsb,xlnsud,xlnsnp,xlnsnpv and xlnsnpb)
  redundant (xlnsr, xlnsnpr)
  global precision with xlnssetF (xlns, xlnsr, xlnsud, xlnsnp, xlnsnpr)
  per instance precision (xlnsv, xlnsb, xlnsnpv, xlnsnpb)
  non-integer precision ignore xlnsF (xlnsb, xlnsnpb)  
"""
import math
import numpy as np

def xlnssetovf(mn,mx):
  """set min and max overflow values"""
  global ovfmin,ovfmax
  ovfmin = mn
  ovfmax = mx  

def xlnscalcB():
  global xlnsB,xlnsF
  xlnsB = 2.0**(2**(-xlnsF))
global xlns1stCons 
def xlnssetF(newF):
  """
  set global xlnsF and corresponding xlnsB = 2**2**(-xlnsF)

  all logarithms internally use the global base, xlnsB, or its local equivalent, B
  this function calculates this given an integer F so that using the internal base
  is isomorphic to a base-two logarithm with F bits of fractional precision
  the default is F=23
  """
  global xlnsF
  xlnsF = newF
  xlnscalcB()
  if not xlns1stCons:
      print("warning changing F after constructor(s) already called")

#assume defaults similar to IEEE 754 FP
xlns1stCons = True
xlnssetF(23)
xlnssetovf(2.938735877055719e-39 ,3.402823669209385e+38)

def sbdb_ufunc_ideal(z,s,B=None,F=None):
  """default ufunc for ideal Gaussian log for all add,sub; may substitute user supplied sbdb_ufunc""" 
  global xlnsB
  #need to deal with z=0 s=1
  if B==None:
    return 2*np.int64(np.round(np.log(np.abs(1.0 - 2.0*s + xlnsB**np.minimum(-s,z)))/math.log(xlnsB)))
  else:
    return 2*np.int64(np.round(np.log(np.abs(1.0 - 2.0*s + B**np.minimum(-s,z)))/math.log(B)))
   
sbdb_ufunc = sbdb_ufunc_ideal

def xlnsapplyfunc(x,func):
  if isinstance(x,xlnsud): 
    return(xlnsud(func(float(x))))
  elif isinstance(x,xlns): 
    return(xlns(func(float(x))))
  elif isinstance(x,xlnsr):
    return(xlnsr(func(float(x))))
  elif isinstance(x,xlnsv):
    return(xlnsv(func(float(x)),x.F))
  elif isinstance(x,xlnsb):
    return(xlnsb(func(float(x)),x.B))
  elif isinstance(x,xlnsnpv): 
    return(xlnsnpv(func(np.float64(x.xlns())),x.f))
  elif isinstance(x,xlnsnpb): 
    return(xlnsnpb(func(np.float64(x.xlns())),x.B))
  elif isinstance(x,xlnsnp): #incl xlnsnpv
    return(xlnsnp(func(np.float64(x.xlns()))))
  elif isinstance(x,xlnsnpr):
    return(xlnsnpr(func(np.float64(x.xlns()))))
  else:
    return func(x)

def xlnsarctan2(x,y):
  if isinstance(x,xlnsud): 
    return(xlnsud(np.arctan2(float(x),float(y))))
  elif isinstance(x,xlns): 
    return(xlns(np.arctan2(float(x),float(y))))
  elif isinstance(x,xlnsr):
    return(xlnsr(np.arctan2(float(x),float(y))))
  elif isinstance(x,xlnsv):
    return(xlnsv(np.arctan2(float(x),float(y)),x.F))
  elif isinstance(x,xlnsb):
    return(xlnsb(np.arctan2(float(x),float(y)),x.B))
  #above scalar ops will fail if y is not also scalar (float() not valid)
  #below array ops will fail if y is not also array (need .xlns() method)
  elif isinstance(x,xlnsnpb): 
    return(xlnsnpb(np.arctan2(np.float64(x.xlns()),np.float64(y.xlns())),x.B))
  elif isinstance(x,xlnsnpv): 
    return(xlnsnpv(np.arctan2(np.float64(x.xlns()),np.float64(y.xlns())),x.f))
  elif isinstance(x,xlnsnp): #incl xlnsnpv
    return(xlnsnp(np.arctan2(np.float64(x.xlns()),np.float64(y.xlns()))))
  elif isinstance(x,xlnsnpr):
    return(xlnsnpr(np.arctan2(np.float64(x.xlns()),np.float64(y.xlns()))))
  else:
    return np.arctan2(x,y)

arctan2 = lambda x,y: xlnsarctan2(x,y)
exp = lambda x: xlnsapplyfunc(x,np.exp)
log = lambda x: xlnsapplyfunc(x,np.log)
sin = lambda x: xlnsapplyfunc(x,np.sin)
cos = lambda x: xlnsapplyfunc(x,np.cos)
tan = lambda x: xlnsapplyfunc(x,np.tan)
sinh = lambda x: xlnsapplyfunc(x,np.sinh)
cosh = lambda x: xlnsapplyfunc(x,np.cosh)
tanh = lambda x: xlnsapplyfunc(x,np.tanh)
arcsin = lambda x: xlnsapplyfunc(x,np.arcsin)
arccos = lambda x: xlnsapplyfunc(x,np.arccos)
arctan = lambda x: xlnsapplyfunc(x,np.arctan)
arcsinh = lambda x: xlnsapplyfunc(x,np.arcsinh)
arccosh = lambda x: xlnsapplyfunc(x,np.arccosh)
arctanh = lambda x: xlnsapplyfunc(x,np.arctanh)

def sqrt(x):
  """square root means x**.5"""
  return x**.5

def sum(x,axis=None):
  """xl.sum(x) means x.sum()"""
  if axis==None:
   return x.sum()
  else:
   return x.sum(axis=axis)

def float64(x):
  """convert to floating point; like np.float64, but works with all classes"""
  if isinstance(x,xlnsnp) or isinstance(x,xlnsnpv) or isinstance(x,xlnsnpb) or isinstance(x,xlnsnpr):
   return np.float64(x.xlns())
  else:
   return np.float64(x)

def xlnsapplynpfunc(x,xlnsnp_func,xlnsnpv_func,xlnsnpb_func,xlnsnpr_func,np_func):
  #print(type(x))
  if isinstance(x,xlnsnpv): 
    return xlnsnpv_func(x)
  elif isinstance(x,xlnsnp): #incl xlnsnpv
    return xlnsnp_func(x)
  elif isinstance(x,xlnsnpb): 
    return xlnsnpb_func(x)
  elif isinstance(x,xlnsnpr):
    return xlnsnpr_func(x)
  else:
    return np_func(x)


transpose = lambda x: xlnsapplynpfunc(x,xlnsnp.transpose,xlnsnpv.transpose,xlnsnpb.transpose,xlnsnpr.transpose,np.transpose)
ravel = lambda x: xlnsapplynpfunc(x,xlnsnp.ravel,xlnsnpv.ravel,xlnsnpb.ravel,xlnsnpr.ravel,np.ravel)
shape = lambda x: xlnsapplynpfunc(x,xlnsnp.shape,xlnsnpv.shape,xlnsnpb.shape,xlnsnpr.shape,lambda x:x.shape)
size = lambda x: xlnsapplynpfunc(x,xlnsnp.size,xlnsnpv.size,xlnsnpb.size,xlnsnpr.size,lambda x:x.size)
sign = lambda x: xlnsapplynpfunc(x,xlnsnp.sign,xlnsnpv.sign,xlnsnpb.sign,xlnsnpr.sign,np.sign)
argmax = lambda x,axis=None: xlnsapplynpfunc(x,xlnsnp.argmax,xlnsnpv.argmax,xlnsnpb.argmax,xlnsnpr.argmax,np.argmax)

def xlnswhere(x,vt,vf):
  if isinstance(x,xlnsnpv): 
    return xlnsnpv.where(x,vt,vf)
  elif isinstance(x,xlnsnp): #incl xlnsnpv
    return xlnsnp.where(x,vt,vf)
  elif isinstance(x,xlnsnpb): 
    return xlnsnpb.where(x,vt,vf)
  elif isinstance(x,xlnsnpr):
    return xlnsnpr.where(x,vt,vf)
  else:
    return np.where(x,vt,vf)

def xlnsreshape(x,shape):
  if isinstance(x,xlnsnpv): 
    return xlnsnpv.reshape(x,shape)
  elif isinstance(x,xlnsnp): #incl xlnsnpv
    return xlnsnp.reshape(x,shape)
  elif isinstance(x,xlnsnpb): 
    return xlnsnpb.reshape(x,shape)
  elif isinstance(x,xlnsnpr):
    return xlnsnpr.reshape(x,shape)
  else:
    return np.reshape(x,shape)

reshape = lambda x,shape: xlnsreshape(x,shape)
where = lambda x,vt,vf: xlnswhere(x,vt,vf)

def xlnsapplynpconstack(x,xlnsnp_func,xlnsnpv_func,xlnsnpb_func,xlnsnpr_func,np_func):
  for item in x:
    if type(x[0]) != type(item):
       print("invalid xlns concatenate,vstack,hstack: types must match")
  if isinstance(x[0],xlnsnpv): 
    return xlnsnpv_func(x)
  elif isinstance(x[0],xlnsnp): #incl xlnsnpv
    return xlnsnp_func(x)
  elif isinstance(x[0],xlnsnpb): 
    return xlnsnpb_func(x)
  elif isinstance(x[0],xlnsnpr):
    return xlnsnpr_func(x)
  else:
    return np_func(x)

concatenate = lambda x: xlnsapplynpconstack(x,xlnsnp.concatenate,xlnsnpv.concatenate,xlnsnpb.concatenate,xlnsnpr.concatenate,np.concatenate)
vstack = lambda x:      xlnsapplynpconstack(x,xlnsnp.vstack,xlnsnpv.vstack,xlnsnpb.vstack,xlnsnpr.vstack,np.vstack)
hstack = lambda x:      xlnsapplynpconstack(x,xlnsnp.hstack,xlnsnpv.hstack,xlnsnpb.hstack,xlnsnpr.hstack,np.hstack)
max = lambda x: xlnsapplynpconstack(x,xlnsnp.max,xlnsnpv.max ,xlnsnpb.max ,xlnsnpr.max ,np.max )
#zeros = lambda x: xlnsapplynpconstack(x,xlnsnp.zeros,xlnsnpv.zeros ,xlnsnpb.zeros ,xlnsnpr.zeros ,np.zeros )
 
class xlns:
 """simple scalar LNS using global base xlnsB"""
 def __init__(self,v):
  global xlns1stCons
  xlns1stCons = False
  if isinstance(v,int) or isinstance(v,float):
   if abs(v)!=0:
    self.x = int(round(math.log(abs(v))/math.log(xlnsB)))
   else:
    self.x = -1e1000 #-inf
   self.s = False if v>=0.0 else True 
  elif isinstance(v,xlns):  # 01/21/24 copy constructor!!!!
   self.x = v.x
   self.s = v.s
  elif isinstance(v,str):
   if v.replace(".","",1).replace("+","",2).replace("-","",2).replace("e","",1).isdigit():
    self.x = xlns(float(v)).x
    self.s = xlns(float(v)).s
   else:
    self.x = v #weird case for empty constructor
    self.s = False
  #elif isinstance(v,xlnsr):
  #  temp = xlns(float(v))
  #  self.x = temp.x
  #  self.s = temp.s
  elif isinstance(v,xlnsnp) or isinstance(v,xlnsnpr) or isinstance(v, xlnsnpv) or isinstance(v,xlnsnpb):
    temp = v.xlns()
    if v.size()==1:
       self.x = temp[0].x
       self.s = temp[0].s
    else:
       print("cannot cast non-scalar xlnsnp[r] as xlns")
       return NotImplemented
  else:
 #self.x = v
   #self.s = False
   # new case that handles floatable types like xlnsv 9/20/24
   # elim xlnsr case above
   #print("new case that handles floatable types like xlnsv 9/20/24")
   temp = xlns(float(v))
   self.x = temp.x
   self.s = temp.s
 def __float__(self):
  global xlnsB
  return (-1 if self.s else 1) * float(xlnsB**self.x)
 def __int__(self):
  return int(float(self))
 def __str__(self):
  if self.x == -1e1000:
   return "0"
  log10 = self.x/(math.log(10)/math.log(xlnsB))
  #print("log10=",log10)
  if abs(log10)>10:
   return ("-" if self.s else "")+str(10.0**(log10%1))+"E"+str(math.floor(log10))
  else: 
   return str(float(self))
 def __repr__(self):
  return "xlns("+str(self)+")"
 def conjugate(self):  #needed for .var() and .std() in numpy
  return(self)
 def __abs__(self):
  t=xlns("")
  t.x = self.x
  t.s = False
  return t
 def exp(x):
  if isinstance(x,float) or isinstance(x,int):
   return xlns(math.exp(x))
  return xlns(math.exp(float(x)))
 def log(x):
  if isinstance(x,float) or isinstance(x,int):
   return xlns(math.log(x))
  return xlns(math.log(float(x)))
 def __mul__(self,v):
  t=xlns("")
  if isinstance(v,int) or isinstance(v,float):
   return self*xlns(v)
  if isinstance(v,xlnsnp) or isinstance(v,xlnsnpr) or isinstance(v,xlnsnpb):
   return v*self
  if not isinstance(v,xlns):
   return self*xlns(v)
  t.x = self.x + v.x
  t.s = self.s != v.s
  return t
 def __pow__(self,v):
  t=xlns("")
  if isinstance(v,int):
    t.x = self.x*v
    t.s = self.s if v%2==1 else False
  elif isinstance(v,float):
    if self.s:
      return float(self)**v  #let it become Complex float; lets -1**2.0 slip thru to float; someday CLNS
    t.x = self.x*v
    t.s = False
  else:
    if self.s:
      return float(self)**float(v)  #let it become Complex float; someday CLNS  
    t.x = self.x*float(v)
    t.s = False
  return t
 def __truediv__(self,v):
  t=xlns("")
  if isinstance(v,int) or isinstance(v,float):
   return self/xlns(v)
  if isinstance(v,xlnsnp) or isinstance(v,xlnsnpr) or isinstance(v,xlnsnpb):
   return (1/v)*self
  if not isinstance(v,xlns):
   return self/xlns(v)
  t.x = self.x - v.x
  t.s = self.s != v.s
  return t
 def __add__(self,v):
  global xlnsB
  # tested F=23 using sbdb_ufunc (instead of orig sb() db()):
  #   >>> iota2=np.arange(1,100000.,2)
  #   >>> (1/((25000*[1,-1])*iota2)).sum()*4
  #   3.141572653589794
  #   >>> iota2x=np.array(xl.xlnscopy(iota2))
  #   >>> (1/((25000*[1.,-1.])*iota2x)).sum()*4
  #   xlns(3.1415673031204436)
  #   >>> iota2xnp=xl.xlnsnp(iota2)
  #   >>> (1/((25000*[1.,-1.])*iota2xnp)).sum()*4
  #   xlnsnp([xlns(3.141572754439485)])
  # distinction of first is FP higher precision
  # distinction of last two due to non-assoc + also
  # np.sum() (of xlns) in order; xlnsnp.sum() out of order
  #
  def sb(z):
     #global xlnsExactsb
     #if xlnsExactsb:
     #return sbexact(z)
     if z<=0:
       #print(('-',sbdb_ufunc(z,0)//2,sbexact(z)))
       return int(sbdb_ufunc(z,0)//2)
     else:
       #print(('+',sbdb_ufunc(-z,0)//2+z,sbexact(z)))
       return int(sbdb_ufunc(-z,0)//2+z)
     #else:
     #return sbmitch(z) 
  def db(z):
     #global xlnsExactdb
     #if xlnsExactdb:
     #return dbexact(z)
     if z==0:
       return -1e1000  #-inf
     elif z<0:
       return int(sbdb_ufunc(z,1)//2)
     else:
       return int(sbdb_ufunc(-z,1)//2+z)
     #else:
     #return dbmitch(z) 
  t=xlns("")
  if isinstance(v,int) or isinstance(v,float):
   return xlns(v)+self
  if isinstance(v,xlnsnp) or isinstance(v,xlnsnpr) or isinstance(v,xlnsnpb):
   return v+self
  if not isinstance(v,xlns):
   return self+xlns(v)
  if v.x==-1e1000:  #-inf
   return self
  if self.x==-1e1000:  #-inf
   return v
  z = self.x - v.x
  if self.s == v.s:
   t.x = v.x + sb(z)
   t.s = v.s
  else:
   t.x = v.x + db(z)
   t.s = v.s if v.x>self.x else self.s
  return t
  #
  #following replaced by lpvip
  #def sbexact(z):
  # return int(round(math.log(1.0 + xlnsB**z)/math.log(xlnsB)))
  #def dbexact(z):
  # if z==0:
  #  return -1e1000  #-inf
  # else:
  #  return int(round(math.log(abs(1.0 - xlnsB**z))/math.log(xlnsB)))
  #def mitch(z):
  # zI = z >> xlnsF 
  # zF = z & xlnsmitchmask
  # return ((1<<xlnsF)+zF) >> (-zI)
  #def sbmitch(z):
  # if z<=0:
  #  return mitch(z)
  # else:
  #  return z+mitch(-z) 
  #def dbcorrect(z):
  # if z < 1<<(xlnsF-4):
  #   return 4 << xlnsF
  # elif z < 1<<(xlnsF-3):
  #   return 3 << xlnsF
  # elif z < 1<<(xlnsF-2):
  #   return 2 << xlnsF
  # elif z < 1<<(xlnsF-1):
  #   return 1 << xlnsF
  # else:
  #   return 0
  #def dbmitch(z):
  # if z==0:
  #  return -1e1000  #-inf
  # elif z<0:
  #  return -mitch(z+xlnsmitchoffset)-dbcorrect(-z)
  # else:
  #  return z-mitch(-z+xlnsmitchoffset)-dbcorrect(z)
  #
 def __neg__(self):
  t=xlns("")
  t.x = self.x
  t.s = False if self.s else True
  return t
 def __pos__(self):
  return self
 def __sub__(self,v):
  return self.__add__(-v)
 def __radd__(self,v):
  return self.__add__(v)
 def __rsub__(self,v):
  return (-self).__add__(v)
 def __rmul__(self,v):
  return self.__mul__(v)
 def __rtruediv__(self,v):
  t=xlns("")
  if isinstance(v,int) or isinstance(v,float):
   return xlns(v)/self
  if not isinstance(v,xlns):
   return xlns(v)/self
  return v/self  
 def __rpow__(self,v):
  t=xlns("")
  if isinstance(v,int) or isinstance(v,float):
   return xlns(v)**self
  if not isinstance(v,xlns):
   return xlns(v)**self
  return v**self  
 def __iadd__(self,v):
  self = self.__add__(v)
  return self
 def __isub__(self,v):
  self = self.__sub__(v)
  return self
 def __imul__(self,v):
  self = self.__mul__(v)
  return self
 def __idiv__(self,v):
  self = self.__div__(v)
  return self
 def __ipow__(self,v):
  self = self.__pow__(v)
  return self
 def __eq__(self,v):
  t=xlns("")
  #if isinstance(v,int) or isinstance(v,float):
  if not isinstance(v,xlns):
   return self==xlns(v)
  return (self.x==v.x) and (self.s==v.s)
 def __ne__(self,v):
  t=xlns("")
  #if isinstance(v,int) or isinstance(v,float):
  if not isinstance(v,xlns):
   return self!=xlns(v)
  return (self.x!=v.x) or (self.s!=v.s)
 def __lt__(self,v):
  t=xlns("")
  #if isinstance(v,int) or isinstance(v,float):
  if not isinstance(v,xlns):
   return self<xlns(v)
  if self.s:
   if v.s:
    return self.x > v.x
   else:
    return True
  else:  
   if v.s:
    return False
   else:
    return self.x < v.x
 def __gt__(self,v):
  t=xlns("")
  #if isinstance(v,int) or isinstance(v,float):
  if not isinstance(v,xlns):
   return self>xlns(v)
  if self.s:
   if v.s:
    return self.x < v.x
   else:
    return False
  else:  
   if v.s:
    return True
   else:
    return self.x > v.x
 def __le__(self,v):
  t=xlns("")
  #if isinstance(v,int) or isinstance(v,float):
  if not isinstance(v,xlns):
   return self<=xlns(v)
  if self.s:
   if v.s:
    return self.x >= v.x
   else:
    return True
  else:  
   if v.s:
    return False
   else:
    return self.x <= v.x
 def __ge__(self,v):
  t=xlns("")
  #if isinstance(v,int) or isinstance(v,float):
  if not isinstance(v,xlns):
   return self>=xlns(v)
  if self.s:
   if v.s:
    return self.x <= v.x
   else:
    return False
  else:  
   if v.s:
    return True
   else:
    return self.x >= v.x

#generalizes all the xlns*copy routines
def xlnscopy(x,xlnstype=xlns,setFB=None):
  """deep copy of list converting to xlns (or type and precision specified by 2nd and 3rd args)"""
  r=[]
  for y in x:
   if isinstance(y,list) or isinstance(y,np.ndarray):
    if setFB == None:
      r+=[xlnscopy(y,xlnstype)]
    else:
      #print('recursive setFB='+str(setFB))
      r+=[xlnscopy(y,xlnstype,setFB)]
   elif isinstance(y,int) or isinstance(y,float) or isinstance(y,np.float64) or isinstance(y,np.float32) or isinstance(y,xlns) or isinstance(y,xlnsr) or isinstance(y,xlnsnp) or isinstance(y,xlnsnpr) or isinstance(y,xlnsnpv) or isinstance(y,xlnsv) or isinstance(y,xlnsb):
    if setFB == None:
      r+=[xlnstype(y)]
    else:
      #print('setFB='+str(setFB))
      r+=[xlnstype(y,setFB)]
   else:
    r+=[y]
  return r

#def xlnscopy(x):
#  r=[]
#  for y in x:
#   if isinstance(y,list) or isinstance(y,np.ndarray):
#    r+=[xlnscopy(y)]
#   elif isinstance(y,int) or isinstance(y,float) or isinstance(y,np.float64) or isinstance(y,np.float32) or isinstance(y,xlns) or isinstance(y,xlnsr) or isinstance(y,xlnsnp) or isinstance(y,xlnsnpr) or isinstance(y,xlnsnpv) or isinstance(y,xlnsv) or isinstance(y,xlnsb):
#    r+=[xlns(y)]
#   else:
#    r+=[y]
#  return r

#all the rest of xlns*copy are deprecated, and will be removed

def XXXxlnsrcopy(x):
  r=[]
  for y in x:
   if isinstance(y,list) or isinstance(y,np.ndarray):
    r+=[XXXxlnsrcopy(y)]
   elif isinstance(y,int) or isinstance(y,float) or isinstance(y,np.float64) or isinstance(y,np.float32) or isinstance(y,xlns) or isinstance(y,xlnsr) or isinstance(y,xlnsnp) or isinstance(y,xlnsnpr) or isinstance(y,xlnsnpv) or isinstance(y,xlnsv) or isinstance(y,xlnsb):
    r+=[xlnsr(y)]
   else:
    r+=[y]
  return r

def XXXxlnsudcopy(x):
  r=[]
  for y in x:
   if isinstance(y,list) or isinstance(y,np.ndarray):
    r+=[XXXxlnsudcopy(y)]
   elif isinstance(y,int) or isinstance(y,float) or isinstance(y,np.float64) or isinstance(y,np.float32) or isinstance(y,xlns) or isinstance(y,xlnsr) or isinstance(y,xlnsnp) or isinstance(y,xlnsnpr) or isinstance(y,xlnsnpv) or isinstance(y,xlnsv) or isinstance(y,xlnsb):
    r+=[xlnsud(y)]
   else:
    r+=[y]
  return r

def XXXxlnsvcopy(x,setF=None):
  if setF == None:
    copyF = xlnsF
  else:
    copyF = setF
  r=[]
  for y in x:
   if isinstance(y,list) or isinstance(y,np.ndarray):
    r+=[XXXxlnsvcopy(y,copyF)]
   elif isinstance(y,int) or isinstance(y,float) or isinstance(y,np.float64) or isinstance(y,np.float32) or isinstance(y,xlns) or isinstance(y,xlnsr) or isinstance(y,xlnsnp) or isinstance(y,xlnsnpr) or isinstance(y,xlnsnpv) or isinstance(y,xlnsv) or isinstance(y,xlnsb):
    r+=[xlnsv(y,copyF)]
   else:
    r+=[y]
  return r

def XXXxlnsbcopy(x,setB=None):
  if setB == None:
    copyB = xlnsB
  else:
    copyB = setB
  r=[]
  for y in x:
   if isinstance(y,list) or isinstance(y,np.ndarray):
    r+=[XXXxlnsbcopy(y,copyB)]
   elif isinstance(y,int) or isinstance(y,float) or isinstance(y,np.float64) or isinstance(y,np.float32) or isinstance(y,xlns) or isinstance(y,xlnsr) or isinstance(y,xlnsnp) or isinstance(y,xlnsnpr) or isinstance(y,xlnsnpv) or isinstance(y,xlnsv) or isinstance(y,xlnsb):
    r+=[xlnsb(y,copyB)]
   else:
    r+=[y]
  return r

class xlnsnp:
 """numpy-like array of non-redundant LNS using global base xlnsB"""
 def __init__(self,v):
  global xlns1stCons
  xlns1stCons = False
  if isinstance(v,int) or isinstance(v,float):
   #print("have scalar: "+str(v))
   if v == 0:
     self.nd = np.array([-0x7fffffffffffffff-1],dtype="i8")      #0x8000000000000000 avoiding int64 limit
   else:
     self.nd = np.array([2*xlns(v).x+xlns(v).s],dtype="i8")
  elif isinstance(v,xlns):
   if v.x == -1e1000:
     self.nd = np.array([-0x7fffffffffffffff-1],dtype="i8")      #0x8000000000000000 avoiding int64 limit
   else:
     self.nd = np.array([2*v.x+v.s],dtype="i8")
  elif isinstance(v,xlnsnp):    #copy constructor
     self.nd = v.nd
  elif isinstance(v,xlnsr):
   if v.p==v.n:
     self.nd = np.array([-0x7fffffffffffffff-1],dtype="i8")      #0x8000000000000000 avoiding int64 limit
   elif v.p>v.n:
     self.nd = np.array([2*(xlns(v).x)],dtype="i8")
   else:
     self.nd = np.array([2*(xlns(v).x)+1],dtype="i8")
  elif isinstance(v,xlnsb) or isinstance(v,xlnsv):
     self.nd = xlnsnp(float(v)).nd
  #elif isinstance(v,xlnsnpr):
  #   #print("start npr->np")
  #   temp = xlnsnp(v.xlns())
  #   #print("stop npr->np")
  #   self.nd = temp.nd
  elif isinstance(v,xlnsnpb) or isinstance(v,xlnsnpv) or isinstance(v,xlnsnpr):
     self.nd = xlnsnp(np.float64(v.xlns())).nd
  elif isinstance(v,np.ndarray):
   #print("have ndarray: "+str(v))
   #print("first elem: "+repr(np.ravel(v)[0]))
   #print(isinstance(np.ravel(v)[0],xlns))
   if isinstance(np.ravel(v)[0],xlns):
     t=[]
     for y in np.ravel(v):
        #print(y)
        if y.x == -1e1000:
           #print("zero")
           t += [ -0x7fffffffffffffff-1 + y.s] #0x8000000000000000 avoiding int64 limit
        else:
           t += [2*y.x + y.s] 
     #print("t="+repr(t))
     self.nd = np.array(np.reshape(t,np.shape(v)),dtype="i8")
   elif isinstance(np.ravel(v)[0],xlnsv):
     #print("xlnsv array -> xlnsnp")
     self.nd = xlnsnp(np.float64(v)).nd
   else:
     t=[]
     for y in np.ravel(xlnscopy(v)):
        #print(y)
        if isinstance(y,xlns):
           if y.x == -1e1000:
             #print("zero")
             t += [ -0x7fffffffffffffff-1 + y.s] #0x8000000000000000 avoiding int64 limit
           else:
             t += [2*y.x + y.s] 
           #print("t="+repr(t))
        else:
           if y == 0:
             #print("zero")
             t += [ -0x7fffffffffffffff-1 ] #0x8000000000000000 avoiding int64 limit
           else:
             t += [2*xlns(y).x + xlns(y).s] 
           #print("t="+repr(t))
     self.nd = np.array(np.reshape(t,np.shape(v)),dtype="i8")
  elif isinstance(v,list):
   if isinstance(v[0],xlnsv):
     #print("xlnsv list -> xlnsnp")
     self.nd = xlnsnp(np.float64(v)).nd
   else:
     t=[]
     for y in xlnscopy(v):
       #print(y)
       if y.x == -1e1000:
         #print("zero")
         t += [ -0x7fffffffffffffff-1 + y.s] #0x8000000000000000 avoiding int64 limit
       else:
         t += [2*y.x + y.s] 
       #print("t="+repr(t))
     self.nd = np.array(np.reshape(t,np.shape(v)),dtype="i8")
  else:
   self.nd = np.array([]) 
 # weird test
 def add(self,v):
   return self+v
 def __str__(self):
   return str(self.xlns())
 def __repr__(self):
  return "xlnsnp("+str(self.xlns())+")"
 def __bool__(self):
  return bool(self.nd == 0)
 #following not found in numpy
 def xlns(self):
  """convert xlnsnp to similar-shaped list of xlns"""
  t=[]
  for y in np.ravel(self.nd):
     txlns = xlns(0)
     if (y|1) == -0x7fffffffffffffff:   #obscure 0x8000000000000001 avoiding int64 limit
       txlns.x = -1e1000
     else:
       txlns.x = y//2
     txlns.s = True if y&1 else False
     t += [txlns]
  return np.reshape(t,np.shape(self.nd)) 
 def ovf(x):
  """return copy except for values that underflow/overflow, which are clipped according to xlnssetovf"""
  global ovfmin,ovfmax
  return xlnsnp.where((abs(x)>=ovfmin)*(abs(x)<ovfmax), x, xlnsnp.zeros(x.shape())+(abs(x)>=ovfmax)*ovfmax*x.sign())
 #end of those not found in numpy
 def conjugate(self):  #needed for .var() and .std() in numpy
  return(self)
 def exp(x):
  if isinstance(x,float) or isinstance(x,int):
   return xlnsnp(math.exp(x))
  if isinstance(x,xlnsnp):
   return math.exp(1)**x
   #return xlnsnp(np.exp(x.xlns()))
 def log(x):
  if isinstance(x,float) or isinstance(x,int):
   return xlnsnp(math.log(x))
  if isinstance(x,xlnsnp):
   return xlnsnp(np.log(x.xlns()))
 def __mul__(self,v):
  if isinstance(v,int) or isinstance(v,float):
    return self*xlnsnp(v)
  if isinstance(v,xlnsnp):
    t = xlnsnp("")
    t.nd = np.where(((self.nd|1)==-0x7fffffffffffffff)|((v.nd|1)==-0x7fffffffffffffff), #obscure way to say 0x8000000000000001 and
                  (-0x7fffffffffffffff-1) |((self.nd^v.nd)&1),                        #0x8000000000000000 avoiding int64 limit
                  (self.nd + v.nd - (v.nd&1)) ^ (v.nd&1) )
    return t
  else:
    return self*xlnsnp(v)
 def __truediv__(self,v):
  if isinstance(v,int) or isinstance(v,float):
    return self/xlnsnp(v)
  if isinstance(v,xlnsnp):
    t = xlnsnp("")
    t.nd = np.where(((self.nd|1)==-0x7fffffffffffffff), #obscure way to say 0x8000000000000001 and
                  (-0x7fffffffffffffff-1) |((self.nd^v.nd)&1),                        #0x8000000000000000 avoiding int64 limit
                  (self.nd - v.nd + (v.nd&1)) ^ (v.nd&1) )
    return t
  else:
    return self/xlnsnp(v)
 def __neg__(self):
  t=self*-1
  return t
 def __pos__(self):
  return self
 def __sub__(self,v):
  return self.__add__(-v)
 def __radd__(self,v):
  #print("radd")
  #print(self.nd)
  #print(v)
  return self.__add__(xlnsnp(v))
 def __rsub__(self,v):
  return (-self).__add__(xlnsnp(v))
 def __rdiv__(self,v):
  return self.__div__(xlnsnp(v))
 def __rmul__(self,v):
  return self.__mul__(xlnsnp(v))
 def __rtruediv__(self,v):
  return xlnsnp(v).__truediv__(self)
 def __pow__(self,v):
  t=xlnsnp("")
  if isinstance(v,int):
   if (v&1)==0:
     t.nd = ((self.nd&-2)*v)&-2
   else:
     t.nd = (((self.nd&-2)*v)&-2)|(self.nd&1) 
   return t
   #return xlnsnp(self.xlns()**v)
  if isinstance(v,float):
   t.nd = np.int64((self.nd&-2)*v)&-2
   return t
  return xlnsnp.exp(xlnsnp.log(self)*v)
 def __rpow__(self,v):
  t=xlnsnp("")
  if isinstance(v,xlnsnp):
   return v**self  
  elif isinstance(v,float) or isinstance(v,int):
   t.nd=(2-4*(self.nd&1))*np.int64(xlnsB**((self.nd//2))*math.log(v,xlnsB))
   return t
  else:
   return xlnsnp(v)**self
 #non-ufunc 
 def __len__(self):
  return self.nd.__len__()
 def __getitem__(self,v):
  t=xlnsnp("")
  t.nd = self.nd.__getitem__(v)
  return t
 def __setitem__(self,v1,v2):
  if isinstance(v2,xlnsnp):
    self.nd.__setitem__(v1,v2.nd)
  else:
    self.nd.__setitem__(v1,xlnsnp(v2).nd)
 def where(self,vt,vf):
  #print(type(self))
  #print(type(vt))
  #print(type(vf))
  if isinstance(vt,xlnsnp) and isinstance(vf,xlnsnp):
   t = xlnsnp("")
   t.nd = np.where(self.nd==0, vt.nd, vf.nd)
   return t
  else:
   print("xlnsnp.where must match type")
   return None
 def sign(v):
  if isinstance(v,xlnsnp):
   t = xlnsnp("")
   t.nd = np.where( (v.nd|1)==-0x7fffffffffffffff, #obscure way to say 0x8000000000000001 
                    #0, 1-2*(v.nd&1))
                    v.nd, v.nd&1)
   return t
  else:
   return np.sign(v)
 def concatenate(list1):
  t = xlnsnp("")
  list2 = []
  for item in list1:
    list2 += [item.nd]
  t.nd = np.concatenate(list2)
  return t 
 def vstack(list1):
  t = xlnsnp("")
  list2 = []
  for item in list1:
    list2 += [item.nd]
  t.nd = np.vstack(list2)
  return t 
 def hstack(list1):
  t = xlnsnp("")
  list2 = []
  for item in list1:
    list2 += [item.nd]
  t.nd = np.hstack(list2)
  return t 
 def zeros(v):
  t = xlnsnp("")
  t.nd = np.zeros(v,dtype="i8")+(-0x7fffffffffffffff-1)
  return t
 def sum(self,axis=None):
  """divide and conquer summation that reorders operands to minimize __add__ calls"""
  #ac = xlnsnp.concatenate((self,xlnsnp.zeros(2**(math.ceil(math.log(len(self),2)))-len(self))))
  #while len(ac)>1:
  #  ac=ac[0:len(ac)//2]+ac[len(ac)//2:]
  #return ac[0]
  if axis == None:
    a = 1.0*xlnsnp.ravel(self)
  elif axis == 1:
    a = 1.0*xlnsnp.transpose(self)
  else:
    a = 1.0*self
  #print('sum shape='+str(xlnsnp.shape(a)))
  if len(a)==1:
    return a
  mid = 2**int(np.floor(math.log(len(a)-1,2)))
  end = len(a)
  while mid > 0:
    a[0:end-mid] += a[mid:end]
    end = mid
    mid //= 2
  return(a[0])
 def prod(self,axis=None):
  if axis == None:
    a = 1.0*xlnsnp.ravel(self)
  elif axis == 1:
    a = 1.0*xlnsnp.transpose(self)
  else:
    a = 1.0*self
  mid = 2**int(np.floor(math.log(len(a)-1,2)))
  end = len(a)
  while mid > 0:
    a[0:end-mid] *= a[mid:end]
    end = mid
    mid //= 2
  return(a[0])
 def ravel(v):
  t = xlnsnp("")
  t.nd = np.ravel(v.nd)
  return t
 def shape(v):
  return np.shape(v.nd)
 def transpose(v):
  t = xlnsnp("")
  t.nd = np.transpose(v.nd)
  return t
 def size(v):
  return np.size(v.nd)
 def max(self,axis=None):
  return xlnsnp(np.max(self.xlns(),axis=axis))
 def min(self,axis=None):
  return xlnsnp(np.min(self.xlns(),axis=axis))
 def argmax(self,axis=None):
  return np.argmax(self.xlns(),axis=axis)
 def reshape(self,shape):
  t = xlnsnp("")
  t.nd = np.reshape(self.nd,shape)
  return t 
 def __matmul__(self,v):
  if isinstance(v, xlnsnpr):
     print("xlnsnp@xlnsnpr not implemented by xlnsnp")
     return NotImplemented
  t = xlnsnp.zeros( (xlnsnp.shape(self)[0],xlnsnp.shape(v)[1]) )
  vtrans = xlnsnp.transpose(v)
  for r in range(xlnsnp.shape(self)[0]):
     t[r] = (self[r]*vtrans).sum(axis=1)
  #  for c in range(xlnsnp.shape(v)[1]):
  #    t[r][c] = (self[r]*vtrans[c]).sum()  
  return t
 def __abs__(self):
  t=xlnsnp("")
  t.nd = self.nd & -2  # way to say 0xfffffffffffffffe
  return t
 def __add__(self,v):
  if isinstance(v,int) or isinstance(v,float):
    return self+xlnsnp(v)
  if isinstance(v,xlnsnp):
    #print("both")
    t = xlnsnp("")
    t.nd = np.where(((self.nd|1)==-0x7fffffffffffffff), #obscure way to say 0x8000000000000001 
                  v.nd,
                  np.where(((v.nd|1)==-0x7fffffffffffffff), #obscure way to say 0x8000000000000001 
                           self.nd,
                           np.maximum(self.nd,v.nd) + sbdb_ufunc(-np.abs(v.nd//2 - self.nd//2), (self.nd^v.nd)&1) ))  
    return t
  else:
    #print("here")
    return self+xlnsnp(v)
 def __lt__(self,v):
  t=xlnsnp("")
  if isinstance(v,int) or isinstance(v,float) or isinstance(v,xlns):
   if v==0:
     t.nd=np.where(self.nd&1,0,-0x7fffffffffffffff-1)
     return t
   elif v>0:
     t.nd=np.where(self.nd&1,0,np.where(self.nd<2*xlns(v).x,0,-0x7fffffffffffffff-1))
     return t
   else:
     t.nd=np.where(self.nd&1,np.where(self.nd>(2*xlns(v).x|1),0,-0x7fffffffffffffff-1),-0x7fffffffffffffff-1)
     return t
  else:
   print("nonscalar comparison")
   return None
 def __gt__(self,v):
  t=xlnsnp("")
  if isinstance(v,int) or isinstance(v,float) or isinstance(v,xlns):
   if v==0:
     t.nd=np.where(self.nd&1,-0x7fffffffffffffff-1,np.where((self.nd|1)!=-0x7fffffffffffffff,0,-0x7fffffffffffffff-1))
     return t
   elif v>0:
     t.nd=np.where(self.nd&1,-0x7fffffffffffffff-1,np.where(self.nd>2*xlns(v).x,0,-0x7fffffffffffffff-1))
     return t
   else:
     t.nd=np.where(self.nd&1,np.where(self.nd<(2*xlns(v).x|1),0,-0x7fffffffffffffff-1),0)
     return t
  else:
   print("nonscalar comparison")
   return None
 def __ge__(self,v):
  t=xlnsnp("")
  if isinstance(v,int) or isinstance(v,float) or isinstance(v,xlns):
   if v==0:
     t.nd=np.where(self.nd&1,-0x7fffffffffffffff-1,0)
     return t
   elif v>0:
     t.nd=np.where(self.nd&1,-0x7fffffffffffffff-1,np.where(self.nd>=2*xlns(v).x,0,-0x7fffffffffffffff-1))
     return t
   else:
     t.nd=np.where(self.nd&1,np.where(self.nd<=(2*xlns(v).x|1),0,-0x7fffffffffffffff-1),0)
     return t
  else:
   print("nonscalar comparison")
   return None
 def __le__(self,v):
  t=xlnsnp("")
  if isinstance(v,int) or isinstance(v,float) or isinstance(v,xlns):
   if v==0:
     t.nd=np.where(self.nd&1,0,np.where((self.nd|1)!=-0x7fffffffffffffff,0,-0x7fffffffffffffff-1))
     return t
   elif v>0:
     t.nd=np.where(self.nd&1,0,np.where(self.nd<=2*xlns(v).x,0,-0x7fffffffffffffff-1))
     return t
   else:
     t.nd=np.where(self.nd&1,np.where(self.nd>=(2*xlns(v).x|1),0,-0x7fffffffffffffff-1),-0x7fffffffffffffff-1)
     return t
  else:
   print("nonscalar comparison")
   return None
 def __eq__(self,v):
  t=xlnsnp("")
  #if isinstance(v,int) or isinstance(v,float) or isinstance(v,xlns):
  # t.nd = np.where(self.nd == 2*xlns(v).x + xlns(v).s, 0, -0x7fffffffffffffff-1)
  if not isinstance(v,xlnsnp):
   t.nd = np.where(self.nd == xlnsnp(v).nd, 0, -0x7fffffffffffffff-1)
  else:
   t.nd = np.where(self.nd == v.nd, 0, -0x7fffffffffffffff-1)
  return t
 def __ne__(self,v):
  t=xlnsnp("")
  #if isinstance(v,int) or isinstance(v,float) or isinstance(v,xlns):
  # t.nd = np.where(self.nd == 2*xlns(v).x + xlns(v).s, -0x7fffffffffffffff-1, 0)
  if not isinstance(v,xlnsnp):
   t.nd = np.where(self.nd == xlnsnp(v).nd, -0x7fffffffffffffff-1, 0)
  else:
   t.nd = np.where(self.nd == v.nd, -0x7fffffffffffffff-1, 0)
  return t


class xlnsr:
 """scalar redundant LNS using global base xlnsB"""
 def __init__(self,v):
  global xlns1stCons
  xlns1stCons = False
  if isinstance(v,xlnsr):
    self.p = v.p
    self.n = v.n
  elif isinstance(v,int) or isinstance(v,float):
    if xlns(v)>=0:
      self.p = xlns(v)
      self.n = xlns(0) 
    else:
      self.p = xlns(0) 
      self.n = -xlns(v)
  elif isinstance(v,xlns):
    if v>=0:
      self.p = v
      self.n = xlns(0) 
    else:
      self.p = xlns(0) 
      self.n = abs(v)
  elif isinstance(v,xlnsnp) or isinstance(v,xlnsnpr):
    temp = v.xlns()
    if v.size()==1:
       if temp[0]>=0:
          self.p = temp[0]
          self.n = xlns(0) 
       else:
          self.p = xlns(0) 
          self.n = -temp[0] 
    else:
       print("cannot cast non-scalar xlnsnp[r] as xlnsr")
       return NotImplemented
  else:
    if xlns(v)>=0:
      self.p = xlns(v)
      self.n = xlns(0) 
    else:
      self.p = xlns(0) 
      self.n = -xlns(v)
 def __float__(self):
  return float(self.p)-float(self.n)
 def __int__(self):
  return int(float(self))
 def __str__(self):
  return str(float(self))
 def __repr__(self):
  return "xlnsr("+str(self)+" = "+str(float(self.p))+'-'+str(float(self.n))+")"
 def __neg__(self):
  t = xlnsr(self)
  t.p = self.n
  t.n = self.p
  return t
 def __add__(self,v):
  if isinstance(v,xlnsnp) or isinstance(v,xlnsnpr) or isinstance(v,xlnsnpb):
   return v+self
  t = xlnsr(v)
  t.p += self.p
  t.n += self.n
  return t
 def __radd__(self,v):
  return self.__add__(v)
 def __sub__(self,v):
  if isinstance(v,xlnsnp) or isinstance(v,xlnsnpr) or isinstance(v,xlnsnpb):
   return (-v)+self
  t = xlnsr(v)
  t2 = self.p + t.n
  t.n = self.n + t.p
  t.p = t2
  return t
 def __rsub__(self,v):
  return (-self).__add__(v)
 def __mul__(self,v):
  if isinstance(v,int) or isinstance(v,float) or isinstance(v,xlns) or isinstance(v,xlnsv) or isinstance(v,xlnsb):
   t = xlnsr(v)
   if v>=0:
     t.n = self.n * t.p 
     t.p = self.p * t.p 
   else:
     t.p = self.n * t.n 
     t.n = self.p * t.n
   return t
  elif isinstance(v, xlnsr):
   t = xlnsr(float(self)*float(v))
   print("xlnsr*xlnsr: result computed in float")
   return t
  elif isinstance(v,xlnsnp) or isinstance(v,xlnsnpr) or isinstance(v,xlnsnpb):
   return v*self
  else:
   return self*xlnsr(v)
 def __rmul__(self,v):
  return self.__mul__(v)
 def __truediv__(self,v):
  if isinstance(v,int) or isinstance(v,float) or isinstance(v,xlns) or isinstance(v,xlnsv) or isinstance(v,xlnsb):
   t = xlnsr(v)
   if v>=0:
     t.n = self.n / t.p 
     t.p = self.p / t.p 
   else:
     t.p = self.n / t.n 
     t.n = self.p / t.n
   #t = xlnsr(abs(v))
   #if v>=0:
   #  t.n = self.n / t.p 
   #  t.p = self.p / t.p 
   #else:
   #  t.n = self.p / t.p 
   #  t.p = self.n / t.p 
   #t.p = self.p / xlns(v)
   #t.n = self.n / xlns(v)
   return t
  elif isinstance(v, xlnsr):
   t = xlnsr(float(self)/float(v))
   print("xlnsr/xlnsr: result computed in float")
   return t
  elif isinstance(v,xlnsnp) or isinstance(v,xlnsnpr) or isinstance(v,xlnsnpb):
   return (1/v)*self
  else:
   return self/xlnsr(v)
 def __rtruediv__(self,v):
  t = xlnsr(float(v)/float(self))
  print("any/xlnsr: result computed in float")
  return t
 def __abs__(self):
  t = xlnsr(self)
  if self.p<self.n:
   t.p = self.n
   t.n = self.p
  return t
 def __pos__(self):
  return self
 def __iadd__(self,v):
  self = self.__add__(v)
  return self
 def __isub__(self,v):
  self = self.__sub__(v)
  return self
 def __imul__(self,v):
  self = self.__mul__(v)
  return self
 def __eq__(self,v):
  #if isinstance(v,int) or isinstance(v,float) or isinstance(v,xlns) or isinstance(v,xlnsnp) or isinstance(v,xlnsnpr):
  if not isinstance(v,xlnsr):
   return self==xlnsr(v)
  t = self-v
  return t.p==t.n 
 def __ne__(self,v):
  #if isinstance(v,int) or isinstance(v,float) or isinstance(v,xlns) or isinstance(v,xlnsnp) or isinstance(v,xlnsnpr):
  if not isinstance(v,xlnsr):
   return self!=xlnsr(v)
  t = self-v
  return t.p!=t.n 
 def __lt__(self,v):
  #if isinstance(v,int) or isinstance(v,float) or isinstance(v,xlns) or isinstance(v,xlnsnp) or isinstance(v,xlnsnpr):
  if not isinstance(v,xlnsr):
   return self<xlnsr(v)
  t = self-v
  return t.p<t.n 
 def __gt__(self,v):
  #if isinstance(v,int) or isinstance(v,float) or isinstance(v,xlns) or isinstance(v,xlnsnp) or isinstance(v,xlnsnpr):
  if not isinstance(v,xlnsr):
   return self>xlnsr(v)
  t = self-v
  return t.p>t.n 
 def __le__(self,v):
  #if isinstance(v,int) or isinstance(v,float) or isinstance(v,xlns) or isinstance(v,xlnsnp) or isinstance(v,xlnsnpr):
  if not isinstance(v,xlnsr):
   return self<=xlnsr(v)
  t = self-v
  return t.p<=t.n 
 def __ge__(self,v):
  #if isinstance(v,int) or isinstance(v,float) or isinstance(v,xlns) or isinstance(v,xlnsnp) or isinstance(v,xlnsnpr):
  if not isinstance(v,xlnsr):
   return self>=xlnsr(v)
  t = self-v
  return t.p>=t.n 


class xlnsnpr:
 """numpy-like array of redundant LNS using global base xlnsB"""
 def __init__(self,v):
  global xlns1stCons
  xlns1stCons = False
  if isinstance(v,int) or isinstance(v,float):
   #print("have scalar: "+str(v))
   if v == 0:
     self.ndp = np.array([0],dtype="i8")      
     self.ndn = np.array([0],dtype="i8")      
   elif v > 0:
     self.ndp = np.array([xlns(v).x],dtype="i8")
     self.ndn = np.array([-0x7fffffffffffffff-1],dtype="i8")      #0x8000000000000000 avoiding int64 limit
   else:
     self.ndp = np.array([-0x7fffffffffffffff-1],dtype="i8")      #0x8000000000000000 avoiding int64 limit
     self.ndn = np.array([xlns(v).x],dtype="i8")
  elif isinstance(v,xlns):
   if v == 0:
     self.ndp = np.array([0],dtype="i8")      
     self.ndn = np.array([0],dtype="i8")      
   elif v > 0:
     self.ndp = np.array([v.x],dtype="i8")
     self.ndn = np.array([-0x7fffffffffffffff-1],dtype="i8")      #0x8000000000000000 avoiding int64 limit
   else:
     self.ndp = np.array([-0x7fffffffffffffff-1],dtype="i8")      #0x8000000000000000 avoiding int64 limit
     self.ndn = np.array([v.x],dtype="i8")
  elif isinstance(v,xlnsr):
   if v.p==v.n:
     self.ndp = np.array([0],dtype="i8")      
     self.ndn = np.array([0],dtype="i8")      
   else:
     if v.p.x != -1e1000: #-inf
        self.ndp = np.array([v.p.x],dtype="i8")
     else:
        self.ndp = np.array([-0x7fffffffffffffff-1],dtype="i8")      #0x8000000000000000 avoiding int64 limit
     if v.n.x != -1e1000: #-inf
        self.ndn = np.array([v.n.x],dtype="i8")      
     else:
        self.ndn = np.array([-0x7fffffffffffffff-1],dtype="i8")      #0x8000000000000000 avoiding int64 limit
  elif isinstance(v,xlnsb) or isinstance(v,xlnsv):
     temp = xlnsnpr(float(v))
     self.ndp = temp.ndp
     self.ndn = temp.ndn
  elif isinstance(v,xlnsnpr):  #copy constructor
     self.ndp = v.ndp
     self.ndn = v.ndn
  elif isinstance(v,xlnsnp):  
     #print("start np->npr")
     temp = xlnsnpr(v.xlns())
     #print("stop np->npr "+str(temp.shape()))
     self.ndp = temp.ndp
     self.ndn = temp.ndn
  elif isinstance(v,xlnsnpb) or isinstance(v,xlnsnpv):
     temp = xlnsnpr(np.float64(v.xlns()))
     self.ndp = temp.ndp
     self.ndn = temp.ndn
  elif isinstance(v,np.ndarray):
   #print("have ndarray: "+str(v))
   #print("first elem: "+repr(np.ravel(v)[0]))
   #print(isinstance(np.ravel(v)[0],xlns))
   if isinstance(np.ravel(v)[0],xlns):
     tp=[]
     tn=[]
     for y in np.ravel(v):
        #print(y)
        if y.x == -1e1000:
           #print("zero")
           tp += [ 0 ] # any value tp==tn
           tn += [ 0 ]
        elif y>0:
           tp += [y.x] 
           tn += [ -0x7fffffffffffffff-1 ] #0x8000000000000000 avoiding int64 limit
        else:
           tp += [ -0x7fffffffffffffff-1 ] #0x8000000000000000 avoiding int64 limit
           tn += [y.x] 
     #print("t="+repr(t))
     self.ndp = np.array(np.reshape(tp,np.shape(v)),dtype="i8")
     self.ndn = np.array(np.reshape(tn,np.shape(v)),dtype="i8")
   elif isinstance(np.ravel(v)[0],xlnsv):
     #print("xlnsv array -> xlnsnpr")
     temp = xlnsnpr(np.float64(v))
     self.ndp = temp.ndp
     self.ndn = temp.ndn
   else:
     tp=[]
     tn=[]
     for y in np.ravel(xlnscopy(v)):
        #print(y)
        if y.x == -1e1000:
           #print("zero")
           tp += [ 0 ] #any val tp==tn 
           tn += [ 0 ] #0x000000000000000 avoiding int64 limit
        elif y>0:
           tp += [y.x] 
           tn += [ -0x7fffffffffffffff-1 ] #0x8000000000000000 avoiding int64 limit
        else:
           tp += [ -0x7fffffffffffffff-1 ] #0x8000000000000000 avoiding int64 limit
           tn += [y.x] 
     #print("t="+repr(t))
     self.ndp = np.array(np.reshape(tp,np.shape(v)),dtype="i8")
     self.ndn = np.array(np.reshape(tn,np.shape(v)),dtype="i8")
  elif isinstance(v,list):
    if isinstance(v[0],xlnsv):
      #print("xlnsv list -> xlnsnpr")
      temp = xlnsnpr(np.float64(v))
      self.ndp = temp.ndp
      self.ndn = temp.ndn
    else:
      tp=[]
      tn=[]
      for y in np.ravel(xlnscopy(v)):
         #print(y)
         if y.x == -1e1000:
            #print("zero")
            tp += [ 0 ] #any tp==tn 
            tn += [ 0 ] #0x000000000000000 avoiding int64 limit
         elif y>0:
            tp += [y.x] 
            tn += [ -0x7fffffffffffffff-1 ] #0x8000000000000000 avoiding int64 limit
         else:
            tp += [ -0x7fffffffffffffff-1 ] #0x8000000000000000 avoiding int64 limit
            tn += [y.x] 
      #print("t="+repr(t))
      self.ndp = np.array(np.reshape(tp,np.shape(v)),dtype="i8")
      self.ndn = np.array(np.reshape(tn,np.shape(v)),dtype="i8")
  #need xlnsnp
  else:
   self.nd = np.array([]) 
 # weird test
 def add(self,v):
   return self+v
 def __str__(self):
   return str(self.xlns())
 def __repr__(self):
  return "xlnsnpr("+str(self.xlns())+")"
 def __bool__(self):
  return bool(self.nd == 0)
 #ovf does not make sense for xlnsnpr
 #def ovf(x):
 # global ovfmin,ovfmax
 # return xlnsnpr.where((abs(x)>=ovfmin)*bool(abs(x)<ovfmax), x, xlnsnpr.zeros(x.shape())+(abs(x)>=ovfmax)*ovfmax)#*x.sign())
 def xlns(self):
  """convert xlnsnpr to similar-shaped list of xlns"""
  t=[]
  yp = np.ravel(self.ndp)
  yn = np.ravel(self.ndn)
  for i in range(len(yp)):
     txlns = xlns(0)
     if yp[i] == yn[i]:
       txlns.x = -1e1000
     else:
       txlns.x = np.maximum(yp[i],yn[i]) + sbdb_ufunc(-np.abs(yp[i]-yn[i]), 1)//2  
     txlns.s = True if yn[i]>yp[i] else False
     t += [txlns]
  return np.reshape(t,np.shape(self.ndp)) 
 def __mul__(self,v):
  if isinstance(v,xlnsnpr) or isinstance(v,xlnsr): 
    print("should not xlnsnpr*xlns(np)r: converted to xlnsnp")
    return self*xlnsnp(v)
  elif isinstance(v,int) or isinstance(v,float):
    return self*xlnsnp(v) #really xlnsnp not xlnsnpr
  elif isinstance(v,xlnsnp): #npr * np
    t = xlnsnpr("")
    t.ndp = np.where((self.ndp==self.ndn)|((v.nd|1)==-0x7fffffffffffffff), #obscure way to say 0x8000000000000001
              0,                                                           #any tp==tn
              np.where(v.nd&1, 
                       np.where((self.ndn|1)==-0x7fffffffffffffff,         #obscure way to say 0x8000000000000001
                           -0x7fffffffffffffff-1,                          #0x8000000000000000 avoiding int64 limit
                           self.ndn + v.nd//2),
                       np.where((self.ndp|1)==-0x7fffffffffffffff,         #obscure way to say 0x8000000000000001
                           -0x7fffffffffffffff-1,                          #0x8000000000000000 avoiding int64 limit
                           self.ndp + v.nd//2))) 
    t.ndn = np.where((self.ndp==self.ndn)|((v.nd|1)==-0x7fffffffffffffff), #obscure way to say 0x8000000000000001
              0,                                                           #any tp==tn
              np.where(v.nd&1, 
                       np.where((self.ndp|1)==-0x7fffffffffffffff,         #obscure way to say 0x8000000000000001
                           -0x7fffffffffffffff-1,                          #0x8000000000000000 avoiding int64 limit
                           self.ndp + v.nd//2),
                       np.where((self.ndn|1)==-0x7fffffffffffffff,         #obscure way to say 0x8000000000000001
                           -0x7fffffffffffffff-1,                          #0x8000000000000000 avoiding int64 limit
                           self.ndn + v.nd//2))) 
    return t
  else:
    return self*xlnsnp(v)
 def __truediv__(self,v):
  if isinstance(v,xlnsnpr) or isinstance(v,xlnsr): 
    print("should not xlnsnpr/xlns(np)r: converted to xlnsnp")
    return self/xlnsnp(v)
  elif isinstance(v,int) or isinstance(v,float):
    return self/xlnsnp(v)
  elif isinstance(v,xlnsnp):
    t = xlnsnpr("")
    t.ndp = np.where(v.nd&1, 
                       np.where((self.ndn|1)==-0x7fffffffffffffff,         #obscure way to say 0x8000000000000001
                           -0x7fffffffffffffff-1,                          #0x8000000000000000 avoiding int64 limit
                           self.ndn - v.nd//2),
                       np.where((self.ndp|1)==-0x7fffffffffffffff,         #obscure way to say 0x8000000000000001
                           -0x7fffffffffffffff-1,                          #0x8000000000000000 avoiding int64 limit
                           self.ndp - v.nd//2)) 
                  #(self.ndn - v.nd//2),
                  #(self.ndp - v.nd//2)) 
    t.ndn = np.where(v.nd&1, 
                       np.where((self.ndp|1)==-0x7fffffffffffffff,         #obscure way to say 0x8000000000000001
                           -0x7fffffffffffffff-1,                          #0x8000000000000000 avoiding int64 limit
                           self.ndp - v.nd//2),
                       np.where((self.ndn|1)==-0x7fffffffffffffff,         #obscure way to say 0x8000000000000001
                           -0x7fffffffffffffff-1,                          #0x8000000000000000 avoiding int64 limit
                           self.ndn - v.nd//2)) 
                  #(self.ndp - v.nd//2),
                  #(self.ndn - v.nd//2)) 
    return t
  else:
    return self/xlnsnp(v)
 def __neg__(self):
  t=self*-1
  return t
 def __pos__(self):
  return self
 def __sub__(self,v):
  return self.__add__(-v)
 def __radd__(self,v):
  #print("radd")
  #print(self.nd)
  #print(v)
  return self.__add__(xlnsnpr(v))
 def __rsub__(self,v):
  return (-self).__add__(xlnsnpr(v))
 def __rdiv__(self,v):
  return self.__div__(xlnsnp(v)) #probably impossible
 def __rmul__(self,v):
  return self.__mul__(xlnsnp(v))
 def __rtruediv__(self,v):       #probably impossible
  return xlnsnp(v).__truediv__(self)
 #non-ufunc 
 def __len__(self):
  return self.ndp.__len__()
 def __getitem__(self,v):
  t=xlnsnpr("")
  t.ndp = self.ndp.__getitem__(v)
  t.ndn = self.ndn.__getitem__(v)
  return t
 def __setitem__(self,v1,v2):
  if isinstance(v2,xlnsnpr):
    self.ndp.__setitem__(v1,v2.ndp)
    self.ndn.__setitem__(v1,v2.ndn)
  else:
    t = xlnsnpr(v2)
    self.ndp.__setitem__(v1,t.ndp)
    self.ndn.__setitem__(v1,t.ndn)
 def where(self,vt,vf):
  #print(type(self))
  #print(type(vt))
  #print(type(vf))
  if isinstance(vt,xlnsnpr) and isinstance(vf,xlnsnp):
   t = xlnsnpr("")
   t.ndp = np.where(self.ndp==self.npn, vt.ndp, vf.ndp)
   t.ndn = np.where(self.ndp==self.npn, vt.ndn, vf.ndn)
   return t
  else:
   print("xlnsnpr.where must match type")
   return None
 def sign(v): #returns xlnsnp not xlnsnpr
  if isinstance(v,xlnsnpr):
   t = xlnsnp("")
   t.nd = np.where( v.ndp==v.ndn, 
                    -0x7fffffffffffffff-1, #obscure way to say 0x8000000000000000 
                    np.where( v.ndp > v.ndn, 0, 1))
   return t
  else:
   return np.sign(v)
 def concatenate(list1):
  t = xlnsnpr("")
  list2p = []
  list2n = []
  for item in list1:
    list2p += [item.ndp]
    list2n += [item.ndn]
  t.ndp = np.concatenate(list2p)
  t.ndn = np.concatenate(list2n)
  return t 
 def vstack(list1):
  t = xlnsnpr("")
  list2p = []
  list2n = []
  for item in list1:
    list2p += [item.ndp]
    list2n += [item.ndn]
  t.ndp = np.vstack(list2p)
  t.ndn = np.vstack(list2n)
  return t 
 def hstack(list1):
  t = xlnsnpr("")
  list2p = []
  list2n = []
  for item in list1:
    list2p += [item.ndp]
    list2n += [item.ndn]
  t.ndp = np.hstack(list2p)
  t.ndn = np.hstack(list2n)
  return t 
 def zeros(v):
  t = xlnsnpr("")
  t.ndp = np.zeros(v,dtype="i8")
  t.ndn = np.zeros(v,dtype="i8")
  return t
 def sum(self,axis=None):
  if axis == None:
    a = 1.0*xlnsnpr.ravel(self)
  elif axis == 1:
    a = 1.0*xlnsnpr.transpose(self)
  else:
    a = 1.0*self
  #print('sum shape='+str(xlnsnp.shape(a)))
  if len(a)==1:
    return a
  mid = 2**int(np.floor(math.log(len(a)-1,2)))
  end = len(a)
  while mid > 0:
    a[0:end-mid] += a[mid:end]
    end = mid
    mid //= 2
  return(a[0])
 #probably shouldn't be implemented
 def prod(self,axis=None):
  if axis == None:
    a = 1.0*xlnsnp.ravel(self)
  elif axis == 1:
    a = 1.0*xlnsnp.transpose(self)
  else:
    a = 1.0*self
  mid = 2**int(np.floor(math.log(len(a)-1,2)))
  end = len(a)
  while mid > 0:
    a[0:end-mid] *= a[mid:end]
    end = mid
    mid //= 2
  return(a[0])
 def ravel(v):
  t = xlnsnpr("")
  t.ndp = np.ravel(v.ndp)
  t.ndn = np.ravel(v.ndn)
  return t
 def shape(v):
  return np.shape(v.ndp)
 def transpose(v):
  t = xlnsnpr("")
  t.ndp = np.transpose(v.ndp)
  t.ndn = np.transpose(v.ndn)
  return t
 def size(v):
  return np.size(v.ndp)
 def max(self,axis=None):
  return xlnsnpr(np.max(self.xlns(),axis=axis))
 def min(self,axis=None):
  return xlnsnpr(np.min(self.xlns(),axis=axis))
 def argmax(self,axis=None):
  return np.argmax(self.xlns(),axis=axis)
 def reshape(self,shape):
  t = xlnsnpr("")
  t.ndp = np.reshape(self.ndp,shape)
  t.ndn = np.reshape(self.ndn,shape)
  return t 
 # v must be not be xlnsnpr
 def __matmul__(self,v):
  #if not isinstance(self, xlnsnpr):
  #   print("matmul not implemented")
  #   return NotImplemented
  if not isinstance(v,xlnsnp):
     print("converting v to xlnsnp for @")
     v = xlnsnp(v) #.xlns()) 
  t = xlnsnpr.zeros( (xlnsnpr.shape(self)[0],xlnsnp.shape(v)[1]) )
  vtrans = xlnsnp.transpose(v)
  for r in range(xlnsnpr.shape(self)[0]):
     t[r] = (self[r]*vtrans).sum(axis=1)
  return t
 # v must be not be xlnsnpr
 def __rmatmul__(self,v):
  print("rmatmul")
  if not isinstance(v,xlnsnp):
     print("converting v to xlnsnp for @")
     v = xlnsnp(v) #.xlns()) 
  t = xlnsnpr.zeros( (xlnsnp.shape(v)[0],xlnsnpr.shape(self)[1]) )
  selftrans = xlnsnpr.transpose(self)
  for r in range(xlnsnp.shape(v)[0]):
     t[r] = (selftrans*v[r]).sum(axis=1)
  return t
 def __abs__(self):
  t=xlnsnpr("")
  t.ndp = np.where(self.ndp>self.ndn, self.ndp, self.ndn)
  t.ndn = np.where(self.ndp>self.ndn, self.ndn, self.ndp)
  return t
 def __add__(self,v):
  if isinstance(v,int) or isinstance(v,float):
    return self+xlnsnp(v)
  if isinstance(v,xlnsnp):
    #print("left")#to do
    t = xlnsnpr("")
    t.ndp = np.where(((v.nd|1)==-0x7fffffffffffffff), #obscure way to say 0x8000000000000001 
              self.ndp,
              np.where((v.nd&1),
                 self.ndp,
                 np.maximum(self.ndp,v.nd//2) + sbdb_ufunc(-np.abs(v.nd//2 - self.ndp), 0 )//2))  
    t.ndn = np.where(((v.nd|1)==-0x7fffffffffffffff), #obscure way to say 0x8000000000000001 
              self.ndn,
              np.where((v.nd&1),
                 np.maximum(self.ndn,v.nd//2) + sbdb_ufunc(-np.abs(v.nd//2 - self.ndn), 0 )//2,  
                 self.ndn))
    return t
  if isinstance(v,xlnsnpr):
    #print("both")
    t = xlnsnpr("")
    #t.ndp = np.maximum(self.ndp,v.ndp) + sbdb_ufunc(-np.abs(v.ndp - self.ndp), 0 )//2  
    #t.ndn = np.maximum(self.ndn,v.ndn) + sbdb_ufunc(-np.abs(v.ndn - self.ndn), 0 )//2  
    t.ndp = np.where(self.ndp==self.ndn,
                     v.ndp,
                     np.where(v.ndp==v.ndn,
                              self.ndp,
                              np.maximum(self.ndp,v.ndp) + sbdb_ufunc(-np.abs(v.ndp - self.ndp), 0 )//2 ))  
    t.ndn = np.where(self.ndp==self.ndn,
                     v.ndn,
                     np.where(v.ndp==v.ndn,
                              self.ndn,
                              np.maximum(self.ndn,v.ndn) + sbdb_ufunc(-np.abs(v.ndn - self.ndn), 0 )//2 ))
    return t
  else:
    #print("here")
    return self+xlnsnp(v)
 def __lt__(self,v):
  #if isinstance(v,int) or isinstance(v,float) or isinstance(v,xlns) or isinstance(v,xlnsr) or isinstance(v,xlnsnp):
  if not isinstance(v,xlnsnpr):
   return self<xlnsnpr(v)
  t = self-v
  return xlnsnpr.zeros(self.shape())+(t.ndp<t.ndn) 
 def __gt__(self,v):
  #if isinstance(v,int) or isinstance(v,float) or isinstance(v,xlns) or isinstance(v,xlnsr) or isinstance(v,xlnsnp):
  if not isinstance(v,xlnsnpr):
   return self>xlnsnpr(v)
  t = self-v
  return xlnsnpr.zeros(self.shape())+(t.ndp>t.ndn) 
 def __ge__(self,v):
  #if isinstance(v,int) or isinstance(v,float) or isinstance(v,xlns) or isinstance(v,xlnsr) or isinstance(v,xlnsnp):
  if not isinstance(v,xlnsnpr):
   return self>=xlnsnpr(v)
  t = self-v
  return xlnsnpr.zeros(self.shape())+(t.ndp>=t.ndn) 
 def __le__(self,v):
  #if isinstance(v,int) or isinstance(v,float) or isinstance(v,xlns) or isinstance(v,xlnsr) or isinstance(v,xlnsnp):
  if not isinstance(v,xlnsnpr):
   return self<=xlnsnpr(v)
  t = self-v
  return xlnsnpr.zeros(self.shape())+(t.ndp<=t.ndn) 
 def __eq__(self,v):
  #if isinstance(v,int) or isinstance(v,float) or isinstance(v,xlns) or isinstance(v,xlnsr) or isinstance(v,xlnsnp):
  if not isinstance(v,xlnsnpr):
   return self==xlnsnpr(v)
  t = self-v
  return xlnsnpr.zeros(self.shape())+(t.ndp==t.ndn) 
 def __ne__(self,v):
  #if isinstance(v,int) or isinstance(v,float) or isinstance(v,xlns) or isinstance(v,xlnsr) or isinstance(v,xlnsnp):
  if not isinstance(v,xlnsnpr):
   return self!=xlnsnpr(v)
  t = self-v
  return xlnsnpr.zeros(self.shape())+(t.ndp!=t.ndn) 

  #t.nd = self.nd + sbdb_ufunc(v.nd//2 - self.nd//2, (self.nd^v.nd)&1)  
  #t.nd = np.maximum(self.nd,v.nd) + sbdb_ufunc(-np.abs(v.nd//2 - self.nd//2), (self.nd^v.nd)&1)  


xlnsud__add__ = lambda self,v: xlnsud(xlns.__add__(self,v)) #to be replaced

class xlnsud(xlns):
 """user-defined subclass of xlns; allows xlnsud__add__ to be changed from Gaussian log to frac norm"""
 def __repr__(self):
  return "xlnsud("+str(self)+")"
 def __abs__(self):
  return xlnsud(xlns.__abs__(self))
 def exp(x):
  return xlnsud(xlns.exp(x))
 def log(x):
  return xlnsud(xlns.log(x))
 def __mul__(self,v):
  return xlnsud(xlns.__mul__(self,v))
 def __pow__(self,v):
  return xlnsud(xlns.__pow__(self,v))
 def __truediv__(self,v):
  return xlnsud(xlns.__truediv__(self,v))
 def __add__(self,v):
  return xlnsud__add__(self,v) #to be replaced
 def __neg__(self):
  return xlnsud(xlns.__neg__(self))
 def __pos__(self):
  return self
 def __sub__(self,v):
  return self.__add__(-v)
 def __radd__(self,v):
  return self.__add__(v)
 def __rsub__(self,v):
  return (-self).__add__(v)
 def __rmul__(self,v):
  return self.__mul__(v)
 def __rtruediv__(self,v):
  if isinstance(v,xlnsud):
    return v/self
  return xlnsud(v)/self
 def __rpow__(self,v):
  if isinstance(v,xlnsud):
    return v**self
  return xlnsud(v)**self
 def __iadd__(self,v):
  self = self.__add__(v)
  return self
 def __isub__(self,v):
  self = self.__sub__(v)
  return self
 def __imul__(self,v):
  self = self.__mul__(v)
  return self
 def __idiv__(self,v):
  self = self.__div__(v)
  return self
 def __ipow__(self,v):
  self = self.__pow__(v)
  return self

class xlnsv:
 """scalar non-redundant LNS using distinct int precision F (unrelated to xlnsF) on each instance"""
 def __init__(self,v,setF=None):
  #global xlns1stCons
  #xlns1stCons = False
  #later use of xlnsv does not depend on possibly changed xlnsF,xlnsB
  if setF==None:
   self.F = xlnsF
  else:
   self.F = setF
  #self.B = 2.0**(2**(-self.F)) #distribute below
  if isinstance(v,int) or isinstance(v,float):
   self.B = 2.0**(2**(-self.F)) #from above
   if abs(v)!=0:
    self.x = int(round(math.log(abs(v))/math.log(self.B)))
   else:
    self.x = -1e1000 #-inf
   self.s = False if v>=0.0 else True 
  elif isinstance(v,xlnsv):  # 01/21/24 copy constructor!!!!
   if v.F == self.F or v.x == -1e1000:  #zero (rep by -inf) same for all F
     self.x = v.x
     self.s = v.s
     self.B = v.B                   #avoid recompute 
   elif v.F < self.F:
     self.x = v.x << (self.F-v.F)
     self.s = v.s
     self.B = 2.0**(2**(-self.F)) #from above
   elif v.F-1 == self.F:
     self.x = v.x >> 1              #v.F-self.F == 1
     self.s = v.s
     self.B = 2.0**(2**(-self.F)) #from above
   else:
     self.x = (v.x + (1<<(v.F-self.F-1))) >> (v.F-self.F) #rnd
     self.s = v.s
     self.B = 2.0**(2**(-self.F)) #from above
  #else:
  #  temp = xlnsv(float(v),setF)   #works, but too expensive
  #  self.x = temp.x               #equiv to above cases
  #  self.s = temp.s
  #  self.F = temp.F
  #  self.B = temp.B
  #elif isinstance(v,xlns) or isinstance(v,xlnsr):  # convert from xlnsF to self.F
  # temp = xlnsv(float(v),setF)     #could optimize like above but assume unusual case
  # self.x = temp.x                 #so speed doesn't matter
  # self.s = temp.s
  # self.F = temp.F
  # self.B = temp.B
  elif isinstance(v,str):
   if v.replace(".","",1).replace("+","",2).replace("-","",2).replace("e","",1).isdigit():
    temp = xlnsv(float(v),setF)
    self.x = temp.x
    self.s = temp.s
    self.F = temp.F
    self.B = temp.B
   else:
    self.x = v
    self.s = False
    self.B = 2.0**(2**(-self.F)) #from above
  #elif isinstance(v,xlnsr):
  #  temp = xlnsv(float(v),setF)
  #  self.x = temp.x
  #  self.s = temp.s
  #  self.F = temp.F
  #  self.B = temp.B
  elif isinstance(v,xlnsnp) or isinstance(v,xlnsnpr) or isinstance(v,xlnsnpv) or isinstance(v,xlnsnpb):
    temp = v.xlns()
    if v.size()==1:
       temp = xlnsv(temp[0],setF)
       self.x = temp.x
       self.s = temp.s
       self.F = temp.F
       self.B = temp.B
    else:
       print("cannot cast non-scalar xlnsnp[r] as xlns")
       return NotImplemented
  else:
   # new case that handles floatable types 9/20/24
   # elimnates xlns and xlnsr cases above
   #self.x = v
   #self.s = False
   temp = xlnsv(float(v),setF)     
   self.x = temp.x                 
   self.s = temp.s
   self.F = temp.F
   self.B = temp.B
 def __float__(self):
  return (-1 if self.s else 1) * float(self.B**self.x)
 def __int__(self):
  return int(float(self))
 def __str__(self):
  if self.x == -1e1000:
   return "0"
  log10 = self.x/(math.log(10)/math.log(self.B))
  #print("log10=",log10)
  if abs(log10)>10:
   return ("-" if self.s else "")+str(10.0**(log10%1))+"E"+str(math.floor(log10))
  else: 
   return str(float(self))
 def __repr__(self):
  return "xlnsv("+str(self)+" F="+str(self.F)+")"
 def conjugate(self):  #needed for .var() and .std() in numpy
  return(self)
 def __abs__(self):
  t=xlnsv("",self.F)
  t.x = self.x
  t.s = False
  return t
 def exp(x):
  if isinstance(x,float) or isinstance(x,int):
   return xlnsv(math.exp(x))
  if not isinstance(x,xlnsv):
   return xlnsv(math.exp(float(x)))
  return xlnsv(math.exp(float(x)),x.F)
 def log(x):
  if isinstance(x,float) or isinstance(x,int):
   return xlnsv(math.log(x))
  if not isinstance(x,xlnsv):
   return xlnsv(math.log(float(x)))
  return xlnsv(math.log(float(x)),x.F)
 def __mul__(self,v):
  t=xlnsv("",self.F)
  if isinstance(v,int) or isinstance(v,float):
   return self*xlnsv(v,self.F)
  if isinstance(v,xlnsnp) or isinstance(v,xlnsnpr) or isinstance(v,xlnsnpb):
   return v*self
  if not isinstance(v,xlnsv) or self.F != v.F:
   return self*xlnsv(v,self.F)
  t.x = self.x + v.x
  t.s = self.s != v.s
  return t
 def __pow__(self,v):
  t=xlnsv("",self.F)
  if isinstance(v,int):
    t.x = self.x*v
    t.s = self.s if v%2==1 else False
  elif isinstance(v,float):
    if self.s:
      return float(self)**v  #let it become Complex float; lets -1**2.0 slip thru to float; someday CLNS
    t.x = self.x*v
    t.s = False
  else:
    if self.s:
      return float(self)**float(v)  #let it become Complex float; someday CLNS  
    t.x = self.x*float(v)
    t.s = False
  return t
 def __truediv__(self,v):
  t=xlnsv("",self.F)
  if isinstance(v,int) or isinstance(v,float):
   return self/xlnsv(v,self.F)
  if isinstance(v,xlnsnp) or isinstance(v,xlnsnpr) or isinstance(v,xlnsnpb):
   return (1/v)*self
  if not isinstance(v,xlnsv) or self.F != v.F:
   return self/xlnsv(v,self.F)
  t.x = self.x - v.x
  t.s = self.s != v.s
  return t
 def __add__(self,v):
  def sbexact(z):
   return int(round(math.log(1.0 + self.B**z)/math.log(self.B)))
  def dbexact(z):
   if z==0:
    return -1e1000  #-inf
   else:
    return int(round(math.log(abs(1.0 - self.B**z))/math.log(self.B)))
  def sb(z):
     if z<=0:
       #print(('-',sbdb_ufunc(z,0,B=self.B,F=self.F)//2,sbexact(z)))
       return int(sbdb_ufunc(z,0,B=self.B,F=self.F)//2)
     else:
       #print(('+',sbdb_ufunc(-z,0,B=self.B)//2+z,sbexact(z)))
       return int(sbdb_ufunc(-z,0,B=self.B,F=self.F)//2+z)
  def db(z):
     if z==0:
       return -1e1000  #-inf
     elif z<0:
       return int(sbdb_ufunc(z,1,B=self.B,F=self.F)//2)
     else:
       return int(sbdb_ufunc(-z,1,B=self.B,F=self.F)//2+z)
  t=xlnsv("",self.F)
  if isinstance(v,int) or isinstance(v,float):
   return self+xlnsv(v,self.F)
  elif isinstance(v,xlnsnp) or isinstance(v,xlnsnpr) or isinstance(v,xlnsnpb):
   return v+self
  elif not isinstance(v,xlnsv) or self.F != v.F:
   return self+xlnsv(v,self.F)
  if v.x==-1e1000:  #-inf
   return self
  if self.x==-1e1000:  #-inf
   return v
  z = self.x - v.x
  if self.s == v.s:
   t.x = v.x + sb(z)
   t.s = v.s
  else:
   t.x = v.x + db(z)
   t.s = v.s if v.x>self.x else self.s
  return t
 def __neg__(self):
  t=xlnsv("",self.F)
  t.x = self.x
  t.s = False if self.s else True
  return t
 def __pos__(self):
  return self
 def __sub__(self,v):
  return self.__add__(-v)
 def __radd__(self,v):
  return self.__add__(v)
 def __rsub__(self,v):
  return (-self).__add__(v)
 def __rmul__(self,v):
  return self.__mul__(v)
 def __rtruediv__(self,v):
  if isinstance(v,int) or isinstance(v,float):
   return xlnsv(v,self.F)/self
  if not isinstance(v,xlnsv) or self.F != v.F:
   return xlnsv(v,self.F)/self
  return v/self  
 def __rpow__(self,v):
  if isinstance(v,int) or isinstance(v,float):
   return xlnsv(v,self.F)**self
  if not isinstance(v,xlnsv) or self.F != v.F:
   return xlnsv(v,self.F)**self
  return v**self  
 def __iadd__(self,v):
  self = self.__add__(v)
  return self
 def __isub__(self,v):
  self = self.__sub__(v)
  return self
 def __imul__(self,v):
  self = self.__mul__(v)
  return self
 def __idiv__(self,v):
  self = self.__div__(v)
  return self
 def __ipow__(self,v):
  self = self.__pow__(v)
  return self
 def __eq__(self,v):
  if not isinstance(v,xlnsv) or self.F != v.F:
   return self==xlnsv(v,self.F)
  return (self.x==v.x) and (self.s==v.s)
 def __ne__(self,v):
  if not isinstance(v,xlnsv) or self.F != v.F:
   return self!=xlnsv(v,self.F)
  return (self.x!=v.x) or (self.s!=v.s)
 def __lt__(self,v):
  if not isinstance(v,xlnsv) or self.F != v.F:
   return self<xlnsv(v,self.F)
  if self.s:
   if v.s:
    return self.x > v.x
   else:
    return True
  else:  
   if v.s:
    return False
   else:
    return self.x < v.x
 def __gt__(self,v):
  if not isinstance(v,xlnsv) or self.F != v.F:
   return self>xlnsv(v,self.F)
  if self.s:
   if v.s:
    return self.x < v.x
   else:
    return False
  else:  
   if v.s:
    return True
   else:
    return self.x > v.x
 def __le__(self,v):
  if not isinstance(v,xlnsv) or self.F != v.F:
   return self<=xlnsv(v,self.F)
  if self.s:
   if v.s:
    return self.x >= v.x
   else:
    return True
  else:  
   if v.s:
    return False
   else:
    return self.x <= v.x
 def __ge__(self,v):
  if not isinstance(v,xlnsv) or self.F != v.F:
   return self>=xlnsv(v,self.F)
  if self.s:
   if v.s:
    return self.x <= v.x
   else:
    return False
  else:  
   if v.s:
    return True
   else:
    return self.x >= v.x



class xlnsnpv(xlnsnp):
 """numpy-like array of non-redundant LNS using int precision f <= xlnsF on each instance"""
 def __init__(self,v,setF=None):
  global xlns1stCons
  xlns1stCons = False
  if setF == None:
    setF = xlnsF
  if setF < xlnsF:
    self.f = setF
  else:
    self.f = xlnsF
  if not isinstance(v,xlnsnp):   # incl. npv
    v = xlnsnp(v)
  if (self.f >= xlnsF) or (len(v.nd)==0):
    self.nd = v.nd
  else:
    #self.nd = (v.nd) &((-1<<(xlnsF-self.f+1))|1) #truncation, not recommended
    self.nd = (v.nd+(1<<(xlnsF-self.f)))&((-1<<(xlnsF-self.f+1))|1) #rounding 
 #following not found in numpy
 def xlns(self):
  """convert xlnsnpv to similar-shaped list of xlnsv"""
  t=[]
  for y in xlnsnp.xlns(self.ravel()):
     t += [xlnsv(y,self.f)]
  return np.reshape(t,np.shape(self.nd)) 
 def ovf(x):
  """return copy except for values that underflow/overflow, which are clipped according to xlnssetovf"""
  global ovfmin,ovfmax
  return xlnsnpv.where((abs(x)>=ovfmin)*(abs(x)<ovfmax), x, x*xlnsnpv.zeros(x.shape())+(abs(x)>=ovfmax)*ovfmax*x.sign())
 #end of those not found in numpy
 def __repr__(self):
  return "xlnsnpv("+str(np.float64(self.xlns()))+" f="+str(self.f)+")"
 def exp(x):
  if isinstance(x,float) or isinstance(x,int):
   return xlnsnpv(math.exp(x))
  if isinstance(x,xlnsnpv):
   return xlnsnpv(math.exp(1)**x,x.f)
  if isinstance(x,xlnsnp):
   return math.exp(1)**x
 def log(x):
  if isinstance(x,float) or isinstance(x,int):
   return xlnsnpv(math.log(x))
  if isinstance(x,xlnsnpv):
   return xlnsnpv(np.log(x.xlns()),x.f)
  if isinstance(x,xlnsnp):
   return xlnsnpv(np.log(x.xlns()))
 def __mul__(self,v):
  if isinstance(v,xlnsnp):      #incl. npv
    return xlnsnpv(xlnsnp.__mul__(self,v),self.f)
  else:
    return self*xlnsnpv(v,self.f)
 def __truediv__(self,v):
  if isinstance(v,xlnsnp):      #incl. npv
    return xlnsnpv(xlnsnp.__truediv__(self,v),self.f)
  else:
    return self/xlnsnpv(v,self.f)
 #def __neg__(self):
 # t=self*-1
 # return t
 #def __pos__(self):
 # return self
 def __sub__(self,v):
  return self.__add__(-v)
 def __radd__(self,v):
  return self.__add__(xlnsnpv(v,self.f))
 def __rsub__(self,v):
  return (-self).__add__(xlnsnpv(v,self.f))
 def __rmul__(self,v):
  return self.__mul__(xlnsnpv(v,self.f))
 def __rtruediv__(self,v):
  return xlnsnpv(xlnsnp.__truediv__(xlnsnp(v),self),self.f)
 def __pow__(self,v):
  t=xlnsnpv("")
  t.f = self.f
  if isinstance(v,int):
   if (v&1)==0:
     t.nd = ((self.nd&-2)*v)&-2
   else:
     t.nd = (((self.nd&-2)*v)&-2)|(self.nd&1) 
   return t
   #return xlnsnpv(self.xlns()**vi,self.f)
  if isinstance(v,float):
   t.nd = np.int64((self.nd&-2)*v)&-2
   return t
  return xlnsnpv.exp(xlnsnpv.log(self)*v)
 def __rpow__(self,v):
  t=xlnsnpv("")
  t.f = self.f
  if isinstance(v,xlnsnp):
   t.nd = (v**self).nd  
  elif isinstance(v,float) or isinstance(v,int):
   t.nd=(2-4*(self.nd&1))*np.int64(xlnsB**((self.nd//2))*math.log(v,xlnsB))
  else:
   t.nd = (v**self).nd
  return t
 #non-ufunc 
 #def __len__(self):
 # return self.nd.__len__()
 def __getitem__(self,v):
  t=xlnsnpv("")
  t.f = self.f
  t.nd = self.nd.__getitem__(v)
  return t
 def __setitem__(self,v1,v2):
  self.nd.__setitem__(v1,xlnsnpv(v2,self.f).nd)
 def concatenate(list1):
  t = xlnsnpv("")
  t.f = list1[0].f
  list2 = []
  for item in list1:
    list2 += [item.nd]
  t.nd = np.concatenate(list2)
  return t 
 def vstack(list1):
  t = xlnsnpv("")
  t.f = list1[0].f
  list2 = []
  for item in list1:
    list2 += [item.nd]
  t.nd = np.vstack(list2)
  return t 
 def hstack(list1):
  t = xlnsnpv("")
  t.f = list1[0].f
  list2 = []
  for item in list1:
    list2 += [item.nd]
  t.nd = np.hstack(list2)
  return t 
 #def zeros(v):
 # t = xlnsnpv("")
 # t.nd = np.zeros(v,dtype="i8")+(-0x7fffffffffffffff-1)
 # return t
 def sum(self,axis=None):
  if axis == None:
    a = 1.0*xlnsnpv.ravel(self)
  elif axis == 1:
    a = 1.0*xlnsnpv.transpose(self)
  else:
    a = 1.0*self
  if len(a)==1:
    return a
  mid = 2**int(np.floor(math.log(len(a)-1,2)))
  end = len(a)
  while mid > 0:
    a[0:end-mid] += a[mid:end]
    end = mid
    mid //= 2
  return(a[0])
 def prod(self,axis=None):
  if axis == None:
    a = 1.0*xlnsnpv.ravel(self)
  elif axis == 1:
    a = 1.0*xlnsnpv.transpose(self)
  else:
    a = 1.0*self
  mid = 2**int(np.floor(math.log(len(a)-1,2)))
  end = len(a)
  while mid > 0:
    a[0:end-mid] *= a[mid:end]
    end = mid
    mid //= 2
  return(a[0])
 def ravel(v):
  t = xlnsnpv("")
  t.f = v.f
  t.nd = np.ravel(v.nd)
  return t
 #def shape(v):
 # return np.shape(v.nd)
 def transpose(v):
  t = xlnsnpv("")
  t.f = v.f
  t.nd = np.transpose(v.nd)
  return t
 #def size(v):
 # return np.size(v.nd)
 def max(self,axis=None):
  return xlnsnpv(np.max(self.xlns(),axis=axis),self.f)
 def min(self,axis=None):
  return xlnsnpv(np.min(self.xlns(),axis=axis),self.f)
 #def argmax(self,axis=None):
 # return np.argmax(self.xlns(),axis=axis)
 def reshape(self,shape):
  t = xlnsnpv("")
  t.f = self.f
  t.nd = np.reshape(self.nd,shape)
  return t 
 def __matmul__(self,v):
  if isinstance(v, xlnsnpr):
     print("xlnsnpv@xlnsnpr not implemented by xlnsnp")
     return NotImplemented
  t = xlnsnpv(xlnsnp.zeros( (xlnsnp.shape(self)[0],xlnsnp.shape(v)[1]) ), self.f)
  vtrans = xlnsnp.transpose(v)
  for r in range(xlnsnp.shape(self)[0]):
     t[r] = (self[r]*vtrans).sum(axis=1)
  return t
 def __abs__(self):
  return xlnsnpv(xlnsnp.__abs__(self),self.f)
 def __add__(self,v):
  if isinstance(v,xlnsnp):      #incl. npv
    return xlnsnpv(xlnsnp.__add__(self,v),self.f)
  else:
    return self+xlnsnpv(v,self.f)


class xlnsb:
 """scalar non-redundant LNS whose precision on each instance defined by arbitrary real B>1"""
 def __init__(self,v,setB):
  #later use of xlnsb does not depend on possibly changed xlnsF
  self.B = setB 
  if isinstance(v,int) or isinstance(v,float):
   if abs(v)!=0:
    self.x = int(round(math.log(abs(v))/math.log(self.B)))
   else:
    self.x = -1e1000 #-inf
   self.s = False if v>=0.0 else True 
  elif isinstance(v,xlnsb):  # 10/18/24 copy constructor has to convert base
     cv = np.log(v.B)/np.log(self.B)
     self.x = v.x if v.x == -1e1000 else int(cv*v.x+0.5)
     self.s = v.s
  elif isinstance(v,str):
   if v.replace(".","",1).replace("+","",2).replace("-","",2).replace("e","",1).isdigit():
    temp = xlnsb(float(v),self.B)
    self.x = temp.x
    self.s = temp.s
    self.B = temp.B
   else:
    self.x = v
    self.s = False
  elif isinstance(v,xlnsnp) or isinstance(v,xlnsnpr) or isinstance(v,xlnsnpv) or isinstance(v,xlnsnpb):
    temp = v.xlns()
    if v.size()==1:
       temp = xlnsb(float(temp[0]),self.B)
       self.x = temp.x
       self.s = temp.s
       self.B = temp.B
    else:
       print("cannot cast non-scalar xlnsnp[r] as xlnsb")
       return NotImplemented
  elif isinstance(v,list):
    print("cannot cast list as xlnsb")
    return NotImplemented
  else:
   # new case that handles floatable types 9/20/24
   # elimnates xlns and xlnsr cases above
   temp = xlnsb(float(v),self.B)     
   self.x = temp.x                 
   self.s = temp.s
   self.B = temp.B
 def __float__(self):
  return (-1 if self.s else 1) * float(self.B**self.x)
 def __int__(self):
  return int(float(self))
 def __str__(self):
  if self.x == -1e1000:
   return "0"
  log10 = self.x/(math.log(10)/math.log(self.B))
  #print("log10=",log10)
  if abs(log10)>10:
   return ("-" if self.s else "")+str(10.0**(log10%1))+"E"+str(math.floor(log10))
  else: 
   return str(float(self))
 def __repr__(self):
  return "xlnsb("+str(self)+" B="+str(self.B)+")"
 def conjugate(self):  #needed for .var() and .std() in numpy
  return(self)
 def __abs__(self):
  t=xlnsb("",self.B)
  t.x = self.x
  t.s = False
  return t
 def exp(x):
  if isinstance(x,float) or isinstance(x,int):
   return xlnsb(math.exp(x),xlnsB)
  if not isinstance(x,xlnsb):
   return xlnsb(math.exp(float(x)),xlnsB)
  return xlnsb(math.exp(float(x)),x.B)
 def log(x):
  if isinstance(x,float) or isinstance(x,int):
   return xlnsb(math.log(x),xlnsB)
  if not isinstance(x,xlnsb):
   return xlnsb(math.log(float(x)),xlnsB)
  return xlnsb(math.log(float(x)),x.B)
 def __mul__(self,v):
  t=xlnsb("",self.B)
  if isinstance(v,int) or isinstance(v,float):
   return self*xlnsb(v,self.B)
  if isinstance(v,xlnsnp) or isinstance(v,xlnsnpr) or isinstance(v,xlnsnpb):
   return v*self
  if not isinstance(v,xlnsb) or self.B != v.B:
   return self*xlnsb(v,self.B)
  t.x = self.x + v.x
  t.s = self.s != v.s
  return t
 def __pow__(self,v):
  t=xlnsb("",self.B)
  if isinstance(v,int):
    t.x = self.x*v
    t.s = self.s if v%2==1 else False
  elif isinstance(v,float):
    if self.s:
      return float(self)**v  #let it become Complex float; lets -1**2.0 slip thru to float; someday CLNS
    t.x = self.x*v
    t.s = False
  else:
    if self.s:
      return float(self)**float(v)  #let it become Complex float; someday CLNS  
    t.x = self.x*float(v)
    t.s = False
  return t
 def __truediv__(self,v):
  t=xlnsb("",self.B)
  if isinstance(v,int) or isinstance(v,float):
   return self/xlnsb(v,self.B)
  if isinstance(v,xlnsnp) or isinstance(v,xlnsnpr) or isinstance(v,xlnsnpb):
   return (1/v)*self
  if not isinstance(v,xlnsb) or self.B != v.B:
   return self/xlnsb(v,self.B)
  t.x = self.x - v.x
  t.s = self.s != v.s
  return t
 def __add__(self,v):
  def sbprevious(z):
   return int(round(math.log(1.0 + self.B**z)/math.log(self.B)))
  def dbprevious(z):
   if z==0:
    return -1e1000  #-inf
   else:
    return int(round(math.log(abs(1.0 - self.B**z))/math.log(self.B)))
  def sbexact(z):
   return int(round(math.log(1.0 + self.B**z)/math.log(self.B)))
  def dbexact(z):
   if z==0:
    return -1e1000  #-inf
   else:
    return int(round(math.log(abs(1.0 - self.B**z))/math.log(self.B)))
  def sb(z):
     if z<=0:
       return int(sbdb_ufunc(z,0,B=self.B)//2)
     else:
       return int(sbdb_ufunc(-z,0,B=self.B)//2+z)
  def db(z):
     if z==0:
       return -1e1000  #-inf
     elif z<0:
       return int(sbdb_ufunc(z,1,B=self.B)//2)
     else:
       return int(sbdb_ufunc(-z,1,B=self.B)//2+z)
  t=xlnsb("",self.B)
  if isinstance(v,int) or isinstance(v,float):
   return self+xlnsb(v,self.B)
  elif isinstance(v,xlnsnp) or isinstance(v,xlnsnpr) or isinstance(v,xlnsnpb):
   return v+self
  elif not isinstance(v,xlnsb) or self.B != v.B:
   return self+xlnsb(v,self.B)
  if v.x==-1e1000:  #-inf
   return self
  if self.x==-1e1000:  #-inf
   return v
  z = self.x - v.x
  if self.s == v.s:
   t.x = v.x + sb(z)
   t.s = v.s
  else:
   t.x = v.x + db(z)
   t.s = v.s if v.x>self.x else self.s
  return t
 def __neg__(self):
  t=xlnsb("",self.B)
  t.x = self.x
  t.s = False if self.s else True
  return t
 def __pos__(self):
  return self
 def __sub__(self,v):
  return self.__add__(-v)
 def __radd__(self,v):
  return self.__add__(v)
 def __rsub__(self,v):
  return (-self).__add__(v)
 def __rmul__(self,v):
  return self.__mul__(v)
#need another 100 lines
 def __rtruediv__(self,v):
  if isinstance(v,int) or isinstance(v,float):
   return xlnsb(v,self.B)/self
  if not isinstance(v,xlnsb) or self.B != v.B:
   return xlnsb(v,self.B)/self
  return v/self  
 def __rpow__(self,v):
  if isinstance(v,int) or isinstance(v,float):
   return xlnsb(v,self.B)**self
  if not isinstance(v,xlnsb) or self.B != v.B:
   return xlnsv(v,self.B)**self
  return v**self  
 def __iadd__(self,v):
  self = self.__add__(v)
  return self
 def __isub__(self,v):
  self = self.__sub__(v)
  return self
 def __imul__(self,v):
  self = self.__mul__(v)
  return self
 def __idiv__(self,v):
  self = self.__div__(v)
  return self
 def __ipow__(self,v):
  self = self.__pow__(v)
  return self
 def __eq__(self,v):
  if not isinstance(v,xlnsb) or self.B != v.B:
   return self==xlnsb(v,self.B) #some hidden infinite recursion
  return (self.x==v.x) and (self.s==v.s)
 def __ne__(self,v):
  if not isinstance(v,xlnsb) or self.B != v.B:
   return self!=xlnsb(v,self.B)
  return (self.x!=v.x) or (self.s!=v.s)
 def __lt__(self,v):
  if not isinstance(v,xlnsb) or self.B != v.B:
   return self<xlnsb(v,self.B)
  if self.s:
   if v.s:
    return self.x > v.x
   else:
    return True
  else:  
   if v.s:
    return False
   else:
    return self.x < v.x
 def __gt__(self,v):
  if not isinstance(v,xlnsb) or self.B != v.B:
   return self>xlnsb(v,self.B)
  if self.s:
   if v.s:
    return self.x < v.x
   else:
    return False
  else:  
   if v.s:
    return True
   else:
    return self.x > v.x
 def __le__(self,v):
  if not isinstance(v,xlnsb) or self.B != v.B:
   return self<=xlnsb(v,self.B)
  if self.s:
   if v.s:
    return self.x >= v.x
   else:
    return True
  else:  
   if v.s:
    return False
   else:
    return self.x <= v.x
 def __ge__(self,v):
  if not isinstance(v,xlnsb) or self.B != v.B:
   return self>=xlnsb(v,self.B)
  if self.s:
   if v.s:
    return self.x <= v.x
   else:
    return False
  else:  
   if v.s:
    return True
   else:
    return self.x >= v.x

class xlnsnpb:
 """numpy-like array of non-redundant LNS whose precision on each instance defined by real B>1"""
 def __init__(self,v,setB):
  #global xlns1stCons
  #xlns1stCons = False
  self.B = setB
  if isinstance(v,xlnsv) or isinstance(v,xlnsb) or isinstance(v,xlnsnpb):
    cv = np.log(v.B)/np.log(self.B)
  #elif isinstance(v,xlnsnpv) or isinstance(v,xlns) or isinstance(v,xlnsr) or isinstance(v,xlnsnp) or isinstance(v,xlnsnpr):
  else:
    cv = np.log(xlnsB)/np.log(self.B)
  if isinstance(v,int) or isinstance(v,float):
   #print("have scalar: "+str(v))
   if v == 0:
     self.nd = np.array([-0x7fffffffffffffff-1],dtype="i8")      #0x8000000000000000 avoiding int64 limit
   else:
     self.nd = np.array([2*xlnsb(v,setB).x+xlnsb(v,setB).s],dtype="i8") 
  elif isinstance(v,xlns) or isinstance(v,xlnsv) or isinstance(v,xlnsb):
   if v.x == -1e1000:
     self.nd = np.array([-0x7fffffffffffffff-1],dtype="i8")      #0x8000000000000000 avoiding int64 limit
   else:
     self.nd = np.array([2*int(cv*v.x+.5)+v.s],dtype="i8")       #convert with rounding
  elif isinstance(v,xlnsnp) or isinstance(v,xlnsnpb):            #copy constructor includes xlnsnpv
     self.nd = np.where(((v.nd|1)==-0x7fffffffffffffff),         #obscure way to say 0x8000000000000001 ie if ==0.0
                         v.nd//2,                                # leave 0 mag unchanged
                         np.int64(cv*(v.nd//2)+.5) )*2|(v.nd&1)  # else convert w/round mag !=0 mag, save sign
  elif isinstance(v,xlnsr):
   if v.p==v.n:
     self.nd = np.array([-0x7fffffffffffffff-1],dtype="i8")      #0x8000000000000000 avoiding int64 limit
   elif v.p>v.n:
     self.nd = np.array([2*int(cv*xlns(v).x+0.5)],dtype="i8")    #convert with round
   else:
     self.nd = np.array([2*int(cv*xlns(v).x+0.5)+1],dtype="i8")  #convert with round
  elif isinstance(v,xlnsnpr):
     temp = xlnsnpb(v.xlns(),setB)
     self.nd = temp.nd
  elif isinstance(v,np.ndarray):
   #print("have ndarray: "+str(v))
   #print("first elem: "+repr(np.ravel(v)[0]))
   #print(isinstance(np.ravel(v)[0],xlns))
   if isinstance(np.ravel(v)[0],xlns):    # or isinstance(np.ravel(v)[0],xlnsv) or isinstance(np.ravel(v)[0],xlnsb):
     t=[]
     for y in np.ravel(v):
        #print(y)
        if y.x == -1e1000:
           #print("zero")
           t += [ -0x7fffffffffffffff-1 + y.s] #0x8000000000000000 avoiding int64 limit
        else:
           t += [2*int(cv*y.x+.5) + y.s] 
     #print("t="+repr(t))
     self.nd = np.array(np.reshape(t,np.shape(v)),dtype="i8")
   elif isinstance(np.ravel(v)[0],xlnsv) or isinstance(np.ravel(v)[0],xlnsb):
     self.nd = xlnsnpb(np.float64(v),setB).nd
   else:
     t=[]
     for y in np.ravel(xlnscopy(v)):
        #print(y)
        if isinstance(y,xlns) or isinstance(y,xlnsv) or isinstance(y,xlnsb):
           if y.x == -1e1000:
             #print("zero")
             t += [ -0x7fffffffffffffff-1 + y.s] #0x8000000000000000 avoiding int64 limit
           else:
             t += [2*int(cv*y.x+.5) + y.s] 
           #print("t="+repr(t))
        else:
           if y == 0:
             #print("zero")
             t += [ -0x7fffffffffffffff-1 ] #0x8000000000000000 avoiding int64 limit
           else:
             t += [2*xlnsb(y,setB).x + xlnsb(y,setB).s] 
           #print("t="+repr(t))
     self.nd = np.array(np.reshape(t,np.shape(v)),dtype="i8")
  elif isinstance(v,list):
     t=[]
     for y in xlnscopy(v):
       #print(y)
       if y.x == -1e1000:
         #print("zero")
         t += [ -0x7fffffffffffffff-1 + y.s] #0x8000000000000000 avoiding int64 limit
       else:
         t += [2*int(cv*y.x+.5) + y.s] 
       #print("t="+repr(t))
     self.nd = np.array(np.reshape(t,np.shape(v)),dtype="i8")
  else:
   self.nd = np.array([]) 
 # weird test
 def add(self,v):
   return self+v
 def __str__(self):
   return str(self.xlns())
 def __repr__(self):
  return "xlnsnpb("+str(np.float64(self.xlns()))+" B="+str(self.B)+")"
 def __bool__(self):
  return bool(self.nd == 0)
 #following not found in numpy
 def xlns(self):
  """convert xlnsnpb to similar-shaped list of xlnsb"""
  t=[]
  for y in np.ravel(self.nd):
     txlns = xlnsb(0,self.B)
     if (y|1) == -0x7fffffffffffffff:   #obscure 0x8000000000000001 avoiding int64 limit
       txlns.x = -1e1000
     else:
       txlns.x = y//2
     txlns.s = True if y&1 else False
     t += [txlns]
  return np.reshape(t,np.shape(self.nd)) 
 def ovf(x):
  """return copy except for values that underflow/overflow, which are clipped according to xlnssetovf"""
  global ovfmin,ovfmax
  return xlnsnpb.where((abs(x)>=ovfmin)*(abs(x)<ovfmax), x, x*xlnsnpb.zeros(x.shape())+(abs(x)>=ovfmax)*ovfmax*x.sign())
 #end of those not found in numpy
 def conjugate(self):  #needed for .var() and .std() in numpy
  return(self)
#skip exp log
 def exp(x):
  if isinstance(x,float) or isinstance(x,int):
   return xlnsnpb(math.exp(x),xlnsB)
  if isinstance(x,xlnsnpb):
   #return math.exp(1)**x
   return xlnsnpb(np.exp(np.float64(x.xlns())),x.B)
 def log(x):
  if isinstance(x,float) or isinstance(x,int):
   return xlnsnpb(math.log(x),xlnsB)
  if isinstance(x,xlnsnpb):
   return xlnsnpb(np.log(np.float64(x.xlns())),x.B)
#resume here
# if isinstance(v,xlnsv) or isinstance(v,xlnsb) or isinstance(v,xlnsnpb):
#    cv = np.log(v.B)/np.log(self.B)
#  else:
#    cv = np.log(v.B)/np.log(xlnsB)
 def __mul__(self,v):
  if isinstance(v,int) or isinstance(v,float):
    return self*xlnsnpb(v,self.B)
  if isinstance(v,xlnsnpb):
    t = xlnsnpb("",self.B)
    if v.B == self.B:
        t.nd = np.where(((self.nd|1)==-0x7fffffffffffffff)|((v.nd|1)==-0x7fffffffffffffff), #obscure way to say 0x8000000000000001 and
                  (-0x7fffffffffffffff-1) |((self.nd^v.nd)&1),                        #0x8000000000000000 avoiding int64 limit
                  (self.nd + (v.nd - (v.nd&1))) ^ (v.nd&1) )
    else:
        cv = np.log(v.B)/np.log(self.B)
        t.nd = np.where(((self.nd|1)==-0x7fffffffffffffff)|((v.nd|1)==-0x7fffffffffffffff), #obscure way to say 0x8000000000000001 and
                  (-0x7fffffffffffffff-1) |((self.nd^v.nd)&1),                        #0x8000000000000000 avoiding int64 limit
                  (self.nd + np.int64(cv*(v.nd//2)+.5)*2) ^ (v.nd&1) )
    return t
  else:
    return self*xlnsnpb(v,self.B)
 def __truediv__(self,v):
  if isinstance(v,int) or isinstance(v,float):
    return self/xlnsnpb(v,self.B)
  if isinstance(v,xlnsnpb):
    t = xlnsnpb("",self.B)
    if v.B == self.B:
        t.nd = np.where(((self.nd|1)==-0x7fffffffffffffff), #obscure way to say 0x8000000000000001 and
                  (-0x7fffffffffffffff-1) |((self.nd^v.nd)&1),                        #0x8000000000000000 avoiding int64 limit
                  (self.nd - v.nd + (v.nd&1)) ^ (v.nd&1) )
    else:
        cv = np.log(v.B)/np.log(self.B)
        t.nd = np.where(((self.nd|1)==-0x7fffffffffffffff)|((v.nd|1)==-0x7fffffffffffffff), #obscure way to say 0x8000000000000001 and
                  (-0x7fffffffffffffff-1) |((self.nd^v.nd)&1),                        #0x8000000000000000 avoiding int64 limit
                  (self.nd - np.int64(cv*(v.nd//2)+.5)*2) ^ (v.nd&1) )
    return t
  else:
    return self/xlnsnpb(v,self.B)
 def __neg__(self):
  t=self*-1
  return t
 def __pos__(self):
  return self
 def __sub__(self,v):
  return self.__add__(-v)
 def __radd__(self,v):
  return self.__add__(xlnsnpb(v,self.B))
 def __rsub__(self,v):
  return (-self).__add__(xlnsnpb(v,self.B))
 def __rdiv__(self,v):
  return self.__div__(xlnsnpb(v,self.B))
 def __rmul__(self,v):
  return self.__mul__(xlnsnpb(v,self.B))
 def __rtruediv__(self,v):
  return xlnsnpb(v,self.B).__truediv__(self)
 def __pow__(self,v):
  t=xlnsnpb("",self.B)
  if isinstance(v,int):
   if (v&1)==0:
     t.nd = ((self.nd&-2)*v)&-2
   else:
     t.nd = (((self.nd&-2)*v)&-2)|(self.nd&1) 
   return t
   #return xlnsnpb(self.xlns()**v,self.B)
  if isinstance(v,float):
   t.nd = np.int64((self.nd&-2)*v)&-2
   return t
  return xlnsnpb.exp(xlnsnpb.log(self)*v) #xl.log?
 def __rpow__(self,v):
  t=xlnsnpb("",self.B)
  if isinstance(v,xlnsnpb):
   return v**self  
  elif isinstance(v,float) or isinstance(v,int):
   t.nd=(2-4*(self.nd&1))*np.int64(self.B**((self.nd//2))*math.log(v,self.B))
   return t
  else:
   return xlnsnpb(v,self.B)**self
 #non-ufunc 
 def __len__(self):
  return self.nd.__len__()
 def __getitem__(self,v):
  t=xlnsnpb("",self.B)
  t.nd = self.nd.__getitem__(v)
  return t
 def __setitem__(self,v1,v2):
  if isinstance(v2,xlnsnpb):
    self.nd.__setitem__(v1,v2.nd)
  else:
    self.nd.__setitem__(v1,xlnsnpb(v2,self.B).nd)
 def where(self,vt,vf):
  #print(type(self))
  #print(type(vt))
  #print(type(vf))
  if isinstance(vt,xlnsnpb) and isinstance(vf,xlnsnpb):
   t = xlnsnpb("",self.B)
   t.nd = np.where(self.nd==0, vt.nd, vf.nd)
   return t
  else:
   print("xlnsnp.where must match type")
   return None
 def sign(v):
  if isinstance(v,xlnsnpb):
   t = xlnsnpb("",v.B)
   t.nd = np.where( (v.nd|1)==-0x7fffffffffffffff, #obscure way to say 0x8000000000000001 
                    #0, 1-2*(v.nd&1))
                    v.nd, v.nd&1)
   return t
  else:
   return np.sign(v) # or v.sign()
 def concatenate(list1):
  list2 = []
  minB = 2.0
  for item in list1:
    minB = min(minB,item.B)
    list2 += [item.nd]
  t = xlnsnpb("",minB)
  t.nd = np.concatenate(list2)
  return t 
 def vstack(list1):
  minB = 2.0
  list2 = []
  for item in list1:
    minB = min(minB,item.B)
    list2 += [item.nd]
  t = xlnsnpb("",minB)
  t.nd = np.vstack(list2)
  return t 
 def hstack(list1):
  minB = 2.0
  list2 = []
  for item in list1:
    minB = min(minB,item.B)
    list2 += [item.nd]
  t = xlnsnpb("",minB)
  t.nd = np.hstack(list2)
  return t 
 def zeros(v):
  t = xlnsnpb("",xlnsB)  #arbitrary choice for precision
  t.nd = np.zeros(v,dtype="i8")+(-0x7fffffffffffffff-1)
  return t
 def sum(self,axis=None):
  if axis == None:
    a = 1.0*xlnsnpb.ravel(self)
  elif axis == 1:
    a = 1.0*xlnsnpb.transpose(self)
  else:
    a = 1.0*self
  #print('sum shape='+str(xlnsnp.shape(a)))
  if len(a)==1:
    return a
  mid = 2**int(np.floor(math.log(len(a)-1,2)))
  end = len(a)
  while mid > 0:
    a[0:end-mid] += a[mid:end]
    end = mid
    mid //= 2
  return(a[0])
 def prod(self,axis=None):
  if axis == None:
    a = 1.0*xlnsnpb.ravel(self)
  elif axis == 1:
    a = 1.0*xlnsnpb.transpose(self)
  else:
    a = 1.0*self
  mid = 2**int(np.floor(math.log(len(a)-1,2)))
  end = len(a)
  while mid > 0:
    a[0:end-mid] *= a[mid:end]
    end = mid
    mid //= 2
  return(a[0])
 def ravel(v):
  t = xlnsnpb("",v.B)
  t.nd = np.ravel(v.nd)
  return t
 def shape(v):
  return np.shape(v.nd)
 def transpose(v):
  t = xlnsnpb("",v.B)
  t.nd = np.transpose(v.nd)
  return t
 def size(v):
  return np.size(v.nd)
 def max(self,axis=None):
  return xlnsnpb(np.max(self.xlns(),axis=axis),self.B)
 def min(self,axis=None):
  return xlnsnpb(np.min(self.xlns(),axis=axis),self.B)
 def argmax(self,axis=None):
  return np.argmax(self.xlns(),axis=axis)
 def reshape(self,shape):
  t = xlnsnpb("",self.B)
  t.nd = np.reshape(self.nd,shape)
  return t 
 def __matmul__(self,v):
  if isinstance(v, xlnsnpr):
     print("xlnsnp@xlnsnpr not implemented by xlnsnp")
     return NotImplemented
  #cast precision by mult 0 mat by [0][0] element (assuming 2D only)
  t = self[0][0]*xlnsnpb.zeros( (xlnsnpb.shape(self)[0],xlnsnpb.shape(v)[1]) )
  vtrans = xlnsnpb.transpose(v)
  for r in range(xlnsnpb.shape(self)[0]):
     t[r] = (self[r]*vtrans).sum(axis=1)
  return t
 def __abs__(self):
  t=xlnsnpb("",self.B)
  t.nd = self.nd & -2  # way to say 0xfffffffffffffffe
  return t
 def __add__(self,v):
  if isinstance(v,int) or isinstance(v,float):
    return self+xlnsnpb(v,self.B)
  if isinstance(v,xlnsnpb):
    #print("both")
    t = xlnsnpb("",self.B)
    if v.B == self.B:
        t.nd = np.where(((self.nd|1)==-0x7fffffffffffffff), #obscure way to say 0x8000000000000001 
                  v.nd,
                  np.where(((v.nd|1)==-0x7fffffffffffffff), #obscure way to say 0x8000000000000001 
                           self.nd,
                           np.maximum(self.nd,v.nd) + sbdb_ufunc(-np.abs(v.nd//2 - self.nd//2), (self.nd^v.nd)&1, B=self.B) ))  
    else:
        vcv = np.int64(np.log(v.B)/np.log(self.B)*(v.nd//2)+.5)*2 | (v.nd&1)
        t.nd = np.where(((self.nd|1)==-0x7fffffffffffffff), #obscure way to say 0x8000000000000001 
                  v.nd,
                  np.where(((v.nd|1)==-0x7fffffffffffffff), #obscure way to say 0x8000000000000001 
                           self.nd,
                           np.maximum(self.nd,vcv) + sbdb_ufunc(-np.abs(vcv//2 - self.nd//2), (self.nd^v.nd)&1, B=self.B) ))  
    return t
  else:
    #print("here")
    return self+xlnsnpb(v,self.B)
 def __lt__(self,v):
  t=xlnsnpb("",self.B)
  if isinstance(v,int) or isinstance(v,float) or isinstance(v,xlns) or isinstance(v,xlnsr) or isinstance(v,xlnsv) or isinstance(v,xlnsb):
   if v==0:
     t.nd=np.where(self.nd&1,0,-0x7fffffffffffffff-1)
     return t
   elif v>0:
     t.nd=np.where(self.nd&1,0,np.where(self.nd<2*xlnsb(v,self.B).x,0,-0x7fffffffffffffff-1))
     return t
   else:
     t.nd=np.where(self.nd&1,np.where(self.nd>(2*xlnsb(v,self.B).x|1),0,-0x7fffffffffffffff-1),-0x7fffffffffffffff-1)
     return t
  else:
   print("nonscalar comparison")
   return None
 def __gt__(self,v):
  t=xlnsnpb("",self.B)
  if isinstance(v,int) or isinstance(v,float) or isinstance(v,xlns) or isinstance(v,xlnsr) or isinstance(v,xlnsv) or isinstance(v,xlnsb):
   if v==0:
     t.nd=np.where(self.nd&1,-0x7fffffffffffffff-1,np.where((self.nd|1)!=-0x7fffffffffffffff,0,-0x7fffffffffffffff-1))
     return t
   elif v>0:
     t.nd=np.where(self.nd&1,-0x7fffffffffffffff-1,np.where(self.nd>2*xlnsb(v,self.B).x,0,-0x7fffffffffffffff-1))
     return t
   else:
     t.nd=np.where(self.nd&1,np.where(self.nd<(2*xlnsb(v,self.B).x|1),0,-0x7fffffffffffffff-1),0)
     return t
  else:
   print("nonscalar comparison")
   return None
 def __ge__(self,v):
  t=xlnsnpb("",self.B)
  if isinstance(v,int) or isinstance(v,float) or isinstance(v,xlns) or isinstance(v,xlnsr) or isinstance(v,xlnsv) or isinstance(v,xlnsb):
   if v==0:
     t.nd=np.where(self.nd&1,-0x7fffffffffffffff-1,0)
     return t
   elif v>0:
     t.nd=np.where(self.nd&1,-0x7fffffffffffffff-1,np.where(self.nd>=2*xlnsb(v,self.B).x,0,-0x7fffffffffffffff-1))
     return t
   else:
     t.nd=np.where(self.nd&1,np.where(self.nd<=(2*xlnsb(v,self.B).x|1),0,-0x7fffffffffffffff-1),0)
     return t
  else:
   print("nonscalar comparison")
   return None
 def __le__(self,v):
  t=xlnsnpb("",self.B)
  if isinstance(v,int) or isinstance(v,float) or isinstance(v,xlns) or isinstance(v,xlnsr) or isinstance(v,xlnsv) or isinstance(v,xlnsb):
   if v==0:
     t.nd=np.where(self.nd&1,0,np.where((self.nd|1)!=-0x7fffffffffffffff,0,-0x7fffffffffffffff-1))
     return t
   elif v>0:
     t.nd=np.where(self.nd&1,0,np.where(self.nd<=2*xlnsb(v,self.B).x,0,-0x7fffffffffffffff-1))
     return t
   else:
     t.nd=np.where(self.nd&1,np.where(self.nd>=(2*xlnsb(v,self.B).x|1),0,-0x7fffffffffffffff-1),-0x7fffffffffffffff-1)
     return t
  else:
   print("nonscalar comparison")
   return None
 def __eq__(self,v):
  t=xlnsnpb("",self.B)
  if not isinstance(v,xlnsnpb):
   t.nd = np.where(self.nd == xlnsnpb(v,self.B).nd, 0, -0x7fffffffffffffff-1)
  elif self.B == v.B:
   t.nd = np.where(self.nd == v.nd, 0, -0x7fffffffffffffff-1)
  else:
   t.nd = np.where(self.nd == xlnsnpb(v,self.B).nd, 0, -0x7fffffffffffffff-1)
  return t
 def __ne__(self,v):
  t=xlnsnpb("",self.B)
  if not isinstance(v,xlnsnpb):
   t.nd = np.where(self.nd == xlnsnpb(v,self.B).nd, -0x7fffffffffffffff-1, 0)
  elif self.B == v.B:
   t.nd = np.where(self.nd == v.nd, -0x7fffffffffffffff-1, 0)
  else:
   t.nd = np.where(self.nd == xlnsnpb(v,self.B).nd, -0x7fffffffffffffff-1, 0)
  return t
xlnsnp.sum_default = xlnsnp.sum
xlnsnpv.sum_default = xlnsnpv.sum
xlnsnpb.sum_default = xlnsnpb.sum
xlnsnpr.sum_default = xlnsnpr.sum

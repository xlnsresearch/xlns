try:
  import sys
  import numpy as np
  import xlns as xl
  import math

except ModuleNotFoundError as me:
  print("Failed import: {}. Have you installed it?\nPerhaps you have a virtual environment available with it?".format(me.args[0]))
  sys.exit(1)

except ImportError as ie:
  print("Failed import: {}.".format(ie.args[0]))
  sys.exit(1)


def xlnstestinequality():
 xa=np.array([[-2,-1,-0.5,-0.25,0.25,.5,1,2]])
 ya=np.array(([[1,1,1,1,1,1,1,1]]))
 x = np.array(xl.xlnscopy(xa))
 y = np.array(xl.xlnscopy(ya))
 print((xa.T@ya) >= (ya.T@xa))
 print((x.T@y) >= (y.T@x))
 print((xa.T@ya) > (ya.T@xa))
 print((x.T@y) > (y.T@x))
 print((xa.T@ya) < (ya.T@xa))
 print((x.T@y) < (y.T@x))
 print((xa.T@ya) <= (ya.T@xa))
 print((x.T@y) <= (y.T@x))

def xlnstest():
 a=xl.xlns(1)
 b=xl.xlns(2)
 c=xl.xlns(-3.14159)
 d=+-c
 print(a.x,a.s)
 print(b.x,b.s)
 print(c.x,c.s)
 print(a-b)
 print(1.0-b)
 print(b-a)
 print(b-1)
 print(a)
 print(b)
 print(c)
 print(d)
 print(a/b)
 print(a/2)
 print(1.0/b)
 print(int(c))
 print(b*-123.4)
 print(-1000*c)
 print(a*b)
 print(b*c)
 print(c*c)
 print(a/c)
 print(c/b)
 print(a+b)
 print(c+c)
 print(a+c)
 print(c+b)
 print(c*c+c)
 print(a+1.0)
 print(1.0+c)
 print(math.exp(a))
 #
 a *= b
 print(a)
 c *= c
 print(c)
 a /= 2
 print(a)
 a += 1
 print(a)
 a -= 1
 print(a)
import numpy as np
import xlns as xl
import math

def xlnsnptestinequality():
 #compares have a lot of cases to test...
 xa=np.array([[-2,-1,-0.5,-0.25,0.25,.5,1,2]])
 ya=np.array(([[1,1,1,1,1,1,1,1]]))
 x = np.array(xl.xlnscopy(xa))
 y = np.array(xl.xlnscopy(ya))
 xn=xl.xlnsnp(x)
 #print(xa.T@ya)
 #print(ya.T@xa)
 
 #slightly awkward since xl.xlnsnp currently (1/24) doesn't allow nonscalar compare
 #and both xl.xlns and xl.xlnsnp don't have reverse compares defined
 print(">=")
 print((xa.T@ya) >= (ya.T@xa))
 #print((x.T@y) >= (y.T@x))
 xlist = []
 for i in range(0,len(x[0])):
   xlist += [list(x[0] <= x[0][i])] #backwards of x[0][i] >= x[0]
 print(np.bitwise_and.reduce(((xa.T@ya) >= (ya.T@xa))==np.array(xlist),axis=None))
 xnlist = []
 for i in range(0,len(xn[0])):
   xnlist += [list((xn[0] <= xa[0][i]).xlns()!=0)] #backwards of xn[0][i] >= xa[0]
 print(np.bitwise_and.reduce(((xa.T@ya) >= (ya.T@xa))==np.array(xnlist),axis=None))

 print(">")
 print((xa.T@ya) > (ya.T@xa))
 #print((x.T@y) > (y.T@x))
 xlist = []
 for i in range(0,len(x[0])):
   xlist += [list(x[0] < x[0][i])] #backwards of x[0][i] > x[0]
 print(np.bitwise_and.reduce(((xa.T@ya) > (ya.T@xa))==np.array(xlist),axis=None))
 xnlist = []
 for i in range(0,len(xn[0])):
   xnlist += [list((xn[0] < xa[0][i]).xlns()!=0)] #backwards of xn[0][i] > xa[0]
 print(np.bitwise_and.reduce(((xa.T@ya) > (ya.T@xa))==np.array(xnlist),axis=None))

 print("<")
 print((xa.T@ya) < (ya.T@xa))
 #print((x.T@y) < (y.T@x))
 xlist = []
 for i in range(0,len(x[0])):
   xlist += [list(x[0] > x[0][i])] #backwards of x[0][i] < x[0]
 print(np.bitwise_and.reduce(((xa.T@ya) < (ya.T@xa))==np.array(xlist),axis=None))
 xnlist = []
 for i in range(0,len(xn[0])):
   xnlist += [list((xn[0] > xa[0][i]).xlns()!=0)] #backwards of xn[0][i] < xa[0]
 print(np.bitwise_and.reduce(((xa.T@ya) < (ya.T@xa))==np.array(xnlist),axis=None))

 print("<=")
 print((xa.T@ya) <= (ya.T@xa))
 #print((x.T@y) <= (y.T@x))
 xlist = []
 for i in range(0,len(x[0])):
   xlist += [list(x[0] >= x[0][i])] #backwards of x[0][i] <= x[0]
 print(np.bitwise_and.reduce(((xa.T@ya) <= (ya.T@xa))==np.array(xlist),axis=None))
 xnlist = []
 for i in range(0,len(xn[0])):
   xnlist += [list((xn[0] >= xa[0][i]).xlns()!=0)] #backwards of xn[0][i] <= xa[0]
 print(np.bitwise_and.reduce(((xa.T@ya) <= (ya.T@xa))==np.array(xnlist),axis=None))


def xlnsvtestinequality(f1,f2):
 xa=np.array([[-2,-1,-0.5,-0.25,0.25,.5,1,2]])
 ya=np.array(([[1,1,1,1,1,1,1,1]]))
 x = np.array(xl.xlnscopy(xa, xl.xlnsv, f1))
 y = np.array(xl.xlnscopy(ya, xl.xlnsv, f2))
 print((xa.T@ya) >= (ya.T@xa))
 print((x.T@y) >= (y.T@x))
 print((xa.T@ya) > (ya.T@xa))
 print((x.T@y) > (y.T@x))
 print((xa.T@ya) < (ya.T@xa))
 print((x.T@y) < (y.T@x))
 print((xa.T@ya) <= (ya.T@xa))
 print((x.T@y) <= (y.T@x))

def xlnsvtest(setF):
 a=xl.xlnsv(1,setF)
 b=xl.xlnsv(2,setF)
 c=xl.xlnsv(-3.14159,setF)
 d=+-c
 print(a.x,a.s)
 print(b.x,b.s)
 print(c.x,c.s)
 print(a-b)
 print(1.0-b)
 print(b-a)
 print(b-1)
 print(a)
 print(b)
 print(c)
 print(d)
 print(a/b)
 print(a/2)
 print(1.0/b)
 print(int(c))
 print(b*-123.4)
 print(-1000*c)
 print(a*b)
 print(b*c)
 print(c*c)
 print(a/c)
 print(c/b)
 print(a+b)
 print(c+c)
 print(a+c)
 print(c+b)
 print(c*c+c)
 print(a+1.0)
 print(1.0+c)
 print(math.exp(a))
 #
 a *= b
 print(a)
 c *= c
 print(c)
 a /= 2
 print(a)
 a += 1
 print(a)
 a -= 1
 print(a)

def mixcasttest():
  v=np.array([xl.xlns(3),xl.xlnsr(3),xl.xlnsv(3,23),xl.xlnsb(3,2**2**-23),xl.xlnsnp(3),xl.xlnsnpr(3),xl.xlnsnpv(3,23),xl.xlnsnpb(3,2**2**-23)],dtype=object)
  v=v.reshape((8,1))
  print(str((3+3))+"=add |min|:" + str(abs(v+v.T).min()))
  print(str((3+3))+"=add |max|:" + str(abs(v+v.T).max()))
  print(str((3+3)*64)+"=add |sum|:" + str(abs(v+v.T).sum()))
  print(str((3-3))+"=sub |min|:" + str(abs(v-v.T).min()))
  print(str((3-3))+"=sub |max|:" + str(abs(v-v.T).max()))
  print(str((3-3)*64)+"=sub |sum|:" + str(abs(v-v.T).sum()))
  print(str((3*3))+"=mul |min|:" + str(abs(v*v.T).min()))
  print(str((3*3))+"=mul |max|:" + str(abs(v*v.T).max()))
  print(str((3*3)*64)+"=mul |sum|:" + str(abs(v*v.T).sum()))
  print(str((3/3))+"=div |min|:" + str(abs(v/v.T).min()))
  print(str((3/3))+"=div |max|:" + str(abs(v/v.T).max()))
  print(str((3/3)*64)+"=div |sum|:" + str(abs(v/v.T).sum()))
  #print((v==v.T).all())
  print((v!=v.T))
  #print((v!=v.T).any())
  print((xl.float64(v==v.T)))
  print((xl.float64(v==v.T)).all())
  print((xl.float64(v!=v.T)))
  print((xl.float64(v!=v.T)).any())
  z=np.array([xl.xlns(0),xl.xlnsr(0),xl.xlnsv(0,23),xl.xlnsb(0,2**2**-23),xl.xlnsnp(0),xl.xlnsnpr(0),xl.xlnsnpv(0,23),xl.xlnsnpb(0,2**2**-23)],dtype=object)
  z=z.reshape((8,1))
  print(str((0+3))+"=add |min|:" + str(abs(z+v.T).min()))
  print(str((0+3))+"=add |max|:" + str(abs(z+v.T).max()))
  print(str((0+3)*64)+"=add |sum|:" + str(abs(z+v.T).sum()))
  print(str(-(0-3))+"=sub |min|:" + str(abs(z-v.T).min()))
  print(str(-(0-3))+"=sub |max|:" + str(abs(z-v.T).max()))
  print(str(-(0-3)*64)+"=sub |sum|:" + str(abs(z-v.T).sum()))
  print(str((0*3))+"=mul |min|:" + str(abs(z*v.T).min()))
  print(str((0*3))+"=mul |max|:" + str(abs(z*v.T).max()))
  print(str((0*3)*64)+"=mul |sum|:" + str(abs(z*v.T).sum()))
  print(str((0/3))+"=div |min|:" + str(abs(z/v.T).min()))
  print(str((0/3))+"=div |max|:" + str(abs(z/v.T).max()))
  print(str((0/3)*64)+"=div |sum|:" + str(abs(z/v.T).sum()))
  print("3 op zero")
  print(str((3+0))+"=add |min|:" + str(abs(v+z.T).min()))
  print(str((3+0))+"=add |max|:" + str(abs(v+z.T).max()))
  print(str((3+0)*64)+"=add |sum|:" + str(abs(v+z.T).sum()))
  print(str((3-0))+"=sub |min|:" + str(abs(v-z.T).min()))
  print(str((3-0))+"=sub |max|:" + str(abs(v-z.T).max()))
  print(str((3-0)*64)+"=sub |sum|:" + str(abs(v-z.T).sum()))
  print(str((3*0))+"=mul |min|:" + str(abs(v*z.T).min()))
  print(str((3*0))+"=mul |max|:" + str(abs(v*z.T).max()))
  print(str((3*0)*64)+"=mul |sum|:" + str(abs(v*z.T).sum()))

def weirdzeroproblems():
  print("for xlnsnpb and xlnsb, should be 3 and 0")
  print(xl.xlnsnpb(xl.xlnsnpb(3,2**2**-14),2**2**-15))
  print(xl.xlnsnpb(xl.xlnsnpb(0,2**2**-14),2**2**-15))
  print(xl.xlnsb(xl.xlnsb(3,2**2**-14),2**2**-15))
  print(xl.xlnsb(xl.xlnsb(0,2**2**-14),2**2**-15))


def testrevnp():
  x = xl.xlns(3.14)
  xr = xl.xlnsr(3.14)
  xv = xl.xlnsv(3.14)
  xb = xl.xlnsb(3.14,2**2**-8)
  y = xl.xlnsnp([2,.5])
  yr = xl.xlnsnpr([2,.5])
  yv = xl.xlnsnpv([2,.5])
  yb = xl.xlnsnpb([2,.5],2**2**-8)

  print(x+y)
  print(xr+y)
  print(xv+y)
  print(xb+y)
  print(x+yr)
  print(xr+yr)
  print(xv+yr)
  print(xb+yr)
  print(x+yv)
  print(xr+yv)
  print(xv+yv)
  print(xb+yv)
  print(x+yb)
  print(xr+yb)
  print(xv+yb)
  print(xb+yb)

  print(x-y)
  print(xr-y)
  print(xv-y)
  print(xb-y)
  print(x-yr)
  print(xr-yr)
  print(xv-yr)
  print(xb-yr)
  print(x-yv)
  print(xr-yv)
  print(xv-yv)
  print(xb-yv)
  print(x-yb)
  print(xr-yb)
  print(xv-yb)
  print(xb-yb)

  print(x*y)
  print(xr*y)
  print(xv*y)
  print(xb*y)
  print(x*yr)
  print(xr*yr)
  print(xv*yr)
  print(xb*yr)
  print(x*yv)
  print(xr*yv)
  print(xv*yv)
  print(xb*yv)
  print(x*yb)
  print(xr*yb)
  print(xv*yb)
  print(xb*yb)

  print(x/y)
  print(xr/y)
  print(xv/y)
  print(xb/y)
  print(x/yr)
  print(xr/yr)
  print(xv/yr)
  print(xb/yr)
  print(x/yv)
  print(xr/yv)
  print(xv/yv)
  print(xb/yv)
  print(x/yb)
  print(xr/yb)
  print(xv/yb)
  print(xb/yb)

def doall():
  xlnstestinequality()
  xlnsnptestinequality()
  xlnsvtestinequality(23,6)
  xlnsvtest(6)
  xlnsvtest(23)
  xlnstest()
  mixcasttest()
  weirdzeroproblems()
  testrevnp()

doall()

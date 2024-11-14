"""
user-defined LNS addition via Fractional Normalization

importing this changes xlnsud ("user defined") class to implement 
one of two versions of LNS Add by Fractional Normalization:
   Original ("pal") by Tsiaras and Paliouras in ISCAS 2017 
   Improved ("han") by Zhang, Han, et al. in GLSVLSI 2024
Needed because above don't use Gaussian Logs (sb/db)
other variations in Han (guards, approx fxpt +) not impl 
"""

import xlns as xl
import numpy as np
import math


def ihanlog2(x):
    if x < 23<<(xl.xlnsF-4):                     #1.4375:
        return x + (x>>3) - (9<<(xl.xlnsF-3))    #1.125*x - 1.125
    else:
        return x - (x>>3) - (3<<(xl.xlnsF-2))    #0.875*x - 0.75

def ihanpow2(x):
    if x < 1<<(xl.xlnsF-1):                      #0.5:
        return x - (x>>3) + (1<<xl.xlnsF)        #0.875*x+1
    else:
        return x + (x>>3) + (7<<(xl.xlnsF-3))    #1.125*x + 0.875

def ipallog2(x): #Mitchell's method (fracnorm limited range 1<=x<2)
    return x - (1<<xl.xlnsF)

def ipalpow2(x): #Mitchell's method (fracnorm limited range 0<=x<2)
    return x + (1<<xl.xlnsF)

#choose "han" or "pal" 
ilog2 = ihanlog2
ipow2 = ihanpow2

def ifracnormAdd(x, y):  #(16) in Zhang and Han; (27) in Paliouras 
    xf = x & ((1<<xl.xlnsF)-1)
    xi = x - xf
    yf = y & ((1<<xl.xlnsF)-1)
    yi = y - yf
    d = (xi - yi) >> xl.xlnsF
    ap = ipow2(xf) + (ipow2(yf) >> d)
    if ap <= 2<<xl.xlnsF:
        return xi + ilog2(ap)
    else:
        return xi + (1<<xl.xlnsF) +  ilog2(ap>>1)


def ifracnormSub(x, y): #(17) in Zhang and Han; (28) in Paliouras 
    xf = (x & ((1<<xl.xlnsF)-1))
    xi = (x - xf)
    yf = (y & ((1<<xl.xlnsF)-1))
    yi = (y - yf)
    d = (xi - yi) >> xl.xlnsF
    am = ipow2(xf) - (ipow2(yf) >> d)
    if am == 0:
        return -1e1000 #-inf 
    elif am < 1<<xl.xlnsF:
        #find pos of leading '1'; crude, slow, but works
        p0 = int(1+math.floor(-math.log(am/2**xl.xlnsF,2))) 
        #print("case a")
        #print(str(p0)+" "+str(am<<p0))
        return xi - (p0<<xl.xlnsF) + ilog2(am<<p0)
    else:
        #print("case b")
        return xi + ilog2(am)


def xlnsud__add__fracnorm(self, v): 
    if isinstance(v, xl.xlns): #includes subclass xlnsud
        if self.x == -1e1000:  #-inf represents 0.0
            return xl.xlnsud(v)
        elif v.x == -1e1000:   #-inf represents 0.0
            return xl.xlnsud(self)
        else:
            t = xl.xlnsud("")  #cons with .x undefined
            if self.x>v.x:
                t.s = self.s
                if self.s==v.s:
                    t.x = ifracnormAdd(self.x, v.x)
                else:
                    t.x = ifracnormSub(self.x, v.x)
            else:
                t.s = v.s
                if self.s==v.s:
                    t.x = ifracnormAdd(v.x, self.x)
                else:
                    t.x = ifracnormSub(v.x, self.x)
            return t
    else:
       #recursive case does convert and tries again
       return xlnsud__add__fracnorm(self,xl.xlnsud(v)) 

#this changes the __add__ method of class xlnsud to use above
xl.xlnsud.__add__ = xlnsud__add__fracnorm


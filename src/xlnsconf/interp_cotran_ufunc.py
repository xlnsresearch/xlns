import xlns as xl
import numpy as np
import math

def sbdb_ufunc_interpsb(z,s,B=None,F=None):
  """interpolate sb with no guards, ideal db"""
  if B == None:
    B = xl.xlnsB
  if F == None:
    F = -math.floor(math.log(math.log(B,2),2))
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

def sbdb_ufunc_cotrdb(z,s,B=None,F=None):
  """ideal sb, flawed implementation of cotran db (doesn't handle special cases)"""
  if B == None:
    B = xl.xlnsB
  if F == None:
    F = -math.floor(math.log(math.log(B,2),2))
  N = (F-5)//2
  J = F - N
  zhmask = -(1<<J)
  #masks are arbitrary, but similar interp is about balanced
  zp = -z
  z2 = zp & zhmask
  z1 = zp - z2
  #instead of 1 use s, which==1 only for cases where dz1 and dz2 needed
  # won't blow up when s==0 and meaningless results discarded
  dz1 = xl.sbdb_ufunc_ideal(-z1,s,B=B,F=F)//2
  dz2 = xl.sbdb_ufunc_ideal(-z2,s,B=B,F=F)//2
  #if z1==0: #this case not needed because dz2 passes through
  # return d(-z2)
  #elif z2==0: #this case is needed
  # return d(-z1)
  #else:
  return np.where(s, np.where(z2==0, 2*dz1, 2*dz2 + xl.sbdb_ufunc_ideal(-z2-dz2+dz1,0,B=B,F=F)), 
                     xl.sbdb_ufunc_ideal(z,s,B=B,F=F))  #eventually not ideal

# actually do need z1==0 case because otherwise min cotr error 10x too big
#f=8
#(-0.004508454874271358, 0.003607134478010154, -0.0013525083564444966, 0.0013527217847684647) interp sb only
#(-0.0013545670545423813, 0.00135273860330414, -0.004117640906797612, 0.002544157551500135) cotran db only
#(-0.004508454874271358, 0.003607134478010154, -0.004117640906797612, 0.004225906378744445) interp sb+cotran db
#f=9
#(-0.0013633543465755961, 0.001943450850234808, -0.000677068610712258, 0.0006759805547302051) interp sb only
#(-0.0006770854994343167, 0.000676576737463107, -0.006525434211914523, 0.0013437892782083747) cotran db only
#(-0.0013633543465755961, 0.001943450850234808, -0.006525434211914523, 0.0023634033070977353) interp sb+cotran db
#f=11
#(-0.0003083659029870453, 0.0004921611921679127, -0.00016923391171403936, 0.00016920985324065616) interp sb only
#(-0.00016923554289142878, 0.0001692045822895741, -0.0037262940989443894, 0.00033484317398419275) cotran db only
#(-0.0003083659029870453, 0.0004921611921679127, -0.0037262940989443894, 0.0006394779906121974) interp sb+cotran db
#f=13
#(-8.252680875192645e-05, 0.00012411839172770596, -4.2307109072688124e-05, 4.230390603799188e-05) interp sb only
#(-4.2306712819998656e-05, 4.230536058343039e-05, -0.0019250925002089856, 8.162440603521942e-05) cotran db only
#(-8.252680875192645e-05, 0.00012411839172770596, -0.0018403205064528476, 0.00016139276934030105) interp sb+cotran db
#f=15
#(-2.07022694591547e-05, 3.140714511480401e-05, -1.0576641000511883e-05, 1.0576475657836816e-05) interp sb only
#(-1.0576595498909253e-05, 1.0576516825874298e-05, -0.000967004741118746, 2.112421874103721e-05) cotran db only
#(-2.07022694591547e-05, 3.140714511480401e-05, -0.0009458313366082998, 4.1009107609236585e-05) interp sb+cotran db

def sbdb_ufunc_interpsbcotrdb(z,s,B=None,F=None):
  """interpolate sb with no guards, correct cotran db using same interpolate"""
  if B == None:
    B = xl.xlnsB
  if F == None:
    #F = xl.xlnsF
    F = -math.floor(math.log(math.log(B,2),2))
    
  N = (F-5)//2
  J = F - N
  zhmask = -(1<<J)
  #masks are arbitrary, but similar to interp is about balanced
  zp = -z
  z2 = zp & zhmask
  z1 = zp - z2
  #instead of 1, use s, which==1 only for cases dz1,dz2 needed
  # won't blow up when s==0 and meaningless results discarded
  dz1 = xl.sbdb_ufunc_ideal(-z1,s,B=B,F=F)//2
  dz2 = xl.sbdb_ufunc_ideal(-z2,s,B=B,F=F)//2
  #return np.where(s, np.where(z2==0, 2*dz1, 
  #                                   2*dz2 + sbdb_ufunc_interpsb(-z2-dz2+dz1,0,B=B,F=F)), 
  #                   sbdb_ufunc_interpsb(z,s,B=B,F=F)) 
  return np.where(s, np.where(z2==0, 2*dz1, 
                              np.where(z1==0, 2*dz2, 
                                              2*dz2 + sbdb_ufunc_interpsb(-z2-dz2+dz1,0,B=B,F=F))), 
                     sbdb_ufunc_interpsb(z,s,B=B,F=F)) 
 
#with both special cases (z1==0 and z2==0), errors of interp sb+cotran db are reasonable
#f=8
#(-0.004508454874271358, 0.003607134478010154, -0.0013525083564444966, 0.0013527217847684647) interp sb only
#(-0.0013545670545423813, 0.00135273860330414, -0.004117640906797612, 0.002544157551500135) cotran db only
#(-0.004508454874271358, 0.003607134478010154, -0.0035211620439409043, 0.004225906378744445) interp sb+cotran db
#f=9
#(-0.0013633543465755961, 0.001943450850234808, -0.000677068610712258, 0.0006759805547302051) interp sb only
#(-0.0006770854994343167, 0.000676576737463107, -0.006525434211914523, 0.0013437892782083747) cotran db only
#(-0.0013633543465755961, 0.001943450850234808, -0.0012748573220109864, 0.0023634033070977353) interp sb+cotran db
#f=11
#(-0.0003083659029870453, 0.0004921611921679127, -0.00016923391171403936, 0.00016920985324065616) interp sb only
#(-0.00016923554289142878, 0.0001692045822895741, -0.0037262940989443894, 0.00033484317398419275) cotran db only
#(-0.0003083659029870453, 0.0004921611921679127, -0.00033434292608454743, 0.0006394779906121974) interp sb+cotran db
#f=13
#(-8.252680875192645e-05, 0.00012411839172770596, -4.2307109072688124e-05, 4.230390603799188e-05) interp sb only
#(-4.2306712819998656e-05, 4.230536058343039e-05, -0.0019250925002089856, 8.162440603521942e-05) cotran db only
#(-8.252680875192645e-05, 0.00012411839172770596, -8.212353245821513e-05, 0.00016139276934030105) interp sb+cotran db
#f=15
#(-2.07022694591547e-05, 3.140714511480401e-05, -1.0576641000511883e-05, 1.0576475657836816e-05) interp sb only
#(-1.0576595498909253e-05, 1.0576516825874298e-05, -0.000967004741118746, 2.112421874103721e-05) cotran db only
#(-2.07022694591547e-05, 3.140714511480401e-05, -2.1583566089955605e-05, 4.1009107609236585e-05) interp sb+cotran db

def sbdb_ufunc_interpsb_g2extra(z,s,B=None,F=None):
  """interpolate sb with 2 guards and extra table, ideal db"""
  if B == None:
    B = xl.xlnsB
  if F == None:
    F = -math.floor(math.log(math.log(B,2),2))
  B = B**.25
  F += 2
  z = z << 2
  N = (F-5)//2
  J = F - N
  #print("guard N="+str(N))
  #print("guard J="+str(J))
  zhmask = -(1<<J)
  zlmask = -1 - zhmask
  #print("guard zhmask="+str(bin(zhmask)))
  #print("guard zlmask="+str(bin(zlmask)))
  zh = z & zhmask
  zl = z & zlmask
  #print("guard zh="+str(bin(zh)))
  #print("guard zl="+str(bin(zl)))
  szh = xl.sbdb_ufunc_ideal(zh,0,B=B,F=F)//2
  szhp1 = xl.sbdb_ufunc_ideal(zh-zhmask,0,B=B,F=F)//2
  #print("guard szh="+str(bin(szh)))
  #print("guard szhp1="+str(bin(szhp1)))
  res = ((szhp1 - szh)* zl)>>J
  #print("guard res="+str(bin(res)))
  res += szh
  return np.where(s,xl.sbdb_ufunc_ideal(z,s,B=B,F=F), ((res+2)&-4)//2)  #rounding add, mask guard, (shift right 2 guard, but shift left 1 for sign)=> /2

def sbdb_ufunc_interpsb_g4extra(z,s,B=None,F=None):
  """interpolate sb with 4 guards and extra table, ideal db"""
  if B == None:
    B = xl.xlnsB
  if F == None:
    F = -math.floor(math.log(math.log(B,2),2))
  B = B**.0625
  F += 4
  z = z << 4
  N = (F-5)//2
  J = F - N
  #print("guard N="+str(N))
  #print("guard J="+str(J))
  zhmask = -(1<<J)
  zlmask = -1 - zhmask
  #print("guard zhmask="+str(bin(zhmask)))
  #print("guard zlmask="+str(bin(zlmask)))
  zh = z & zhmask
  zl = z & zlmask
  #print("guard zh="+str(bin(zh)))
  #print("guard zl="+str(bin(zl)))
  szh = xl.sbdb_ufunc_ideal(zh,0,B=B,F=F)//2
  szhp1 = xl.sbdb_ufunc_ideal(zh-zhmask,0,B=B,F=F)//2
  #print("guard szh="+str(bin(szh)))
  #print("guard szhp1="+str(bin(szhp1)))
  res = ((szhp1 - szh)* zl)>>J
  #print("guard res="+str(bin(res)))
  res += szh
  return np.where(s,xl.sbdb_ufunc_ideal(z,s,B=B,F=F), ((res+8)&-16)//8)  #rounding add, mask guard, (shift right 4 guard, but shift left 1 for sign)=> /8

def sbdb_ufunc_interpsb_g6extra(z,s,B=None,F=None):
  """interpolate sb with 6 guards and extra table, ideal db"""
  if B == None:
    B = xl.xlnsB
  if F == None:
    F = -math.floor(math.log(math.log(B,2),2))
  B = B**.015625
  F += 6
  z = z << 6
  N = (F-5)//2
  J = F - N
  #print("guard N="+str(N))
  #print("guard J="+str(J))
  zhmask = -(1<<J)
  zlmask = -1 - zhmask
  #print("guard zhmask="+str(bin(zhmask)))
  #print("guard zlmask="+str(bin(zlmask)))
  zh = z & zhmask
  zl = z & zlmask
  #print("guard zh="+str(bin(zh)))
  #print("guard zl="+str(bin(zl)))
  szh = xl.sbdb_ufunc_ideal(zh,0,B=B,F=F)//2
  szhp1 = xl.sbdb_ufunc_ideal(zh-zhmask,0,B=B,F=F)//2
  #print("guard szh="+str(bin(szh)))
  #print("guard szhp1="+str(bin(szhp1)))
  res = ((szhp1 - szh)* zl)>>J
  #print("guard res="+str(bin(res)))
  res += szh
  return np.where(s,xl.sbdb_ufunc_ideal(z,s,B=B,F=F), ((res+32)&-64)//32)  #rounding add, mask guard, (shift right 6 guard, but shift left 1 for sign)=> /32

def sbdb_ufunc_interpsb_g4(z,s,B=None,F=None):
  """interpolate sb with 4 guards, ideal db"""
  if B == None:
    B = xl.xlnsB
  if F == None:
    F = -math.floor(math.log(math.log(B,2),2))
  N = (F-5)//2
  J = F - N
  #print("guard N="+str(N))
  #print("guard J="+str(J))
  zhmask = -(1<<J)
  zlmask = -1 - zhmask
  #print("guard zhmask="+str(bin(zhmask)))
  #print("guard zlmask="+str(bin(zlmask)))
  zh = z & zhmask
  zl = z & zlmask
  #print("guard zh="+str(bin(zh)))
  #print("guard zl="+str(bin(zl)))

  B = B**.0625
  F += 4
  zh = zh << 4
  zhmask = zhmask << 4

  szh = xl.sbdb_ufunc_ideal(zh,0,B=B,F=F)//2
  szhp1 = xl.sbdb_ufunc_ideal(zh-zhmask,0,B=B,F=F)//2
  #print("guard szh="+str(bin(szh)))
  #print("guard szhp1="+str(bin(szhp1)))
  res = ((szhp1 - szh)* zl)>>J
  #print("guard res="+str(bin(res)))
  res += szh
  return np.where(s,xl.sbdb_ufunc_ideal(z,s,B=B,F=F), ((res+8)&-16)//8)  #rounding add, mask guard, (shift right 4 guard, but shift left 1 for sign)=> /8

def sbdb_ufunc_interpsb_g6(z,s,B=None,F=None):
  """interpolate sb with 4 guards, ideal db"""
  if B == None:
    B = xl.xlnsB
  if F == None:
    F = -math.floor(math.log(math.log(B,2),2))
  N = (F-5)//2
  J = F - N
  #print("guard N="+str(N))
  #print("guard J="+str(J))
  zhmask = -(1<<J)
  zlmask = -1 - zhmask
  #print("guard zhmask="+str(bin(zhmask)))
  #print("guard zlmask="+str(bin(zlmask)))
  zh = z & zhmask
  zl = z & zlmask

  B = B**.015625
  F += 6
  zh = zh << 6
  zhmask = zhmask << 6

  #print("guard zh="+str(bin(zh)))
  #print("guard zl="+str(bin(zl)))
  szh = xl.sbdb_ufunc_ideal(zh,0,B=B,F=F)//2
  szhp1 = xl.sbdb_ufunc_ideal(zh-zhmask,0,B=B,F=F)//2
  #print("guard szh="+str(bin(szh)))
  #print("guard szhp1="+str(bin(szhp1)))
  res = ((szhp1 - szh)* zl)>>J
  #print("guard res="+str(bin(res)))
  res += szh
  return np.where(s,xl.sbdb_ufunc_ideal(z,s,B=B,F=F), ((res+32)&-64)//32)  #rounding add, mask guard, (shift right 6 guard, but shift left 1 for sign)=> /32

def sbdb_ufunc_interpsb_g2(z,s,B=None,F=None):
  """interpolate sb with 2 guards, ideal db"""
  if B == None:
    B = xl.xlnsB
  if F == None:
    F = -math.floor(math.log(math.log(B,2),2))
  N = (F-5)//2
  J = F - N
  #print("guard N="+str(N))
  #print("guard J="+str(J))
  zhmask = -(1<<J)
  zlmask = -1 - zhmask
  #print("guard zhmask="+str(bin(zhmask)))
  #print("guard zlmask="+str(bin(zlmask)))
  zh = z & zhmask
  zl = z & zlmask

  B = B**.25
  F += 2
  zh = zh << 2
  zhmask = zhmask << 2

  #print("guard zh="+str(bin(zh)))
  #print("guard zl="+str(bin(zl)))
  szh = xl.sbdb_ufunc_ideal(zh,0,B=B,F=F)//2
  szhp1 = xl.sbdb_ufunc_ideal(zh-zhmask,0,B=B,F=F)//2
  #print("guard szh="+str(bin(szh)))
  #print("guard szhp1="+str(bin(szhp1)))
  res = ((szhp1 - szh)* zl)>>J
  #print("guard res="+str(bin(res)))
  res += szh
  return np.where(s,xl.sbdb_ufunc_ideal(z,s,B=B,F=F), ((res+2)&-4)//2)  #rounding add, mask guard, (shift right 2 guard, but shift left 1 for sign)=> /2

def sbdb_ufunc_interpsbcotrdb_g2(z,s,B=None,F=None):
  """interpolate sb with 2 guards, correct cotran db with same interpolate"""
  if B == None:
    B = xl.xlnsB
  if F == None:
    #F = xl.xlnsF
    F = -math.floor(math.log(math.log(B,2),2))
    
  N = (F-5)//2
  J = F - N
  zhmask = -(1<<J)  #masks arbitrary, similar to interp about balanced
  zp = -z
  z2 = zp & zhmask
  z1 = zp - z2
  #instead of 1, use s, which==1 only for cases dz1,dz2 needed
  # won't blow up when s==0 and meaningless results discarded
  dz1 = xl.sbdb_ufunc_ideal(-z1,s,B=B,F=F)//2
  dz2 = xl.sbdb_ufunc_ideal(-z2,s,B=B,F=F)//2
  return np.where(s, np.where(z2==0, 2*dz1, 
                              np.where(z1==0, 2*dz2, 
                                              2*dz2 + sbdb_ufunc_interpsb_g2(-z2-dz2+dz1,0,B=B,F=F))), 
                     sbdb_ufunc_interpsb_g2(z,s,B=B,F=F)) 
 

def sbdb_ufunc_interpsbcotrdb_g4(z,s,B=None,F=None):
  """interpolate sb with 4 guards, correct cotran db with same interpolate"""
  if B == None:
    B = xl.xlnsB
  if F == None:
    #F = xl.xlnsF
    F = -math.floor(math.log(math.log(B,2),2))
    
  N = (F-5)//2
  J = F - N
  zhmask = -(1<<J)  #masks arbitrary, similar to interp about balanced
  zp = -z
  z2 = zp & zhmask
  z1 = zp - z2
  #instead of 1, use s, which==1 only for cases dz1,dz2 needed
  # won't blow up when s==0 and meaningless results discarded
  dz1 = xl.sbdb_ufunc_ideal(-z1,s,B=B,F=F)//2
  dz2 = xl.sbdb_ufunc_ideal(-z2,s,B=B,F=F)//2
  return np.where(s, np.where(z2==0, 2*dz1, 
                              np.where(z1==0, 2*dz2, 
                                              2*dz2 + sbdb_ufunc_interpsb_g4(-z2-dz2+dz1,0,B=B,F=F))), 
                     sbdb_ufunc_interpsb_g4(z,s,B=B,F=F)) 
 

def sbdb_ufunc_interpsbcotrdb_g6(z,s,B=None,F=None):
  """interpolate sb with 6 guards, correct cotran db with same interpolate"""
  if B == None:
    B = xl.xlnsB
  if F == None:
    #F = xl.xlnsF
    F = -math.floor(math.log(math.log(B,2),2))
    
  N = (F-5)//2
  J = F - N
  zhmask = -(1<<J)  #masks arbitrary, similar to interp about balanced
  zp = -z
  z2 = zp & zhmask
  z1 = zp - z2
  #instead of 1, use s, which==1 only for cases dz1,dz2 needed
  # won't blow up when s==0 and meaningless results discarded
  dz1 = xl.sbdb_ufunc_ideal(-z1,s,B=B,F=F)//2
  dz2 = xl.sbdb_ufunc_ideal(-z2,s,B=B,F=F)//2
  return np.where(s, np.where(z2==0, 2*dz1, 
                              np.where(z1==0, 2*dz2, 
                                              2*dz2 + sbdb_ufunc_interpsb_g6(-z2-dz2+dz1,0,B=B,F=F))), 
                     sbdb_ufunc_interpsb_g6(z,s,B=B,F=F)) 
 


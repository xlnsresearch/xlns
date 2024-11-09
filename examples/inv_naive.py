import xlns as xl 
import numpy as np

def myinv(m):
    if isinstance(m, xl.xlnsnp):   #incl. xlnsnpv
        m = xl.hstack((m,xl.xlnsnp(np.eye(len(m)))))
    elif isinstance(m, xl.xlnsnpb):
        m = xl.hstack((m,xl.xlnsnpb(np.eye(len(m)),m.B)))
    elif isinstance(m, xl.xlnsnpr):
        m = xl.hstack((m,xl.xlnsnpr(np.eye(len(m)))))
    else:
        m = xl.hstack((m,np.eye(len(m))))
    print(m)
    print('-------')
    for p in range(0,len(m)):
        if m[p,p] == 0:  #here is why xlnsnp,etc need __bool__
            rn0 = -1
            for r in range(p,len(m)):
                if m[r][p] != 0:
                    rn0 = r
            if rn0 == -1:
                print("singular")
                quit()
            t = 1.*m[p,:]
            m[p,:] = m[rn0,:]
            m[rn0,:] = t
        m[p,:] /= m[p,p]
        for r in range(0,len(m)):
            if p!=r: 
                m[r,:] -= m[p,:] * m[r,p]
    return m[:,len(m):]

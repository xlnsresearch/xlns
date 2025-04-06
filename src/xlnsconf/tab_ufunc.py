"""Gaussian Log from lookup table in file (F<=20) to allows bit-for-bit emulation of hardware"""
import xlns as xl
import numpy as np
import os

global tab_sbdb,tab_ez,tab_B,tab_mismatch

def sbdb_ufunc_tab(z,s,B=None,F=None):
    """use table if B matches tab_B, otherwise revert to ideal"""
    global tab_mismatch
    if B == None:
        B = xl.xlnsB
    if B == tab_B:
        return tab_sbdb[s,np.maximum(tab_ez,np.where(z==0,-1,z))]
    else:
        if tab_mismatch == False:
            print("warning: table B="+str(tab_B)+" but requested B="+str(B)+", using ideal instead")
        tab_mismatch = True
        return xl.sbdb_ufunc_ideal(z,s,B,F)

def get_table(filestem):
    """Create/load file into table; file name user-supplied stem + frac digits of B"""
    global tab_sbdb,tab_ez,tab_B,tab_mismatch
    tab_mismatch = False
    tab_B = xl.xlnsB
    filename = "./"+filestem+"_"+str(tab_B)[2:]+".npz"
    if os.path.isfile(filename):
        print("using table from "+filename)
        tablefile = np.load(filename,"r")
        tab_ez = tablefile["tab_ez"]
        tab_sbdb = tablefile["tab_sbdb"] 
        tablefile.close()
    elif tab_B > 1.0000006:
        print("creating ideal table as "+filename)
        tab_ez=xl.sbdb_ufunc_ideal(1,1)
        zrange=np.arange(tab_ez,0)
        sbt=xl.sbdb_ufunc_ideal(zrange,0)
        dbt=xl.sbdb_ufunc_ideal(zrange,1)
        tab_sbdb=np.vstack((sbt,dbt))
        np.savez_compressed(filename,tab_ez=tab_ez,tab_sbdb=tab_sbdb)
    else:
        print("error: table too large to create")
        tab_B = None


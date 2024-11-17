# xlns Configurations

This folder contains files that can be imported into a Python program using xlns to alter the way addition and/or summation is performed.  By default,  xlns uses 64-bit build-in floating point (FP) to calculate the Gaussian Log as accurately as is possible with such hardware.  We call this approach _ideal_, which is not quite the same as "round to nearest" or "exact" but very nearly so, especially for small values of F (which is set by xl.xlnssetF() and internally known as xl.xlnsF).  

There are several notations that have been used for the Gaussian Log.  The one that this package uses is __sbdb_ufunc(z,zs)__.  The reason for this is that the Gaussian Log is often thought of as two separate functions:  one for when (zs==0) the signs of the numbers added are the _same_:

sb(z) = np.log(B**z+1)/np.log(B)

and the other for when (zs==1) the signs _differ_.

db(z) = np.log(B**z+1)/np.log(B)

Notice in this package these are always described in a way compatible with the NumPy concept of a _ufunc_, in other words z and anything done with z are compatible with element-by-element NumPy operations.  In the implementation of __sbdb_ufunc__ this requires using np.where based on zs.

The recent interest in LNS is mostly because there are many approximations which greatly reduce the cost of the hardware (by not actually carrying out the logarithm and exponetial in floating point) at the expense of added error.  The purpose of the code in this folder is to implement some of the hundreds approximations that have been published in the literature.  It is not possible in software to acheive the power and speed improvement that these techniques do in hardware.  The only goal of the code in this folder is to simulate the numerical characteristics of such hardware as accurately as possible.  We actively seek open-source contribution to this folder. (See below for details)

# Available Configurations

__dally__: summation technique disclosed in U.S. Patent Application Publication US20210056446A1 (academic, non-commercial use only)

__interp_cotran_ufunc__: linear Lagrange interpolation and/or (Arnold,1998) cotransformation of Gaussian Log

__lpvip_ufunc__: Low Precision Very Insignificant Power (Arnold,2004) with (Mitchell,1962) to approximate Gaussian Log 

__xlnsudFracnorm__: override \_\_add\_\_ of xlnsud for Fractional Normalization, (Tsiraras-Paliouras,2017) or (Zhang-Han,2024) 

These can be classified as those that are ufunc configurations and those that are not.

# ufunc Configurations

The configurations that end in "_ufunc" are intended to replace the default ideal computation of the Gaussian Log.  Doing this will impact the behavior of all the classes (except xlnsud in some cases, see below).  To use one of the "_ufunc" modules, you need to import the module and then assign xl.sbdb_ufunc the pointer to one of the "_ufunc"s that are defined in that module (there are often more than one version in the file giving options for how the approximation is accomplished). For example,

import xlnsconf.lpvip_ufunc
help(xlnsconf.lpvip_ufunc)
xl.sbdb_ufunc = xlnsconf.lpvip_ufunc.sbdb_ufunc_lpvip

The help command lists several functions to choose from that start with "sbdb_ufunc_".  The simplest (but least accurate) among these is sbdb_ufunc_lpvip.

To restore the ideal behavior of the Gaussian Log,

xl.sbdb_ufunc = xl.sbdb_ufunc_ideal

# other Configurations

See the relevant help for more information about the other configurations.
Note: simply importing xlnsudFracnorm alters the behavior of xlnsud, but all other classes operate using whatever function sbdb_ufunc is currently pointing to.





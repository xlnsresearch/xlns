# xlns Configurations

This folder contains files that can be imported into a Python program using xlns to alter the way addition and/or summation is performed.  By default,  xlns uses 64-bit build-in floating point (FP) to calculate the Gaussian Log as accurately as is possible with such hardware.  We call this approach "ideal," which is not quite the same as "round to nearest" or "exact" but very nearly so, especially for small values of F (which is set by xl.xlnssetF() and internally known as xl.xlnsF).  

The recent interest in LNS is mostly because there are approximations which greatly reduce the cost of the hardware at the expense of added error.  The purpose of the code in this folder is to implement some of the hundreds approximations that have been published in the literature.  We actively seek open-source contribution to this folder. (See below for details)

# Available Configurations

dally: summation technique disclosed in U.S. Patent Application Publication US20210056446A1 (academic, non-commercial use only)

interp_cotran_ufunc

lpvip_ufunc

xlnsudFracnorm

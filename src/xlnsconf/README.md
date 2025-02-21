# xlns Configurations

This folder contains files that can be imported into a Python program using ``xlns`` to alter the way addition and/or summation is performed.  By default,  ``xlns`` uses 64-bit built-in floating point (FP) to calculate the Gaussian Log as accurately as is possible with such hardware.  We call this approach _ideal_, which is not quite the same as "round to nearest" or "exact" but very nearly so, especially for small values of F (which is set by ``xl.xlnssetF()`` and internally known as ``xl.xlnsF``).  

The recent interest in LNS is mostly because there are many approximations which greatly reduce the cost of the hardware (by not actually carrying out the logarithm and exponetial in floating point) at the expense of added error.  The purpose of the code in this folder is to implement some of the hundreds approximations that have been published in the literature.  It is not possible in software to acheive the power and speed improvement that these techniques do in hardware.  The only goal of the code in this folder is to simulate the numerical characteristics of such hardware as accurately as possible.  We actively seek open-source contribution to this folder. (See below for details)

# Notation

There are several notations that have been used for the Gaussian Log.  The one that this package uses is __sbdb_ufunc(z,zs)__.  The reason for this is that the Gaussian Log is often thought of as two separate functions:  one for when (``zs==0``) the signs of the numbers added are the _same_:
```
sb(z) = np.log(b**z+1)/np.log(b)
```
and the other for when (``zs==1``) the signs _differ_:
```
db(z) = np.log(np.abs(b**z-1))/np.log(b)
```
See [Wikipedia:Gaussian Logarithm](https://en.wikipedia.org/wiki/Gaussian_logarithm) for a graph of sb and db. 
As given above, they are defined for both positive and negative z, but in this package it is only required to implement them for negative z because ``sb(z) == sb(-z)+z`` and ``db(z) == db(-z)+z``. Also in this package the Gaussian Log is always described in a way compatible with the NumPy concept of a _ufunc_, in other words ``z`` and anything done with ``z`` are compatible with element-by-element NumPy operations.  

_Note to implementators of __sbdb_ufunc___: this ufunc assumption may require using ``np.where`` based on zs or other NumPy rank-polymorphic tricks.  Also note: the constant b is the base of the logarithm, which is given (in most instances) by the global variable ``xl.xlnsB`` that is automatically calculated by ``xl.xlnssetF()``.  (Another observation from the weeds of maths and programming:  the Python math library offers the base-b logarithm of x, ``math.log(x,b)``, but NumPy only has ``np.log()``, which is limited to base e=2.71828...  One of the laws of logarithms is that the base-b logarithm of x is the base-e logarithm, ``np.log(x)``, divided by the base-e log of that constant, ``np.log(b)``. Since ``b`` is scalar but ``z`` is often a large array, the overhead of this conversion is insignificant for many simulations.) 

# Available Configurations

__dally__: summation technique disclosed in U.S. Patent Application Publication US20210056446A1 (academic, non-commercial use only)

__interp_cotran_ufunc__: unpartitioned linear Lagrange interpolation and/or (Arnold,1998) cotransformation of Gaussian Log

__lpvip_ufunc__: Low Precision Very Insignificant Power (Arnold,2004) with (Mitchell,1962) to approximate Gaussian Log 

__utah_tayco_ufunc__: unpartitioned linear Taylor interpolation and/or (Coleman, 2000) cotransformation of Gaussian Log contributed by the University of Utah (Thanh Son Nguyen and Alexey Solovyev), the accuracy of which has been proven using Lean4. 

__xlnsudFracnorm__: override ``__add__`` of ``xlnsud`` for Fractional Normalization, (Tsiraras-Paliouras,2017) or (Zhang-Han,2024) 

These can be classified as those that are ufunc configurations and those that are not.

# ufunc Configurations

The configurations that end in "_ufunc" are intended to replace the default ideal computation of the Gaussian Log.  Doing this will impact the behavior of all the classes (except ``xlnsud`` in some cases, see below).  To use one of the "_ufunc" modules, you need to import the module and then assign ``xl.sbdb_ufunc`` the pointer to one of the "_ufunc"s that are defined in that module (there are often more than one version in the file giving options for how the approximation is accomplished). For example,
```
import xlnsconf.lpvip_ufunc
help(xlnsconf.lpvip_ufunc)
xl.sbdb_ufunc = xlnsconf.lpvip_ufunc.sbdb_ufunc_lpvip
```
The help command lists several functions to choose from that start with "sbdb_ufunc_".  The simplest (but least accurate) among these is ``sbdb_ufunc_lpvip``.

To restore the ideal behavior of the Gaussian Log,
```
xl.sbdb_ufunc = xl.sbdb_ufunc_ideal
```
In the ``xlns/examples`` folder there are programs (``arn_generic.py``,``arnnp_lpvip``, and ``arnnpr_lpvip``) that use ``xlnsconf.lpvip_ufunc`` as the addition algorithm in back-propagation training of a neural network with the MNIST dataset.

# Other Configurations

See the relevant ``help`` for more information about the other configurations.
Note: simply importing ``xlnsudFracnorm`` alters the behavior of ``xlnsud``, but all other classes operate using whatever function ``xl.sbdb_ufunc`` is currently pointing to.

In the ``xlns/examples`` folder is a program (``arn_generic.py``), which when run with the command line option ``--type xlnsud`` uses fractional normalization as the addition algorithm in back-propagation training of a neural network with the MNIST dataset.

# Characterizing Numerical Behavior of User Configurations

_to be implemented_

# Submission Procedure for new User Configurations

We actively encourage your open-source contribution to this folder. There are hundreds of techniques in the literature.  A great project for a student or new person to the LNS field would be to learn enough about one of them and implement it with compatible Python.  There are also researchers developing new LNS methods, and we welcome them to contribute here.

There is a catch, of course.  Here is the deal: your code must pass certain tests using the tools described in the previous section 
 _to be described later_ 

 # References

M. G. Arnold, et al. “Arithmetic cotransformations in the Real and
Complex Logarithmic Number Systems,” _IEEE Trans. Comput._, vol. 47,
no. 7, pp.777–786, July 1998.


M. G. Arnold, "LPVIP: A Low-power ROM-Less ALU for Low-Precision LNS," 
_14th International Workshop on Power and Timing Modeling, Optimization and Simulation_,
LNCS 3254, pp.675-684, Santorini, Greece, Sept. 2004.

W. J. Dally, et al., “Inference accelerator using logarithmic-based arithmetic,” 
U.S. Patent Application Publication US20210056446A1, 2021.

G. Tsiaras and V. Paliouras, “Logarithmic Number System Addition and
Subtraction Using Fractional Normalization,” 
_IEEE Symp. on Circuits and Systems_, pp.1–4, 2017.

W. Zhang, J. Han, et al., “A Low-Power and High-Accuracy Approximate Adder
for the Logarithmic Number Systems,” _Great Lakes Symposium on VLSI_, pp. 125–131, 2024.




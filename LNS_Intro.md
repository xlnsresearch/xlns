# Introduction to Logarthmic Number Systems
In case you are using this package and would like to know what is going on, and why.

## What is LNS
The Logarithmic Number System (LNS) is a numerical representation system that uses logarithms to encode values instead of direct values. If you are familiar with floating point (FP) representation and arithmetic, then you will understand that in FP, a fractional number (typically with a hidden leading '1') is raised to some exponent of 2. The exponents in FP are integers.

In the LNS, numbers are represented as the logarithm of a value, rather than the value itself. To compare it with floating point, this is analogous to setting the significand to a constant 1, and allowing the exponent to become a fraction. Now, only a single value is used to represent a real, rather than the significand-exponent pair.

There is a further generalisation which is important to consider: the 'power of 2' need not be of 2 - the base of the logarithm may be any (sensible) number we choose.

## How, Where and Why LNS Can Be Useful

The back story to all this is that in the early days of computing and the introduction of real arithmetic co-processors and sophisticated on-chip ALUs, it quickly became apparent that some important numerical tasks (multiplication, division, square root) were profoundly slow compared to addition and substraction, and typically were either iterative processes using microcode to refine an approximation at the rate of about 1 bit per pipeline cycle, or used vast (relatively speaking) areas of silicon real estate to implement complete circuits.

The establishment of floating point as a general purpose representation and arithmetic basis was powerful, but it did nothing to mitigate the difficulty of these operations. There are several good tutorials available online for how floating point arithmetic works. _Take a pencil..._

Early researchers in LNS demonstrated that doing difficult operations in a logarithmic scheme was akin to giving humans slide rules. The key idea is that multiplication and division of numbers in the LNS becomes much simpler, since they become fixed point (integer) addition and subtraction of the logarithms. For base 2 logarithms, square roots and exponentiation become trivial.

For most of the rest of this introduction, we will stick with bsae-2 (binary) logarithms, since this is easier to imagine, and represents the bulk of the published research.

## Operations in LNS

The point of LNS is to recognise that arithmetic performance (however you measure it - accuracy, precision, atomic speed, pipeline throughout, accumulated noise, power consumption...) ultimately dominates any quality metric in numerical computing, signal processing, graphics engines, and so on. In the modern era of cloud computing and the growth of machine learning and AI, improving arithmetic performance even a little can be scaled to have significant impacts.

Here's how LNS arithmetic operations play out.

### Multiplication and Division
Real-valued multiplication maps to logarithmic addition:
```
c = log(a * b)
```
is identical (assuming perfect arithmetic) to
```
c = log(a) + log(b)
```
while similarly division maps to subtraction:
```
d = log(a / b) 
d = log(a) - log(b)
```
If you are happy with this so far, hopefully it is already apparent why LNS can be a powerful alternative to floating point -- and even to fixed point arithmetic. Division is especially notorious in conventional standards. 

It is worth now introducing a couple of additional issues that need attention. Regardless of the overall sign of the real value, the logarithm itself is a signed quantity, since the real value may be larger or smaller than the base of the logarithm. So, the log values are signed - but we also need an overall sign flag ``s`` for the complete representation. We choose the sign bit to be 0 for positive values, and 1 for negative values; i.e. if the flag is set, the value is negative.

An LNS maps a real value _A_ into its absolute logarithm ``a`` in some base _B_:
```
# for A != 0: this is equivalent to math.log(A)/math.log(B):
a = math.log(A, B)
s = int (a<0)
```
The second issue that needs consideration is representation of zero, since the logarithm of 0 is undefined. A special representation must be chosen for this, and different LNS implementations and researchers have their own preferences. This is not unlike how floating point has reserved representations for NaN and infinity, etc. So, the exact value that ``A`` represents in LNS is *(sticking with base-2 for now)*:
```
A = np.float_power(-1, s) * (2 ** a)
```
**Note:** *float_power() is used here since the power operand ``**`` doesn't behave as you'd expect for negative powers.
Also - remember that ``a`` itself is signed.*

Here's a complete code block to illustrate the values in the LNS representation.
```
import numpy as np
import math

def logn(value, base):
    '''
    a function to calculate the logarithm of value to some base
    '''
    if (abs(value)>0):
        return math.log(abs(value))/math.log(base)
    else:
        return float('-inf')

# choose base of log
b = 2

# set a value
a = -1 * math.pi

# calculate the log part of the value
la = logn(a, b)

# calculate its sign bit
sa = int (a<0)

# combine them to restore the value
A = np.float_power(-1, sa) * b**la
print(A)
```

If you experiment with the base ``b``, you should find that it does not change the output. _Exercise for the reader - what happens if ``b`` is negative, or a small fraction?_

### Exponentiation
Having reduced the order of operations, exponentiation (including root) now maps into multiplication. For the typical case of base-2 logarithms and integer powers of exponentiation, square and square root respectively correspond to doubling and halving the logarithmic value.

These are now the easist of all operations, rather than the hardest - since doubling and halving can be achieved by single left-shifts or right-shifts of the value. In real values, these operations are crucial in signal processing, matrix arithmetic, graphics, and control systems.

*It all sounds too good to be true, right?*

Yep.

### Addition and Subtraction

The mathemetically inclined, or those who are thinking ahead, have possibly realised that this happy land of easy arithmetic operations could not last long. Addition and substraction are significantly harder tasks in the LNS, and they have absorbed nearly all the research and development efforts of the LNS community for 50 years. It is sufficiently hard that some folks prefer to take the antilogarithm (i.e. convert back to a linear scale), perform the operation(s), and then take the logarithm of the result. This is expensive and tedious.

Gauss (Johann Carl Friedrich, famous for many things) noticed a helpful relation to compute the sum of logarithms:
```
a + b = b * (a/b + 1)
```
wherein the multiply and divide are fast and cheap in the LNS. So, the sum ``c`` of a pair of logarithms (``a``, ``b``) is given by
```
c ≈ b + g(z, zs)
``` 
for ``z = b - a``, and ``zs`` is the sign bit formed by the XOR of the sign bits of ``a`` and ``b``.

Finding the difference between LNS values is very similar to addition, with the additional snag that for similar values (i.e. the difference between ``a`` and ``b`` approaches zero) the Gaussian log function ``g()`` enjoys an asymptote to negative infinity.

Just how difficult it is to implement addition and subtraction is determined by how much accuracy you need to achieve, and what the hardware will support. For a dedicated hardware implementation, the popular approach is to perform some approximation for ``g()``: as a quantised look up from a stored table, as a stepwise linear approximation, as a first order taylor expansion, or as a higher-order curve. The complexity of the approximation broadly trades off the achieved accuracy with time, power consumption and/or silicon area.

So now you know enough to go and explore the published literature about alternative machine arithmetic using the Logarithmic Number System. The best collection of such material is given in the Resources section below.

You are also ready to play with the xlns package. It overloads the normal arithmetic operators with their LNS equivalents, and makes the LNS operations available to numpy types including large vectors.

### Beyond Base 2

At the start of this introduction we limited consideration to base 2 logarithms, because it is easier to imagine how it works when compared with IEEE standard floating point. But since the base of the logarithm doesn't really affect the operations other than when converting to/from other representations, we are free to choose other values. There are some interesting choices to make here, and the merits of each are beyond the scope of this introduction.

What you do need to know is that the xlns package supports other bases, and you should probably choose this before you try anything. *The default value is not 2!* This is achieved with the function that effectively sets the precision of the represention. For example,
```
import xlns as xl
xl.xlnssetF(10)
```

Here, F is a value that represents the number of bits of precision in the representation. `F=23` approximately corresponds to single precision floating point. The default base of the logarithms is set to 
```
B = 2 ** (2 ** -F)
```

which is approximately 1 (but crucially, not actually 1). 

## Using the package

The `xlns` package provides you with variables and arithmetic that are internally represented/implemented as an LNS. That is, the logarithm part and the sign bit from the example above are abstracted and you can (mostly) forget you are using a different number system. The operators on such variables are overloaded with their LNS equivalents, and the package provides you with functions and hooks to adapt the representation and the arithmetic to suit your needs. 


## Resources

The Logarithmic Number System offers a powerful way to simplify computations, especially where multiplication and division of large values are involved. It is most useful in specialized fields that require efficient handling of wide-ranging magnitudes and where performance gains from reducing computational complexity are important.

For more information, a comprehensive bibliography of published research is available here:
- [XLNS Research](https://xlnxresearch.com/)

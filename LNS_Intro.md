# Introduction to Logarthmic Number Systems
In case you are using this package and would like to know what is going on, and why.

## What is LNS
The Logarithmic Number System (LNS) is a numerical representation system that uses logarithms to encode values instead of direct values. If you are familiar with floating point (FP) representation and arithmetic, then you will understand that in FP, a fractional number (typically with a hidden leading '1') is raised to some exponent of 2. The exponents in FP are integers.

In the LNS, numbers are represented as the logarithm of a value, rather than the value itself. To compare it with floating point, this is analogous to setting the significand to a constant 1, and allowing the exponent to become a fraction. Now, only a single value is used to represent a real, rather than the significand-exponent pair.

There is a further generalisation which is important to consider: the 'power of 2' need not be of 2 - the base of the logarithm may be any number we choose.

## How, Where and Why LNS Can Be Useful

The back story to all this is that in the early days of computing and the introduction of real arithmetic co-processors and sophisticated on-chip ALUs, it quickly became apparent that some important numerical tasks (multiplication, division, square root) were profoundly slow compared to addition and substraction, and typically were either iterative processes using microcode to refine an approximation at the rate of about 1 bit per pipeline cycle, or used vast (relatively speaking) areas of silicon real estate to implement complete circuits.

The establishment of floating point as a general purpose representation and arithmetic basis is powerful, but it did nothing to mitigate the difficulty of these operations.

Early researchers in LNS demonstrated that doing difficult operations in a logarithmic scheme was akin to giving humans slide rules. The key idea is that multiplication and division of numbers in the LNS becomes much simpler, since they become fixed point (integer) addition and subtraction of the logarithms. For base 2 logarithms, square roots and exponentiation become trivial.

For most of the rest of this introduction, we will stick with bsae-2 (binary) logarithms, since this is easier to imagine, and represents the bulk of the published research.

## Operations in LNS

The point of LNS is to recognise that arithmetic performance (however you measure it - accuracy, precision, atomic speed, pipeline throughout, accumulated noise, power consumption...) ultimately dominates any quality metric in numerical computing, signal processing, graphics engines, and so on. In the modern era of cloud computing and the growth of machine learning and AI, improving arithmetic performance even a little can be scaled to have significant impacts.

Here's how LNS arithmetic operations play out.

### Multiplication and Division
Real-valued multiplication maps to logarithmic addition:
```
log(a * b) = log(a) + log(b)
```
while similarly division maps to subtraction:
```
log(a / b) = log(a) - log(b)
```
If you are happy with this so far, hopefully it is already apparent why LNS can be a powerful alternative to floating point -- and even to fixed point arithmetic. It is worth now introducing a couple of additional issues that need attention.
Regardless of the overall sign of the real value, the logarithm itself is a signed quantity, since the real value may be larger or smaller than the base of the logarithm. So, the log values are signed - but we also need an overall sign flag for the complete representation.

An LNS maps a real value A into its absolute logarithm a:
```
a ≈ log_B |A|
```
and its sign s_A into sign bit s: 
```
s = (−1)^s_A
```
The second issue that needs consideration is representation of zero, since the logarithm of 0 is undefined. A special representation must be chosen for this, and different LNS implementations and researchers have their own preferences. This is not unlike how floating point has reserved representations for NaN and infinity, etc. So, the exact value that U represents in LNS is *(sticking with base-2 for now)*:
```
A = (−1)^s * 2^a
```

### Exponentiation
Having reduced the order of operations, exponentiation (including root) now maps into multiplication. For the typical case of base-2 logarithms and integer powers of exponentiation, square and square root respectively correspond to doubling and halving the logarithmic value.

These are now the easist of all operations, rather than the hardest - since doubling and halving can be achieved by single left-shifts or right-shifts of the value. In real values, these operations are crucial in signal processing, matrix arithmetic, graphics, and control systems.

*It all sounds too good to be true, right?*

Yep.

### Addition
_to be written_

### Subtraction
_to be written__

## Resources
The Logarithmic Number System offers a powerful way to simplify computations, especially where multiplication and division of large values are involved. It is most useful in specialized fields that require efficient handling of wide-ranging magnitudes and where performance gains from reducing computational complexity are important.

For more information, a comprehensive bibliography of published research is available here:
- [XLNS Research](https://xlnxresearch.com/)

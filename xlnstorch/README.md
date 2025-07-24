# xlns
XLNS: a configurable python package for Logarithmic Number System eXperimentation


# Getting Started

It is recommended to create a python virtual environment for experimentation with the ``xlns`` package. Within that environment, ensure you have the dependencies listed below installed.

Explanatory information about LNS is in the ``LNS_Intro.md`` file.  By default,  ``xlns`` uses 64-bit built-in floating point (FP) to calculate as accurately as is possible.  We call this approach _ideal_. The recent interest in LNS is mostly because there are many _non-ideal_ approximations which greatly reduce the cost of the hardware, and some of these methods are available here as _user configurations_.  Information on available user configurations can be found in the ``README.md`` file in ``src/xlnsconf``.  It also explains how you can contribute your own configurations to this open-source project.

# Dependencies
numpy

````
pip3 install numpy
````

For some examples, matplotlib is also needed:
````
pip3 install matplotlib
````

You can grab all the dependencies in one go using the requirements file thus:
````
pip install -r requirements.txt
````

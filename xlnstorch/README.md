# xlnstorch
XLNSTorch: a PyTorch addon python package for simulating Logarithmic Number System (LNS) arithmetic.
You can find the docs at https://xlnsresearch.github.io/xlns/.


# Getting Started

To install ``xlnstorch``, you can run ``pip3 install xlnstorch``. If you have a C++ compiler, this
will allow you to use more efficient operations and layers. Otherwise, ``xlnstorch`` will fallback
to pure python, slower implementations.

To learn more about LNS, see the ``LNS_Intro.md`` and ``src/xlnsconf/README.md`` files. For examples,
look at the documentation and the ``examples`` directory.


# Dependencies

xlnstorch has several dependencies that are automatically installed when you install it:
- torch
- xlns
- numpy

xlnstorch also has several optional dependencies for additional features:
- matplotlib: For many of the ``viz`` submodule's graphs
- graphviz: For visualizing the computational graph
- torchvision: For tensor transforms and datasets


# To Do list
- Implement more transformer layers.
- Improve support for saving and loading LNSTensor weights, and copying
  weights between torch and xlnstorch.
- Rework float64 storage to bitcast rather than reinterpret types.
- Implement more layers in the C++ backend.
- Implement positive and negative infinity sentinel values.
- Improve type and shape checks/error messages.
- Support LNSTensors and operations performed on the GPU.
- Add more implementations from ``xlnsconf``.
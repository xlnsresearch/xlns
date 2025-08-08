Installing xlnstorch
====================

To install xlnstorch, you can use pip. The package is available on PyPI, so you
can install it directly using the following command:

.. code-block:: bash

    pip install xlnstorch

xlnstorch has several dependencies that are required for its functionality. These
will be automatically installed when you install xlnstorch via pip. They include:

- `torch`: PyTorch handles the core machine learning functionalities.
- `xlns`: For casting to and from xlns objects and LNSTensor.

xlnstorch also has several optional dependencies for additional features:

- `matplotlib`: For many of the viz submodule plots.
- `graphviz`: For visualizing the computational graph.
- `torchvision`: For tensor transforms and datasets.
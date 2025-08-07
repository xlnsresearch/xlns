.. currentmodule:: xlnstorch.viz

.. _visualization-doc:

Visualization
=============

The ``xlnstorch.viz`` submodule provides tools for visualizing and analyzing
LNS operations and their numerical properties. This module includes functionality
for creating error heatmaps, generating autograd graphs, and analyzing the
precision characteristics of LNS arithmetic operations.

Quick Start
-----------

Here's a simple example of visualizing the error characteristics of an LNS operation:

.. code-block:: python

    import torch
    import xlnstorch as xltorch
    from xlnstorch.viz import make_error_grid, plot_error_heatmap
    import matplotlib.pyplot as plt

    # Create error grid for multiplication operation
    xs, ys, errors = make_error_grid(
        torch.mul,
        x_range=(-2.0, 2.0),
        y_range=(-2.0, 2.0),
        steps=100,
        f=8,
    )

    # Plot the error heatmap
    fig, ax = plt.subplots(figsize=(8, 6))
    plot_error_heatmap(errors, xs, ys, ax=ax)
    plt.show()

For visualizing autograd graphs with LNS tensors:

.. code-block:: python

    from xlnstorch.viz.graph import make_autograd_graph

    # Create some LNS tensors with gradients
    x = xltorch.randn(3, 3, requires_grad=True)
    y = xltorch.randn(3, 3, requires_grad=True)

    # Perform operations
    z = x * y + x.pow(2)
    loss = z.sum()

    # Generate autograd graph
    graph = make_autograd_graph(loss, params={'x': x, 'y': y})
    graph.render('autograd_graph', format='svg')

Overview
--------

The visualization module is organized into these main areas:

.. toctree::
   :maxdepth: 2

   viz/error_analysis
   viz/precision_comparison  
   viz/autograd_graphs

Error Analysis Functions
------------------------

The error analysis tools help understand the numerical behavior of LNS operations
compared to exact arithmetic.

.. autosummary::
    :toctree: generated/viz
    :nosignatures:

    make_error_grid
    plot_error_heatmap
    plot_staircase
    plot_spacing_heatmap

Precision Comparison Functions
------------------------------

The precision comparison tools allow you to analyze how different LNS configurations
affect numerical accuracy for the same operation.

.. autosummary::
    :toctree: generated/viz
    :nosignatures:

    precision_sweep_analysis
    plot_precision_comparison
    plot_precision_heatmap_grid

Autograd visualization
----------------------

The autograd visualization tools provide insight into the computational graph
structure when using LNS tensors.

.. autosummary::
    :toctree: generated/viz
    :nosignatures:

    graph.make_autograd_graph

Dependencies
------------

The visualization module has optional dependencies:

- **matplotlib**: Required for the plotting functions.
- **graphviz**: Required for ``make_autograd_graph``

Install with:

.. code-block:: bash

    pip install matplotlib graphviz

Note that for ``graphviz``, you may also need to install the system graphviz libraries:

.. code-block:: bash

    # On Ubuntu/Debian
    sudo apt-get install graphviz

    # On macOS with Homebrew
    brew install graphviz

    # On Windows
    # Download from https://graphviz.org/download/
.. currentmodule:: xlnstorch

.. _vizualization-doc:

Vizualization
=============

The ``xlnstorch.viz`` submodule provides tools for vizualizing and analyzing
LNS operations and their numerical properties. This module includes functionality
for creating error heatmaps, generating autograd graphs, and analyzing the
precision characteristics of LNS arithmetic operations.

Quick Start
-----------

Here's a simple example of vizualizing the error characteristics of an LNS operation:

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
        f=8, b=23
    )

    # Plot the error heatmap
    fig, ax = plt.subplots(figsize=(8, 6))
    plot_error_heatmap(errors, xs, ys, ax=ax)
    plt.show()

For vizualizing autograd graphs with LNS tensors:

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
    graph.render('autograd_graph', format='png')

Error Analysis Functions
------------------------

The error analysis tools help understand the numerical behavior of LNS operations
compared to exact arithmetic.

.. autosummary::
    :toctree: generated
    :nosignatures:

    viz.make_error_grid
    viz.plot_error_heatmap

Autograd vizualization
----------------------

The autograd vizualization tools provide insight into the computational graph
structure when using LNS tensors.

.. autosummary::
    :toctree: generated
    :nosignatures:

    viz.graph.make_autograd_graph

Error Grid Generation
---------------------

:func:`viz.make_error_grid` generates uniformly-sampled grids of differences
between xlnstorch operations and exact reference computations using high-precision
Decimal arithmetic. This is useful for understanding how LNS errors vary across
different input ranges.

**Unary Operations**

For unary operations (like ``torch.exp``, ``torch.log``), provide only ``x_range``:

.. code-block:: python

    # Analyze exponential function errors
    xs, errors = make_error_grid(
        torch.exp,
        x_range=(-5.0, 5.0),
        steps=200,
        f=8, b=16
    )

**Binary Operations**

For binary operations (like ``torch.mul``, ``torch.add``), provide both ranges:

.. code-block:: python

    # Analyze multiplication errors across 2D input space
    xs, ys, errors = make_error_grid(
        torch.mul,
        x_range=(-1.0, 1.0),
        y_range=(-1.0, 1.0),
        steps=150,
        f=8, b=23
    )

**Custom Reference Functions**

You can provide custom reference functions for operations not in the default mapping:

.. code-block:: python

    def exact_sigmoid(x):
        return 1 / (1 + (-x).exp())

    xs, errors = make_error_grid(
        torch.sigmoid,
        ideal_op=exact_sigmoid,
        x_range=(-10.0, 10.0),
        f=8, b=23
    )

Error vizualization
-------------------

:func:`viz.plot_error_heatmap` creates visual representations of error grids:

- **Unary operations**: Displayed as horizontal color stripes
- **Binary operations**: Displayed as 2D heatmaps

.. code-block:: python

    import matplotlib.pyplot as plt

    # Create subplot for multiple error comparisons
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Compare different precision settings
    for i, (f, b) in enumerate([(8, 16), (8, 23)]):
        xs, ys, errors = make_error_grid(
            torch.mul, 
            x_range=(-2, 2), 
            y_range=(-2, 2),
            f=f, b=b
        )
        plot_error_heatmap(errors, xs, ys, ax=axes[i])
        axes[i].set_title(f'f={f}, b={b}')

    plt.tight_layout()
    plt.show()

Autograd Graph vizualization
----------------------------

:func:`viz.graph.make_autograd_graph` creates graphviz vizualizations of PyTorch's
autograd computation graphs, with special support for LNS tensors. This is
particularly useful for:

- Debugging gradient flows in LNS models
- Understanding computational graph structure
- vizualizing parameter relationships

**Basic Usage**

.. code-block:: python

    # Simple computation graph
    x = xltorch.randn(2, 2, requires_grad=True)
    y = x.pow(2).sum()
    
    graph = make_autograd_graph(y)
    graph.view()  # Opens in default viewer

**Advanced Features**

.. code-block:: python

    # More complex graph with parameter highlighting
    model = xltorch.nn.LNSLinear(10, 5)
    x = xltorch.randn(3, 10, requires_grad=True)
    output = model(x)
    loss = output.sum()

    # Highlight model parameters in the graph
    graph = make_autograd_graph(
        loss,
        params={
            'input': x,
            'weight': model.weight,
            'bias': model.bias
        },
        show_saved=True,
        leaf_color='lightblue',
        node_color='lightgrey'
    )
    
    graph.render('model_graph', format='svg')

**Customization Options**

The vizualization can be customized with various parameters:

- ``show_saved=True``: Display saved tensors within function nodes
- ``leaf_color``, ``node_color``, ``output_color``: Control node colors
- ``node_attr``, ``edge_attr``: Set graphviz attributes for fine-tuning

Dependencies
------------

The vizualization module has optional dependencies:

- **matplotlib**: Required for ``plot_error_heatmap``
- **numpy**: Required for ``plot_error_heatmap``
- **graphviz**: Required for ``make_autograd_graph``

Install with:

.. code-block:: bash

    pip install matplotlib numpy graphviz

Note that for ``graphviz``, you may also need to install the system graphviz libraries:

.. code-block:: bash

    # On Ubuntu/Debian
    sudo apt-get install graphviz

    # On macOS with Homebrew
    brew install graphviz

    # On Windows
    # Download from https://graphviz.org/download/

Examples
--------

**Comparing LNS Configurations**

.. code-block:: python

    import matplotlib.pyplot as plt
    from xlnstorch.viz import make_error_grid, plot_error_heatmap

    configs = [(8, 16), (8, 23), (16, 32)]
    fig, axes = plt.subplots(1, len(configs), figsize=(15, 4))

    for i, (f, b) in enumerate(configs):
        xs, ys, errors = make_error_grid(
            torch.add,
            x_range=(0.1, 10.0),
            y_range=(0.1, 10.0),
            f=f, b=b,
            steps=100
        )
        plot_error_heatmap(errors, xs, ys, ax=axes[i])
        axes[i].set_title(f'LNS({f},{b}) Addition Errors')

    plt.tight_layout()
    plt.show()

**Analyzing Unary Function Errors**

.. code-block:: python

    functions = [torch.exp, torch.log, torch.sqrt]
    fig, axes = plt.subplots(len(functions), 1, figsize=(8, 12))

    for i, func in enumerate(functions):
        if func == torch.log:
            x_range = (0.1, 10.0)  # Avoid log(0)
        elif func == torch.sqrt:
            x_range = (0.0, 10.0)  # Avoid sqrt of negative
        else:
            x_range = (-5.0, 5.0)

        xs, errors = make_error_grid(func, x_range=x_range, f=8, b=23)
        plot_error_heatmap(errors, xs, ax=axes[i])
        axes[i].set_title(f'{func.__name__} Error Analysis')

    plt.tight_layout()
    plt.show()
.. currentmodule:: xlnstorch

.. _error-analysis-doc:

Error Analysis
==============

This section covers the tools for analyzing numerical errors in LNS operations
compared to exact arithmetic. These functions help understand how LNS errors
vary across different input ranges for single precision configurations.

Error Grid Generation
---------------------

:func:`viz.make_error_grid` generates uniformly-sampled grids of differences
between xlnstorch operations and exact reference computations using high-precision
Decimal arithmetic. This is useful for understanding how LNS errors vary across
different input ranges.

**Unary Operations**

For unary operations (like ``torch.exp``, ``torch.log``), provide only ``x_range``:

.. code-block:: python

    import torch
    from xlnstorch.viz import make_error_grid

    # Analyze exponential function errors
    xs, errors = make_error_grid(
        torch.exp,
        x_range=(-5.0, 5.0),
        steps=200,
        f=8,
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
        f=8,
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
        f=8,
    )

Error Visualization
-------------------

:func:`viz.plot_error_heatmap` creates visual representations of error grids:

- **Unary operations**: Displayed as horizontal color stripes
- **Binary operations**: Displayed as 2D heatmaps

.. code-block:: python

    import matplotlib.pyplot as plt
    import torch
    from xlnstorch.viz import make_error_grid, plot_error_heatmap

    # Analyze a single operation
    xs, ys, errors = make_error_grid(
        torch.mul,
        x_range=(-2.0, 2.0),
        y_range=(-2.0, 2.0),
        f=8,
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    plot_error_heatmap(errors, xs, ys, ax=ax)
    ax.set_title('LNS (f=8) Multiplication Errors')
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

        xs, errors = make_error_grid(func, x_range=x_range, f=8)
        plot_error_heatmap(errors, xs, ax=axes[i])
        axes[i].set_title(f'{func.__name__} Error Analysis')

    plt.tight_layout()
    plt.show()

Range Analysis
--------------

:func:`viz.plot_staircase` visualizes the staircase representation of LNS numbers
and shows how different precisions affect the representation of numbers.

.. code-block:: python

    fig, ax = plt.subplots()
    xlnstorch.viz.plot_staircase(ax, [4, 6, 8], -60, 60)
    plt.show()

:func:`viz.plot_spacing_heatmap` visualizes the spacing of representations of LNS
numbers across different precisions. This helps understand how LNS numbers are
distributed and how precision affects their representation.

.. code-block:: python

    xlnstorch.viz.plot_spacing_heatmap([3, 4, 5, 6, 7, 8], -5, 5, step=0.05, rows=2)
    plt.show()
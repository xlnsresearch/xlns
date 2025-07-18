.. currentmodule:: xlnstorch

.. _precision-comparison-doc:

Precision Comparison
====================

This section covers tools and examples for comparing the numerical behavior
of LNS operations across different precision configurations. These visualizations
help understand the trade-offs between precision and **theoretical** computational
efficiency. Note that in this library's simulated LNS, computational efficiency does
not vary with precision. 

Precision Sweep Analysis
------------------------

:func:`viz.precision_sweep_analysis` is the core function for comparing different
precision configurations. It automatically tests multiple precision levels and
computes error metrics for each.

**Basic Precision Sweep**

.. code-block:: python

    import matplotlib.pyplot as plt
    import torch
    from xlnstorch.viz import precision_sweep_analysis, plot_precision_comparison

    # Analyze multiplication across different precisions
    results = precision_sweep_analysis(
        torch.mul,
        x_range=(-2.0, 2.0),
        y_range=(-2.0, 2.0),
        precisions=[4, 6, 8, 10, 12, 16, 20, 24],
        steps=100
    )

    # Plot how maximum error changes with precision
    plot_precision_comparison(results, metric='max_error')
    plt.show()

**Unary Operations**

.. code-block:: python

    # Analyze unary operations (no y_range needed)
    results = precision_sweep_analysis(
        torch.exp,
        x_range=(-5.0, 5.0),
        precisions=[4, 8, 12, 16, 20],
        steps=200
    )

Visualizing Precision Comparisons
---------------------------------

**Error Metric Plots**

:func:`viz.plot_precision_comparison` creates line plots showing how different
error metrics change with precision:

.. code-block:: python

    import matplotlib.pyplot as plt
    from xlnstorch.viz import precision_sweep_analysis, plot_precision_comparison

    # Analyze division operation
    results = precision_sweep_analysis(
        torch.div,
        x_range=(1.0, 10.0),
        y_range=(0.1, 5.0),
        steps=100
    )

    # Create subplot comparing different metrics
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    metrics = ['max_error', 'mean_error', 'median_error', 'std_error']
    for i, metric in enumerate(metrics):
        row, col = i // 2, i % 2
        plot_precision_comparison(results, metric=metric, ax=axes[row, col])

    plt.tight_layout()
    plt.show()

**Heatmap Grid Visualization**

:func:`viz.plot_precision_heatmap_grid` creates a grid of heatmaps showing
error patterns at different precision levels:

.. code-block:: python

    from xlnstorch.viz import precision_sweep_analysis, plot_precision_heatmap_grid

    # Analyze power operation
    results = precision_sweep_analysis(
        torch.pow,
        x_range=(0.5, 3.0),
        y_range=(0.5, 3.0),
        precisions=[6, 8, 10, 12, 16, 20],
        steps=80
    )

    # Create heatmap grid
    fig, axes = plot_precision_heatmap_grid(results, figsize=(15, 10))
    plt.show()

**Logarithmic vs Linear Scaling**

.. code-block:: python

    # Compare log vs linear scaling for error visualization
    results = precision_sweep_analysis(
        torch.mul,
        x_range=(-1.0, 1.0),
        y_range=(-1.0, 1.0),
        steps=100
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    plot_precision_comparison(results, metric='max_error', 
                             log_scale=True, ax=ax1)
    ax1.set_title('Log Scale')

    plot_precision_comparison(results, metric='max_error', 
                             log_scale=False, ax=ax2)
    ax2.set_title('Linear Scale')

    plt.tight_layout()
    plt.show()

Comparing Different Operations
------------------------------

**Multi-Operation Analysis**

Compare how different operations behave across precision levels:

.. code-block:: python

    operations = [
        ('Addition', torch.add),
        ('Multiplication', torch.mul),
        ('Division', torch.div),
        ('Power', torch.pow)
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    for i, (name, op) in enumerate(operations):
        row, col = i // 2, i % 2
        
        results = precision_sweep_analysis(
            op,
            x_range=(0.5, 5.0),
            y_range=(0.5, 5.0),
            precisions=[4, 6, 8, 10, 12, 16],
            steps=80
        )
        
        plot_precision_comparison(results, metric='max_error', ax=axes[row, col])
        axes[row, col].set_title(f'{name} Operation')

    plt.tight_layout()
    plt.show()

**Operation-Specific Heatmap Grids**

.. code-block:: python

    # Compare heatmap patterns for different operations
    operations = [torch.add, torch.mul, torch.div]

    for i, op in enumerate(operations):
        results = precision_sweep_analysis(
            op,
            x_range=(0.1, 5.0),
            y_range=(0.1, 5.0),
            precisions=[4, 6, 8, 10, 12, 16, 20, 24],
            steps=100
        )

        fig, axes = plot_precision_heatmap_grid(results, figsize=(12, 6))
        plt.show()

Range-Dependent Analysis
------------------------

**Input Range Sensitivity**

Different input ranges may show different precision sensitivity:

.. code-block:: python

    ranges = [
        ('Small values', (0.01, 0.1), (0.01, 0.1)),
        ('Unit range', (0.1, 1.0), (0.1, 1.0)),
        ('Medium values', (1.0, 10.0), (1.0, 10.0)),
        ('Large values', (10.0, 100.0), (10.0, 100.0))
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    for i, (name, x_range, y_range) in enumerate(ranges):
        row, col = i // 2, i % 2
        
        results = precision_sweep_analysis(
            torch.mul,
            x_range=x_range,
            y_range=y_range,
            precisions=[4, 6, 8, 10, 12, 16],
            steps=100
        )
        
        plot_precision_comparison(results, metric='max_error', ax=axes[row, col])
        axes[row, col].set_title(f'{name}')

    plt.suptitle('Range Sensitivity for Multiplication')
    plt.tight_layout()
    plt.show()

**Unary Function Range Analysis**

.. code-block:: python

    # Analyze how different input ranges affect unary operations
    exp_ranges = [
        ('Small exp', (-2.0, 2.0)),
        ('Medium exp', (-5.0, 5.0)),
        ('Large exp', (-10.0, 10.0))
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for i, (name, x_range) in enumerate(exp_ranges):
        results = precision_sweep_analysis(
            torch.exp,
            x_range=x_range,
            precisions=[4, 6, 8, 10, 12, 16, 20],
            steps=200
        )
        
        plot_precision_comparison(results, metric='max_error', ax=axes[i])
        axes[i].set_title(name)

    plt.tight_layout()
    plt.show()

Error Statistics Analysis
-------------------------

**Extracting and Analyzing Results**

The results from `precision_sweep_analysis` contain detailed error statistics:

.. code-block:: python

    results = precision_sweep_analysis(
        torch.add,
        x_range=(-2.0, 2.0),
        y_range=(-2.0, 2.0),
        steps=100
    )

    # Print summary statistics
    print("Precision Analysis Summary")
    print("=" * 50)
    print(f"{'f':<4} {'Max Error':<12} {'Mean Error':<12} {'Std Error':<12}")
    print("-" * 50)

    for f in sorted(results.keys()):
        stats = results[f]
        print(f"{f:<4} {stats['max_error']:<12.2e} "
              f"{stats['mean_error']:<12.2e} {stats['std_error']:<12.2e}")

**Finding Optimal Precision**

.. code-block:: python

    def find_optimal_precision(results, error_threshold=1e-6):
        """Find minimum precision that meets error threshold."""
        for f in sorted(results.keys()):
            if results[f]['max_error'] <= error_threshold:
                return f
        return None

    # Find minimum precision for different error thresholds
    thresholds = [1e-4, 1e-5, 1e-6, 1e-7]
    
    for threshold in thresholds:
        optimal_f = find_optimal_precision(results, threshold)
        if optimal_f:
            print(f"For max error ≤ {threshold:.0e}: f ≥ {optimal_f}")
        else:
            print(f"For max error ≤ {threshold:.0e}: Need f > 24")

**Custom Analysis Functions**

.. code-block:: python

    def analyze_precision_efficiency(operation, x_range, y_range=None):
        """Comprehensive precision analysis with efficiency metrics."""
        results = precision_sweep_analysis(
            operation,
            x_range=x_range,
            y_range=y_range,
            precisions=[4, 6, 8, 10, 12, 16, 20, 24],
            steps=100
        )
        
        # Create comprehensive visualization
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
        
        # Plot different metrics
        plot_precision_comparison(results, metric='max_error', ax=ax1)
        ax1.set_title('Maximum Error')
        
        plot_precision_comparison(results, metric='mean_error', ax=ax2)
        ax2.set_title('Mean Error')
        
        plot_precision_comparison(results, metric='std_error', ax=ax3)
        ax3.set_title('Standard Deviation')
        
        # Error reduction rate
        precisions = sorted(results.keys())
        max_errors = [results[f]['max_error'] for f in precisions]
        reduction_rate = [max_errors[i-1]/max_errors[i] if i > 0 else 1 
                         for i in range(len(max_errors))]
        
        ax4.plot(precisions[1:], reduction_rate[1:], 'o-')
        ax4.set_xlabel('Precision (f parameter)')
        ax4.set_ylabel('Error Reduction Factor')
        ax4.set_title('Error Reduction per Precision Increase')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig, results

    # Use the analysis function
    fig, results = analyze_precision_efficiency(
        torch.mul, 
        x_range=(-1.0, 1.0), 
        y_range=(-1.0, 1.0)
    )
    plt.show()

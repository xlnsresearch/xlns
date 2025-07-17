import matplotlib.pyplot as plt
import torch
import xlnstorch.viz as viz

def demo_error_visualization():
    """
    Demonstrate error visualization for LNS operations.

    This function shows how to analyze the accuracy of LNS operations
    by comparing them to exact decimal arithmetic.
    """
    print("=== Error Visualization Demo ===")

    # Demo 1: Unary operation error analysis
    print("1. Analyzing sqrt operation accuracy...")

    xs, err_sqrt = viz.make_error_grid(
        torch.sqrt,
        x_range=(0.1, 10.0), # sqrt domain must be positive
        steps=101,
        f=8, # 8 bits of fractional precision
        absolute=True
    )

    # Find max error
    max_error = torch.max(err_sqrt)
    avg_error = torch.mean(err_sqrt)

    print(f"  Max absolute error: {max_error:.2e}")
    print(f"  Average absolute error: {avg_error:.2e}")

    # Plot error as heatmap
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Plot sqrt error as heatmap stripe
    viz.plot_error_heatmap(err_sqrt, xs, ax=ax1, cmap="plasma")
    ax1.set_title("sqrt() Error Analysis")

    # Demo 2: Binary operation error analysis
    print("2. Analyzing multiplication operation accuracy...")

    xs_mul, ys_mul, err_mul = viz.make_error_grid(
        torch.mul,
        x_range=(-2.0, 2.0),
        y_range=(-2.0, 2.0),
        steps=51,
        f=8,
        absolute=True
    )

    max_mul_error = torch.max(err_mul)
    avg_mul_error = torch.mean(err_mul)

    print(f"  Max absolute error: {max_mul_error:.2e}")
    print(f"  Average absolute error: {avg_mul_error:.2e}")

    # Plot multiplication error as 2D heatmap
    viz.plot_error_heatmap(err_mul, xs_mul, ys_mul, ax=ax2, cmap="viridis")
    ax2.set_title("Multiplication Error Analysis")

    plt.tight_layout()
    plt.savefig("lns_error_analysis.png", dpi=150, bbox_inches='tight')
    print("  Error plots saved as 'lns_error_analysis.png'")
    plt.close()

    # Demo 3: Compare different precisions
    print("3. Comparing error across different precisions...")

    results = viz.precision_sweep_analysis(
        torch.mul,
        precisions=[4, 6, 8, 10, 12, 14, 16, 18],
        x_range=(-2.0, 2.0),
        y_range=(-2.0, 2.0),
        steps=51
    )

    # Create comparison plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    viz.plot_precision_comparison(results, ax=ax1, metric='max_error')
    viz.plot_precision_comparison(results, ax=ax2, metric='mean_error')

    plt.tight_layout()
    plt.savefig('precision_comparison.png', dpi=150, bbox_inches='tight')
    print("  Precision comparison plots saved as 'precision_comparison.png'")
    plt.close()

    # Create heatmap grid
    fig, axes = viz.plot_precision_heatmap_grid(results)
    plt.savefig('precision_heatmaps.png', dpi=150, bbox_inches='tight')
    print("  Precision heatmap plots saved as 'precision_heatmaps.png'")
    plt.close()

def main():
    print("xlnstorch Visualization Demo")
    print("=" * 40)
    print()

    # Run all demos
    demo_error_visualization()

    print("\n" + "=" * 40)
    print("Demo completed!")
    print("\nGenerated files:")
    print("  - lns_error_analysis.png (if matplotlib available)")
    print("  - lns_autograd_graph.png (if graphviz available)")
    print("  - lns_autograd_graph.dot (if graphviz available)")
    print("\nTo install optional dependencies:")
    print("  pip install matplotlib graphviz")

if __name__ == "__main__":
    main()

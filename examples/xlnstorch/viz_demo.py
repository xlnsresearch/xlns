import matplotlib.pyplot as plt
import torch
import xlnstorch as xltorch
from xlnstorch.viz import make_error_grid, plot_error_heatmap
from xlnstorch.viz.graph import make_autograd_graph

def demo_error_visualization():
    """
    Demonstrate error visualization for LNS operations.

    This function shows how to analyze the accuracy of LNS operations
    by comparing them to exact decimal arithmetic.
    """
    print("=== Error Visualization Demo ===")

    # Demo 1: Unary operation error analysis
    print("1. Analyzing sqrt operation accuracy...")

    xs, err_sqrt = make_error_grid(
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
    plot_error_heatmap(err_sqrt, xs, ax=ax1, cmap="plasma")
    ax1.set_title("sqrt() Error Analysis")

    # Demo 2: Binary operation error analysis
    print("2. Analyzing multiplication operation accuracy...")

    xs_mul, ys_mul, err_mul = make_error_grid(
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
    plot_error_heatmap(err_mul, xs_mul, ys_mul, ax=ax2, cmap="viridis")
    ax2.set_title("Multiplication Error Analysis")

    plt.tight_layout()
    plt.savefig("lns_error_analysis.png", dpi=150, bbox_inches='tight')
    print("  Error plots saved as 'lns_error_analysis.png'")
    plt.close()

    # Demo 3: Compare different precisions
    print("3. Comparing error across different precisions...")

    precisions = [4, 8, 12, 16]
    errors = []

    for f in precisions:
        _, _, err = make_error_grid(
            torch.add,
            x_range=(-1.0, 1.0),
            y_range=(-1.0, 1.0),
            steps=21,
            f=f,
            absolute=True
        )
        avg_err = torch.mean(err)
        errors.append(avg_err)
        print(f"  f={f:2d} bits: avg error = {avg_err:.2e}")

def demo_autograd_visualization():
    """
    Demonstrate autograd graph visualization.

    This function shows how to visualize the computation graph
    for LNS operations with automatic differentiation.
    """
    print("\n=== Autograd Graph Visualization Demo ===")

    # Create a simple neural network computation
    print("1. Creating a simple computation graph...")

    # Input data
    x = xltorch.lnstensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)

    # Simple "neural network" weights
    w1 = xltorch.lnstensor([[0.5, 0.3], [0.2, 0.7]], requires_grad=True)
    w2 = xltorch.lnstensor([[0.8], [0.6]], requires_grad=True)

    # Forward pass
    h = torch.matmul(x, w1)      # hidden layer
    h_activated = torch.tanh(h)   # activation
    output = torch.matmul(h_activated, w2)  # output layer
    loss = torch.sum(output * output)  # simple loss (sum of squares)

    print(f"  Input: {x}")
    print(f"  Output: {output}")
    print(f"  Loss: {loss}")

    # Create parameter dictionary for better visualization
    params = {
        "input_x": x,
        "weight_1": w1,
        "weight_2": w2,
    }

    # Generate autograd graph
    try:
        print("2. Generating autograd graph...")

        graph = make_autograd_graph(
            loss,
            graph_name="LNS Neural Network",
            show_saved=True,
            params=params,
            leaf_color="lightblue",
            node_color="lightgrey",
            output_color="orange"
        )

        # Save the graph
        graph.render("lns_autograd_graph", format="png", cleanup=True)
        print("  Autograd graph saved as 'lns_autograd_graph.png'")

        # Also save as DOT file for inspection
        with open("lns_autograd_graph.dot", "w") as f:
            f.write(graph.source)
        print("  Graph source saved as 'lns_autograd_graph.dot'")

    except ImportError:
        print("  graphviz not available - skipping graph visualization")
        print("  Install with: pip install graphviz")

    # Compute gradients
    print("3. Computing gradients...")
    loss.backward()

    print(f"  x.grad: {x.grad}")
    print(f"  w1.grad: {w1.grad}")
    print(f"  w2.grad: {w2.grad}")

def main():
    print("xlnstorch Visualization Demo")
    print("=" * 40)
    print()

    # Run all demos
    demo_error_visualization()
    demo_autograd_visualization()

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

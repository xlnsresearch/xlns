import torch
import xlnstorch as xltorch
from xlnstorch.viz.graph import make_autograd_graph

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
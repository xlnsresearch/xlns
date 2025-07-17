import matplotlib.pyplot as plt
import xlnstorch.viz.range as viz_range

def demo_staircase_plots():
    """
    Demonstrate staircase visualization for different LNS precisions.
    """
    print("=== Staircase Visualization Demo ===")

    # Demo 1: Multiple precision comparison
    print("1. Comparing staircase plots for multiple precisions...")

    fig, ax = plt.subplots()

    viz_range.plot_staircase(
        ax=ax,
        f_range=[4, 6, 8],
        low=-60,
        high=60,
        step=1,
    )

    ax.set_title("LNS Staircase Representation Comparison")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    plt.savefig("staircase_precision_comparison.png", dpi=150, bbox_inches='tight')
    print("  Multiple precision comparison saved as 'staircase_precision_comparison.png'")
    plt.close()

def demo_spacing_heatmaps():
    """
    Demonstrate spacing difference heatmaps for LNS precisions.

    This function visualizes how LNS spacing differs from ideal float64
    spacing across different value ranges and precisions.
    """
    print("\n=== Spacing Heatmap Demo ===")

    # Demo 1: Multiple precision comparison
    print("1. Comparing spacing heatmaps for multiple precisions...")

    axes = viz_range.plot_spacing_heatmap(
        f_range=[4, 6, 8, 10],
        low=0.5,
        high=8.0,
        step=0.05,
        rows=8,
        n_cols=2,
        levels=40,
        cmap='plasma'
    )

    plt.suptitle("LNS Spacing Difference Comparison", fontsize=16)
    plt.tight_layout()
    plt.show()
    plt.savefig("spacing_heatmap_comparison.png", dpi=150, bbox_inches='tight')
    print("  Multiple precision comparison saved as 'spacing_heatmap_comparison.png'")
    plt.close()

    # Demo 2: Wide range analysis
    print("2. Wide range spacing analysis...")
    
    axes = viz_range.plot_spacing_heatmap(
        f_range=[8, 12, 16],
        low=0.01,
        high=100.0,
        step=0.5,
        rows=6,
        n_cols=3,
        levels=35,
        cmap='coolwarm'
    )

    plt.suptitle("Wide Range LNS Spacing Analysis", fontsize=16)
    plt.tight_layout()
    plt.show()
    plt.savefig("spacing_heatmap_widerange.png", dpi=150, bbox_inches='tight')
    print("  Wide range analysis saved as 'spacing_heatmap_widerange.png'")
    plt.close()

def main():
    """
    Run all numerical range visualization demos.
    """
    print("xlnstorch Numerical Range Visualization Demo")
    print("=" * 50)
    print()

    # Run all demos
    demo_staircase_plots()
    demo_spacing_heatmaps()

    print("\n" + "=" * 50)
    print("Demo completed successfully!")
    print("\nGenerated files:")
    print("  - staircase_precision_comparison.png") 
    print("  - spacing_heatmap_comparison.png")
    print("  - spacing_heatmap_widerange.png")

if __name__ == "__main__":
    main()
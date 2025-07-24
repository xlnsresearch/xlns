
import matplotlib.pyplot as plt
import xlnstorch.viz.range as viz_range

def demo_lns_error_heatmaps():
    """
    Demonstrate the error heatmap for LNS representations across
    different precisions in comparison to float64.
    """
    print("\n=== LNS Error Heatmap Demo ===")

    # Demo 1: Multiple precision comparison
    print("1. Comparing LNS error heatmaps for multiple precisions...")

    axes = viz_range.plot_lns_error_heatmap(
        f_range=[4, 6, 8, 10, 12, 14, 16, 18, 20],
        low=-5,
        high=5,
        steps=1000,
        n_cols=3,
    )

    plt.suptitle("LNS Error Comparison", fontsize=16)
    plt.tight_layout()
    plt.show()
    plt.savefig("repr_error_heatmap_comparison.png", dpi=150, bbox_inches='tight')
    print("  Multiple precision comparison saved as 'repr_error_heatmap_comparison.png'")
    plt.close()

def main():
    """
    Run all distribution plot visualization demos.
    """
    print("xlnstorch Distribution Visualization Demo")
    print("=" * 50)
    print()

    # run all demos
    demo_lns_error_heatmaps()

    print("\n" + "=" * 50)
    print("Demo completed successfully!")
    print("\nGenerated files:")
    print("  - repr_error_heatmap_comparison.png")

if __name__ == "__main__":
    main()
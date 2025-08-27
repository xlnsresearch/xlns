from typing import List, Tuple, Union
import torch
import math
from xlnstorch import LNSTensor, lnstensor, LNS_ZERO
import xlnstorch.tensor_utils as tensor_utils

def plot_lns_error_heatmap(
        f_range: Union[Tuple[int], List[int], int],
        low: float,
        high: float,
        steps: int = 1000,
        rows: int = 5,
        n_cols: int = 2,
        levels: int = 50,
        cmap : str = 'viridis'
    ):
    """
    Create a heatmap showing the error in LNS representation for various precisions.

    Parameters
    ----------
    f_range : Tuple[int] | List[int] | int
        The range of LNS precisions to visualize. Can be a single integer,
        a list of integers, or a tuple of integers.
    low : float
        The lower bound of the range to visualize.
    high : float
        The upper bound of the range to visualize.
    steps : int, optional
        The number of steps to use for the range, by default 1000.
    rows : int, optional
        The number of rows to display for each precision, by default 5.
        This is the height of each panel in the heatmap.
    n_cols : int, optional
        The number of columns in the heatmap layout, by default 2.
        This determines how many different precisions are shown side by side.
    levels : int, optional
        The number of contour levels to use in the heatmap, by default 50.
    cmap : str, optional
        The colormap to use for the heatmap, by default 'viridis'.

    Returns
    -------
    matplotlib.axes.Axes
        The axes containing the heatmap.

    Raises
    ------
    ImportError
        If matplotlib is not installed.
    ValueError
        If the 'low' value is greater than or equal to the 'high' value.
    """
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.axes_grid1 import make_axes_locatable
    except ImportError:
        raise ImportError("matplotlib is required for plotting repr error heatmaps")

    if low >= high:
        raise ValueError("The 'low' value must be less than the 'high' value.")

    if isinstance(f_range, int):
        f_range = [f_range]

    panel_data = []

    # Create a range of float values over [low, high].
    float_range = torch.linspace(low, high, steps=steps, dtype=torch.float64)
    X = float_range.repeat(rows, 1)
    Y = torch.linspace(0, 1, steps=rows).repeat(X.size(1), 1).T

    for f in f_range:
        lns_range = lnstensor(float_range, f=f).value
        delta_range = (lns_range - float_range).abs()

        # this is the absolute difference in spacing between the LNS
        # representations and the float64 representations.
        Z = delta_range.repeat(rows, 1)
        panel_data.append((f, Z))

    # Create a grid of subplots
    n_panels = len(f_range)
    n_rows = math.ceil(n_panels / n_cols)

    figsize = (4.5 * n_cols, 2.2 * n_rows)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
    fig.subplots_adjust(hspace=0.35, wspace=0.35) # Adjust spacing between subplots

    for ax, (f, Z) in zip(axes.flat, panel_data):
        vmin, vmax = Z.min().item(), Z.max().item()
        cs = ax.contourf(X, Y, Z, levels=levels, cmap=cmap, vmin=vmin, vmax=vmax, extend='both')

        ax.set_title(f"f = {f}")
        ax.set_yticks([])
        ax.set_xlim(low, high)

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="4%", pad=0.05)
        fig.colorbar(cs, cax=cax, label="spacing difference",
                     extendrect=True)
        cax.tick_params(labelsize=7)

    # Hide any unused subplots
    for ax in axes.flat[n_panels:]:
        ax.axis("off")

    return axes

def plot_lns_distribution(
        f: int,
        low: float,
        high: float,
        step_size: float = 1.0,
    ):
    """
    Creates a bar plot showing the density of LNS representations
    over a given range for a given precision.

    Parameters
    ----------
    f : int
        The precision for which to create the visualization.
    low : float
        The lower bound of the range to visualize.
    high : float
        The upper bound of the range to visualize.
    step_size : float, optional
        The size of each bin

    Returns
    -------
    matplotlib.axes.Axes
        The axes containing the bar plot.

    Raises
    ------
    ImportError
        If matplotlib is not installed.
    ValueError
        If the 'low' value is greater than or equal to the 'high' value.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib is required for plotting repr error heatmaps")

    if low >= high:
        raise ValueError("The 'low' value must be less than the 'high' value.")

    base = tensor_utils.get_base_from_precision(f)

    bin_bounds = torch.arange(low, high + step_size / 2, step=step_size)
    n_bins = len(bin_bounds) - 1
    ticks = [f"{bin_bounds[i]} - {bin_bounds[i + 1]}" for i in range(n_bins)]
    counts = [0] * n_bins

    zero_log = LNS_ZERO.to(torch.int64) >> 1
    for i in range(n_bins):

        bin_start, bin_end = bin_bounds[i], bin_bounds[i + 1]
        bin_start_lns = LNSTensor.get_internal_tensor(bin_start, base).to(torch.int64)
        bin_end_lns = LNSTensor.get_internal_tensor(bin_end, base).to(torch.int64)

        bin_start_log, bin_start_sign = bin_start_lns >> 1, bin_start_lns & 1
        bin_end_log, bin_end_sign = bin_end_lns >> 1, bin_end_lns & 1

        # both bounds are negative
        if bin_start_sign == 1 and bin_end_sign == 1:
            counts[i] = bin_start_log - bin_end_log + 1
        # both bounds are negative
        elif bin_start_sign == 0 and bin_end_sign == 0:
            counts[i] = bin_end_log - bin_start_log + 1
        # mixed bounds
        else:
            counts[i] = bin_start_log + bin_end_log - 2 * zero_log + 1

    fig_width = max(6, 0.6 * n_bins) # scale width to the number of bins
    fig, ax = plt.subplots(figsize=(fig_width, 4))

    ax.bar(range(n_bins), counts, width=0.9, align='center', color="#3182bd")
    ax.set_xticks(range(n_bins))
    ax.set_xticklabels(ticks, rotation=45, ha="right")
    ax.set_ylabel("Number of representable values")
    ax.set_xlabel("Real-value interval")
    ax.set_title(f"Distribution of representable LNS numbers (precision f={f})")

    plt.tight_layout()
    return ax

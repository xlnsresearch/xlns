from typing import List, Tuple, Union
import torch
import math
from xlnstorch import lnstensor

def plot_staircase(
        ax,
        f_range: Union[Tuple[int], List[int], int],
        low: float,
        high: float,
        step: float = 1,
    ):
    """
    Plot staircase functions for different LNS precisions on a logarithmic x-axis.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes object on which to plot the staircase functions.
    f_range : Tuple[int] | List[int] | int
        The range of LNS precisions to visualize. Can be a single integer,
        a list of integers, or a tuple of integers.
    low : float
        The lower bound of the x-axis range.
    high : float
        The upper bound of the x-axis range.
    step : float, optional
        The step size for the x-axis, by default 1.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If the 'low' value is greater than or equal to the 'high' value.
    """
    if low >= high:
        raise ValueError("The 'low' value must be less than the 'high' value.")

    if isinstance(f_range, int):
        f_range = [f_range]

    ax.set_xscale("log")
    ax.set_xlabel("Real value (log scale)")
    ax.set_ylabel("Representable levels")

    for f in f_range:
        float_range = torch.arange(low, high + 2 * step, dtype=torch.float64)
        x = lnstensor(float_range, f=f).value
        ax.stairs(x, fill=False, label=f"f={f}")

    ax.legend()

def plot_spacing_heatmap(
        f_range: Union[Tuple[int], List[int], int],
        low: float,
        high: float,
        step: float = 0.05,
        rows: int = 5,
        n_cols: int = 2,
        levels: int = 50,
        cmap : str = 'viridis'
    ):
    """
    Create a heatmap showing a heatmap of the spacing differences
    for various LNS precisions.

    Parameters
    ----------
    f_range : Tuple[int] | List[int] | int
        The range of LNS precisions to visualize. Can be a single integer,
        a list of integers, or a tuple of integers.
    low : float
        The lower bound of the range to visualize.
    high : float
        The upper bound of the range to visualize.
    step : float, optional
        The step size for the range, by default 0.05.
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
        raise ImportError("matplotlib is required for plotting spacing heatmaps")

    if low >= high:
        raise ValueError("The 'low' value must be less than the 'high' value.")

    if isinstance(f_range, int):
        f_range = [f_range]

    panel_data = []

    # Create a range of float values over [low, high + step] with the specified step
    # so that the spacing differences can be plotted over [low, high].
    float_range = torch.arange(low, high + step * 2, step=step, dtype=torch.float64)
    X = float_range[:-1].repeat(rows, 1)
    Y = torch.linspace(0, 1, steps=rows).repeat(X.size(1), 1).T

    delta_float_range = (float_range[1:] - float_range[:-1]).abs()
    for f in f_range:
        lns_range = lnstensor(float_range, f=f).value
        delta_lns_range = (lns_range[1:] - lns_range[:-1]).abs()
        resolution_range = (delta_lns_range - delta_float_range).abs()

        # this is the absolute difference in spacing between the LNS
        # representations and the float64 representations.
        Z = resolution_range.abs().repeat(rows, 1)
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
from __future__ import annotations
from decimal import Decimal, getcontext
from typing import Callable, Tuple, List, Dict, Any, Optional, Union
import torch
from xlnstorch import LNSTensor, lnstensor

def _to_dec(x: Union[float, torch.Tensor, LNSTensor]) -> Decimal:
    """Convert a float or tensor to an *exact* Decimal"""
    if isinstance(x, torch.Tensor):
        x = float(x)
    elif isinstance(x, LNSTensor):
        x = float(x.value)
    return Decimal(repr(x))

def _decimal_linspace(lo: float, hi: float, steps: int) -> List[Decimal]:
    "Return `steps` evenly-spaced Decimal numbers in [lo, hi]."
    start = Decimal(str(lo))
    stop  = Decimal(str(hi))
    if steps == 1:
        return [start]
    delta = (stop - start) / (steps - 1)
    return [start + i * delta for i in range(steps)]

# A dictionary mapping torch operations to their ideal Decimal counterparts.
# This is used to compute the "exact" reference values for error grids.
operator_reference = {
    torch.mul: lambda x, y: x * y,
    torch.add: lambda x, y: x + y,
    torch.sub: lambda x, y: x - y,
    torch.div: lambda x, y: x / y,
    torch.pow: lambda x, y: x ** y,

    torch.neg: lambda x: -x,
    torch.abs: lambda x: abs(x),
    torch.sqrt: lambda x: x.sqrt(),
    torch.exp: lambda x: x.exp(),
    torch.log: lambda x: x.ln(),
}

def make_error_grid(
        op: Callable,
        ideal_op: Optional[Callable] = None,
        *,
        f: Optional[float] = None,
        b: Optional[float] = None,
        x_range: Tuple[float, float] = (-1.0, 1.0),
        y_range: Optional[Tuple[float, float]] = None,
        steps: int = 201,
        device: Union[torch.device, str] = "cpu",
        absolute: bool = True,
        decimal_prec: int = 50,
    ):
    """
    Generate a uniformly-sampled grid of differences between an xlnstorch
    operation `op` and an *exact* reference computed with Decimal.

    Set `decimal_prec` high enough that Decimal is effectively exact
    over the chosen ranges (50 digits is usually plenty up to ~1e15).

    Parameters
    ----------
    op : Callable
        The xlnstorch operation to benchmark, e.g., `torch.mul`.
    ideal_op : Callable, optional
        A reference function that computes the exact result using Decimal.
        If not provided, a default mapping from `op` to an ideal function
        is used if available.
    f : float, optional
        The `f` parameter for the LNSTensor constructor.
    b : float, optional
        The `b` parameter for the LNSTensor constructor.
    x_range : Tuple[float, float], optional
        The range of x values to sample, defaulting to (-1.0, 1.0).
    y_range : Tuple[float, float], optional
        The range of y values to sample for binary operations. If None,
        only a unary operation is performed.
    steps : int, optional
        The number of steps to sample in each dimension. Default is 201.
    device : torch.device or str, optional
        The device on which to create the input tensors (default is "cpu").
    absolute : bool, optional
        If True, return the absolute error; otherwise, return signed error.
        Default is True.
    decimal_prec : int, optional
        The precision for Decimal operations. This should be set high enough
        to ensure that Decimal calculations are effectively exact over the
        specified ranges. Default is 50 digits.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor] or Tuple[torch.Tensor, torch.Tensor]
        A tuple containing:
        - `xs`: A tensor of x values sampled from the specified range.
        - `ys`: A tensor of y values sampled from the specified range (if binary operation).
        If `y_range` is None, only `xs` and `err` are returned.
        - `err`: A tensor of errors, where each element is the absolute or signed
        difference between the computed value and the exact value.
    """
    getcontext().prec = decimal_prec
    ideal_op = ideal_op or operator_reference.get(op)

    xs_dec = _decimal_linspace(x_range[0], x_range[1], steps)
    xs_torch = torch.tensor([float(d) for d in xs_dec], device=device, dtype=torch.float64)
    xs = lnstensor(xs_torch, f=f, b=b)

    # unary operation
    if y_range is None:

        got = op(xs)
        err = torch.empty((steps,), device=device, dtype=torch.float64)

        for i, x_d in enumerate(xs_dec):

            got_dec = _to_dec(got[i])
            exact_dec = ideal_op(x_d)
            diff_dec = got_dec - exact_dec

            if absolute:
                diff_dec = abs(diff_dec)

            err[i] = float(diff_dec)

        return xs_torch, err

    # binary operation
    ys_dec = _decimal_linspace(y_range[0], y_range[1], steps)
    ys_torch = torch.tensor([float(d) for d in ys_dec], device=device, dtype=torch.float64)
    ys = lnstensor(ys_torch, f=f, b=b)

    got_grid = op(xs[:, None], ys[None, :]) # broadcasting equivalent to meshgrid
    err = torch.empty((steps, steps), device=device, dtype=torch.float64)

    for i, x_d in enumerate(xs_dec):
        for j, y_d in enumerate(ys_dec):

            got_dec = _to_dec(got_grid[i, j])
            exact_dec = ideal_op(x_d, y_d)
            diff_dec  = got_dec - exact_dec

            if absolute:
                diff_dec = abs(diff_dec)

            err[i, j] = float(diff_dec)

    return xs_torch, ys_torch, err

def plot_error_heatmap(
        err: torch.Tensor,
        xs: torch.Tensor,
        ys: torch.Tensor | None = None,
        *,
        ax=None,
        cmap: str = "viridis",
        vmin=None,
        vmax=None,
    ):
    """
    Visualise the error tensor returned by `make_error_grid`.  Unary
    errors are drawn as a coloured stripe; binary errors as a 2-D map.

    Parameters
    ----------
    err : torch.Tensor
        The error tensor, typically returned by `make_error_grid`.
    xs : torch.Tensor
        The x values corresponding to the error tensor.
    ys : torch.Tensor, optional
        The y values corresponding to the error tensor. If None, a unary
        operation is assumed and the error is visualised as a single
        horizontal stripe.
    ax : matplotlib.axes.Axes, optional
        The axes on which to plot the heatmap. If None, a new figure and axes
        will be created.
    cmap : str, optional
        The colormap to use for the heatmap. Default is "viridis".
    vmin : float, optional
        The minimum value for the colormap. If None, it will be set to the
        minimum value of the error tensor.
    vmax : float, optional
        The maximum value for the colormap. If None, it will be set to the
        maximum value of the error tensor.

    Returns
    -------
    matplotlib.axes.Axes
        The axes containing the heatmap.

    Raises
    ------
    ImportError
        If `matplotlib` is not installed, an ImportError is raised.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib is required for plotting error heatmaps")

    err_np = err.detach().cpu().numpy()
    xs_np = xs.detach().cpu().numpy()

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))

    if ys is None:
        img = err_np[None, :]
        extent = [xs_np.min(), xs_np.max(), 0, 1]

        im = ax.imshow(
            img,
            extent=extent,
            origin="lower",
            aspect="auto",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_yticks([])
        ax.set_xlabel("x")
        ax.set_title("Absolute error (unary)")

    else:
        ys_np = ys.detach().cpu().numpy()

        im = ax.imshow(
            err_np.T,  # transpose so x = horizontal axis
            extent=[xs_np.min(), xs_np.max(), ys_np.min(), ys_np.max()],
            origin="lower",
            aspect="auto",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title("Absolute error")

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("error (|float - exact|)")
    return ax

def precision_sweep_analysis(
        op: Callable,
        ideal_op: Callable | None = None,
        *,
        precisions: List[int] = None,
        x_range: Tuple[float, float] = (-1.0, 1.0),
        y_range: Tuple[float, float] = None,
        steps: int = 101,
        device: torch.device | str = "cpu"
    ) -> Dict[str, Any]:
    """
    Analyze how different precision levels affect operation accuracy.

    Parameters
    ----------
    op : Callable
        The operation to analyze (e.g., torch.mul, torch.add)
    ideal_op : Callable, optional
        A reference function that computes the exact result using Decimal.
        If not provided, a default mapping from `op` to an ideal function
        is used if available.
    precisions : List[int], optional
        List of precision values (f parameter) to test. 
        Defaults to [4, 6, 8, 10, 12, 16, 20, 24]
    x_range : Tuple[float, float]
        Range of x values to test
    y_range : Tuple[float, float], optional
        Range of y values for binary operations
    steps : int
        Number of sample points per dimension
    device : torch.device | str
        Device for computations

    Returns
    -------
    Dict[str, Any]
        Dictionary containing error metrics for each precision level
    """
    if precisions is None:
        precisions = [4, 6, 8, 10, 12, 16, 20, 24]

    results = {}

    for f in precisions:
        if y_range is None:
            # Unary operation
            xs, err = make_error_grid(op, ideal_op, f=f, x_range=x_range,
                                      steps=steps, device=device)
            results[f] = {
                'max_error': float(torch.max(err)),
                'mean_error': float(torch.mean(err)),
                'median_error': float(torch.median(err)),
                'std_error': float(torch.std(err)),
                'error_tensor': err,
                'xs': xs
            }

        else:
            # Binary operation
            xs, ys, err = make_error_grid(op, ideal_op, f=f, x_range=x_range,
                                          y_range=y_range, steps=steps, device=device)
            results[f] = {
                'max_error': float(torch.max(err)),
                'mean_error': float(torch.mean(err)),
                'median_error': float(torch.median(err)),
                'std_error': float(torch.std(err)),
                'error_tensor': err,
                'xs': xs,
                'ys': ys
            }

    return results

def plot_precision_comparison(
        results: Dict[int, Dict],
        *,
        ax=None,
        metric: str = 'max_error',
        log_scale: bool = True
    ):
    """
    Plot how error metrics change with precision level.

    Parameters
    ----------
    results : Dict[int, Dict]
        Results from precision_sweep_analysis.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on.
    metric : str
        Which error metric to plot ('max_error', 'mean_error', etc.)
    log_scale : bool
        Whether to use log scale for y-axis.

    Returns
    -------
    matplotlib.axes.Axes
        The axes containing the plot.

    Raises
    ------
    ImportError
        If `matplotlib` is not installed, an ImportError is raised.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib required for plotting")

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))

    precisions = sorted(results.keys())
    errors = [results[f][metric] for f in precisions]

    ax.plot(precisions, errors, 'o-', linewidth=2, markersize=6)
    ax.set_xlabel('Precision (f parameter)')
    ax.set_ylabel(f'{metric.replace("_", " ").title()}')
    ax.grid(True, alpha=0.3)

    if log_scale:
        ax.set_yscale('log')

    ax.set_title(f'Error vs Precision: {metric.replace("_", " ").title()}')

    return ax

def plot_precision_heatmap_grid(
        results: Dict[int, Dict],
        *,
        figsize: Tuple[int, int] = None,
        cmap: str = 'viridis'
    ):
    """
    Create a grid of heatmaps showing error patterns at different precisions.

    Parameters
    ----------
    results : Dict[int, Dict]
        Results from precision_sweep_analysis.
    figsize : Tuple[int, int], optional
        Size of the figure to create.
    cmap : str, optional
        Colormap to use for heatmaps. Default is 'viridis'.

    Returns
    -------
    Tuple[matplotlib.figure.Figure, List[matplotlib.axes.Axes]]
        The figure and list of axes containing the heatmaps.

    Raises
    ------
    ImportError
        If `matplotlib` is not installed, an ImportError is raised.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib required for plotting")

    precisions = sorted(results.keys())
    n_precs = len(precisions)

    # Determine grid layout
    cols = min(4, n_precs)
    rows = (n_precs + cols - 1) // cols

    if figsize is None:
        figsize = (4 * cols, 3 * rows)

    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    if n_precs == 1:
        axes = [axes]
    elif rows == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for i, f in enumerate(precisions):
        ax = axes[i]
        result = results[f]
        err = result['error_tensor']

        if 'ys' in result:
            # Binary operation - 2D heatmap
            xs, ys = result['xs'], result['ys']
            xs_np = xs.detach().cpu().numpy()
            ys_np = ys.detach().cpu().numpy()

            im = ax.imshow(
                err.detach().cpu().numpy().T,
                extent=[xs_np.min(), xs_np.max(), ys_np.min(), ys_np.max()],
                origin="lower",
                aspect="auto",
                cmap=cmap
            )
            ax.set_xlabel('x')
            ax.set_ylabel('y')
        else:
            # Unary operation - 1D stripe
            xs = result['xs']
            xs_np = xs.detach().cpu().numpy()
            err_np = err.detach().cpu().numpy()

            im = ax.imshow(
                err_np[None, :],
                extent=[xs_np.min(), xs_np.max(), 0, 1],
                origin="lower",
                aspect="auto",
                cmap=cmap
            )
            ax.set_xlabel('x')
            ax.set_yticks([])

        ax.set_title(f'f={f} (max_err={result["max_error"]:.2e})')
        plt.colorbar(im, ax=ax)

    # Hide unused subplots
    for i in range(n_precs, len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()
    return fig, axes[:n_precs]
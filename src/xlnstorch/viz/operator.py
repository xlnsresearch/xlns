from __future__ import annotations
import math
from decimal import Decimal, getcontext
from typing import Callable, Tuple, List
import torch
from .. import LNSTensor, lnstensor

def _to_dec(x: float | torch.Tensor | LNSTensor) -> Decimal:
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
        ideal_op: Callable | None = None,
        *,
        f: float | None = None,
        b: float | None = None,
        x_range: Tuple[float, float] = (-1.0, 1.0),
        y_range: Tuple[float, float] | None = None,
        steps: int = 201,
        device: torch.device | str = "cpu",
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
        If `matplotlib` or `numpy` is not installed, an ImportError is raised.
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        raise ImportError("matplotlib and numpy are required for plotting error heatmaps")

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
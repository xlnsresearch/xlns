from typing import List, Optional
import warnings
from pathlib import Path
from torch.utils.cpp_extension import load, include_paths
import xlnstorch
import torch

try:
    import xlnstorch._csrc as _C
except Exception:
    _C = None

def _ensure_built() -> None:
    if _C is None:
        raise ImportError(
            "xlnstorch C++ extension is not built. "
            "Install a prebuilt wheel or build from source with a C++17 compiler."
        )

def load_backend(
        build_dir: Optional[str] = None,
        verbose: bool = False,
        enable_cpp: bool = True,
) -> bool:
    """
    Try to compile and load the C++ extension if it is not
    already built and loaded. If the extension is already
    loaded, this function exits and returns True.

    Parameters
    ----------
    build_dir : Optional[str], optional
        Directory to use for building the extension. If None, a temporary directory is used.
    verbose : bool, optional
        If True, enables verbose output during the build process.
    enable_cpp : bool, optional
        If True, sets the C++ operators as the default implementations for LNS operations.

    Returns
    -------
    bool
        True if the C++ extension is successfully loaded, False otherwise.
    """
    global _C
    if _C is not None:
        return True

    src_dir = Path(__file__).resolve().parent
    cpp_files = [str(p) for p in src_dir.glob("*.cpp")]

    try:
        mod = load(
            name="xlnstorch_csrc",
            sources=cpp_files,
            build_directory=build_dir,
            extra_cflags=["-O3", "-std=c++17", "-ffast-math"],
            verbose=verbose,
            extra_include_paths=include_paths(),
        )

    except (RuntimeError, OSError) as e:
        warnings.warn(f"Could not build C++ backend: {e}")
        return False

    _C = mod
    xlnstorch.CSRC_AVAILABLE = True

    if enable_cpp:
        xlnstorch.operators.toggle_cpp_implementations(True)

    return True

def float_to_lns_forward(x: torch.Tensor, base: torch.Tensor) -> torch.Tensor:
    """
    Convert a tensor from floating-point representation to LNS representation.

    Parameters
    ----------
    x : torch.Tensor
        The input tensor in floating-point format (float64).
    base : torch.Tensor
        The base for the LNS representation.

    Returns
    -------
    torch.Tensor
        The output tensor in LNS format (torch.int64).
    """
    _ensure_built()
    return _C.float_to_lns_forward(x, base)

def float_to_lns_backward(grad: torch.Tensor, base: torch.Tensor) -> torch.Tensor:
    """
    Convert the gradient from LNS representation back to floating-point representation.

    Parameters
    ----------
    grad : torch.Tensor
        The gradient tensor in LNS format (torch.int64).
    base : torch.Tensor
        The base for the LNS representation.

    Returns
    -------
    torch.Tensor
        The output gradient tensor in floating-point format (float64).
    """
    _ensure_built()
    return _C.float_to_lns_backward(grad, base)

def change_base_forward(x: torch.Tensor, old_base: torch.Tensor, new_base: torch.Tensor) -> torch.Tensor:
    """
    Change the base of an LNS tensor.

    Parameters
    ----------
    x : torch.Tensor
        The input tensor in LNS format (torch.int64).
    old_base : torch.Tensor
        The current base of the LNS tensor.
    new_base : torch.Tensor
        The new base to convert the LNS tensor to.
    """
    _ensure_built()
    return _C.change_base_forward(x, old_base, new_base)

def change_base_backward(grad: torch.Tensor, old_base: torch.Tensor, new_base: torch.Tensor) -> torch.Tensor:
    """
    Change the base of the gradient tensor from LNS representation.

    Parameters
    ----------
    grad : torch.Tensor
        The gradient tensor in LNS format (torch.int64).
    old_base : torch.Tensor
        The original base of the LNS tensor (input to forward).
    new_base : torch.Tensor
        The base of the gradient tensor (same baseas output to forward).

    Returns
    -------
    torch.Tensor
        The output gradient tensor in LNS format (torch.int64) with the old base.
    """
    _ensure_built()
    return _C.change_base_backward(grad, old_base, new_base)

def set_default_sbdb_implementation(impl_key: str) -> None:
    """
    Sets the default implementation for the SBDB function used in LNS addition.

    Parameters
    ----------
    impl_key : str
        The key for the implementation to set as default. This should match one of the
        keys defined in the xlnstorch.operators.implementations module.
    """
    _ensure_built()
    return _C.set_default_sbdb_implementation(impl_key)

def get_table(ez: torch.Tensor, sbdb: torch.Tensor, base: torch.Tensor) -> None:
    """
    Gives access to the underlying data pointer of the sbdb Tensor and saves
    copies of the ez and base tensors. This allows us to perform lookup table
    addition without doubling the memory usage.

    Parameters
    ----------
    ez : torch.Tensor
        A tensor used for the lookup table.
    sbdb : torch.Tensor
        The SBDB tensor containing the lookup table data.
    base : torch.Tensor
        The base for the LNS representation.
    """
    _ensure_built()
    return _C.get_table(ez, sbdb, base)

def add_forward(x: torch.Tensor, y: torch.Tensor, base: torch.Tensor) -> torch.Tensor:
    """
    Perform LNS addition on two tensors internal representations.

    Parameters
    ----------
    x : torch.Tensor
        The first tensor in LNS format (torch.int64).
    y : torch.Tensor
        The second tensor in LNS format (torch.int64).
    base : torch.Tensor
        The base for the LNS representation.
    """
    _ensure_built()
    return _C.add_forward(x, y, base)

def sum_forward(x: torch.Tensor, base: torch.Tensor, dim: List[int], keepdim: bool) -> torch.Tensor:
    """
    Perform LNS summation over specified dimensions.

    Parameters
    ----------
    x : torch.Tensor
        The input tensor in LNS format (torch.int64).
    base : torch.Tensor
        The base for the LNS representation.
    dim : List[int]
        The dimensions over which to sum.
    keepdim : bool
        Whether to keep the dimensions of the output tensor the same as the input tensor.

    Returns
    -------
    torch.Tensor
        The output tensor in LNS format (torch.int64) after summation.
    """
    _ensure_built()
    return _C.sum_forward(x, base, dim, keepdim)

def matmul_forward(A: torch.Tensor, B: torch.Tensor, base: torch.Tensor) -> torch.Tensor:
    """
    Perform LNS matrix multiplication.

    Parameters
    ----------
    A : torch.Tensor
        The first tensor in LNS format (torch.int64).
    B : torch.Tensor
        The second tensor in LNS format (torch.int64).
    base : torch.Tensor
        The base for the LNS representation.

    Returns
    -------
    torch.Tensor
    """
    _ensure_built()
    return _C.matmul_forward(A, B, base)

def matmul_backward(grad: torch.Tensor, A: torch.Tensor, B: torch.Tensor, base: torch.Tensor) -> torch.Tensor:
    """
    Perform the backward pass for LNS matrix multiplication.

    Parameters
    ----------
    grad : torch.Tensor
        The gradient tensor in LNS format (torch.int64).
    A : torch.Tensor
        The first tensor in LNS format (torch.int64).
    B : torch.Tensor
        The second tensor in LNS format (torch.int64).
    base : torch.Tensor
        The base for the LNS representation.

    Returns
    -------
    torch.Tensor
        The output gradient tensor in LNS format (torch.int64).
    """
    _ensure_built()
    return _C.matmul_backward(grad, A, B, base)

def conv1d_forward(
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        base: torch.Tensor,
        stride: int,
        padding: int,
        dilation: int,
        groups: int,
) -> torch.Tensor:
    """
    Perform LNS 1D convolution.

    Parameters
    ----------
    x : torch.Tensor
        The input tensor in LNS format (torch.int64).
    weight : torch.Tensor
        The convolution kernel in LNS format (torch.int64).
    bias : torch.Tensor
        The bias tensor in LNS format (torch.int64).
    base : torch.Tensor
        The base for the LNS representation.
    stride : int
        The stride of the convolution.
    padding : int
        The padding applied to the input tensor.
    dilation : int
        The dilation factor for the convolution.
    groups : int
        The number of groups for the convolution.

    Returns
    -------
    torch.Tensor
        The output tensor in LNS format (torch.int64) after convolution.
    """
    _ensure_built()
    return _C.conv1d_forward(x, weight, bias, base, stride, padding, dilation, groups)

def conv1d_backward(
        grad: torch.Tensor,
        x: torch.Tensor,
        weight: torch.Tensor,
        base: torch.Tensor,
        bias_defined: bool,
        stride: int,
        padding: int,
        dilation: int,
        groups: int,
) -> torch.Tensor:
    """
    Perform the backward pass for LNS 1D convolution.

    Parameters
    ----------
    grad : torch.Tensor
        The gradient tensor in LNS format (torch.int64).
    x : torch.Tensor
        The input tensor in LNS format (torch.int64).
    weight : torch.Tensor
        The convolution kernel in LNS format (torch.int64).
    base : torch.Tensor
        The base for the LNS representation.
    bias_defined : bool
        Whether the bias tensor is defined.
    stride : int
        The stride of the convolution.
    padding : int
        The padding applied to the input tensor.
    dilation : int
        The dilation factor for the convolution.
    groups : int
        The number of groups for the convolution.

    Returns
    -------
    torch.Tensor
        The output gradient tensor in LNS format (torch.int64).
    """
    _ensure_built()
    return _C.conv1d_backward(grad, x, weight, base, bias_defined, stride, padding, dilation, groups)

def conv2d_forward(
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        base: torch.Tensor,
        stride_h: int,
        stride_w: int,
        padding_h: int,
        padding_w: int,
        dilation_h: int,
        dilation_w: int,
        groups: int,
) -> torch.Tensor:
    """
    Perform LNS 2D convolution.

    Parameters
    ----------
    x : torch.Tensor
        The input tensor in LNS format (torch.int64).
    weight : torch.Tensor
        The convolution kernel in LNS format (torch.int64).
    bias : torch.Tensor
        The bias tensor in LNS format (torch.int64).
    base : torch.Tensor
        The base for the LNS representation.
    stride_h : int
        The vertical stride of the convolution.
    stride_w : int
        The horizontal stride of the convolution.
    padding_h : int
        The vertical padding applied to the input tensor.
    padding_w : int
        The horizontal padding applied to the input tensor.
    dilation_h : int
        The vertical dilation factor for the convolution.
    dilation_w : int
        The horizontal dilation factor for the convolution.
    groups : int
        The number of groups for the convolution.

    Returns
    -------
    torch.Tensor
        The output tensor in LNS format (torch.int64) after convolution.
    """
    _ensure_built()
    return _C.conv2d_forward(x, weight, bias, base,
                             stride_h, stride_w, padding_h, padding_w,
                             dilation_h, dilation_w, groups)

def conv2d_backward(
        grad: torch.Tensor,
        x: torch.Tensor,
        weight: torch.Tensor,
        base: torch.Tensor,
        bias_defined: bool,
        stride_h: int,
        stride_w: int,
        padding_h: int,
        padding_w: int,
        dilation_h: int,
        dilation_w: int,
        groups: int,
) -> torch.Tensor:
    """
    Perform the backward pass for LNS 2D convolution.

    Parameters
    ----------
    grad : torch.Tensor
        The gradient tensor in LNS format (torch.int64).
    x : torch.Tensor
        The input tensor in LNS format (torch.int64).
    weight : torch.Tensor
        The convolution kernel in LNS format (torch.int64).
    base : torch.Tensor
        The base for the LNS representation.
    bias_defined : bool
        Whether the bias tensor is defined.
    stride_h : int
        The vertical stride of the convolution.
    stride_w : int
        The horizontal stride of the convolution.
    padding_h : int
        The vertical padding applied to the input tensor.
    padding_w : int
        The horizontal padding applied to the input tensor.
    dilation_h : int
        The vertical dilation factor for the convolution.
    dilation_w : int
        The horizontal dilation factor for the convolution.
    groups : int
        The number of groups for the convolution.

    Returns
    -------
    torch.Tensor
        The output gradient tensor in LNS format (torch.int64).
    """
    _ensure_built()
    return _C.conv2d_backward(grad, x, weight, base, bias_defined,
                              stride_h, stride_w, padding_h, padding_w,
                              dilation_h, dilation_w, groups)

def conv3d_forward(
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        base: torch.Tensor,
        stride_d: int,
        stride_h: int,
        stride_w: int,
        padding_d: int,
        padding_h: int,
        padding_w: int,
        dilation_d: int,
        dilation_h: int,
        dilation_w: int,
        groups: int,
) -> torch.Tensor:
    """
    Perform LNS 3D convolution.

    Parameters
    ----------
    x : torch.Tensor
        The input tensor in LNS format (torch.int64).
    weight : torch.Tensor
        The convolution kernel in LNS format (torch.int64).
    bias : torch.Tensor
        The bias tensor in LNS format (torch.int64).
    base : torch.Tensor
        The base for the LNS representation.
    stride_d : int
        The depth stride of the convolution.
    stride_h : int
        The height stride of the convolution.
    stride_w : int
        The width stride of the convolution.
    padding_d : int
        The depth padding applied to the input tensor.
    padding_h : int
        The height padding applied to the input tensor.
    padding_w : int
        The width padding applied to the input tensor.
    dilation_d : int
        The depth dilation factor for the convolution.
    dilation_h : int
        The height dilation factor for the convolution.
    dilation_w : int
        The width dilation factor for the convolution.
    groups : int
        The number of groups for the convolution.

    Returns
    -------
    torch.Tensor
        The output tensor in LNS format (torch.int64) after convolution.
    """
    _ensure_built()
    return _C.conv3d_forward(x, weight, bias, base,
                             stride_d, stride_h, stride_w,
                             padding_d, padding_h, padding_w,
                             dilation_d, dilation_h, dilation_w, groups)

def conv3d_backward(
        grad: torch.Tensor,
        x: torch.Tensor,
        weight: torch.Tensor,
        base: torch.Tensor,
        bias_defined: bool,
        stride_d: int,
        stride_h: int,
        stride_w: int,
        padding_d: int,
        padding_h: int,
        padding_w: int,
        dilation_d: int,
        dilation_h: int,
        dilation_w: int,
        groups: int,
) -> torch.Tensor:
    """
    Perform the backward pass for LNS 3D convolution.

    Parameters
    ----------
    grad : torch.Tensor
        The gradient tensor in LNS format (torch.int64).
    x : torch.Tensor
        The input tensor in LNS format (torch.int64).
    weight : torch.Tensor
        The convolution kernel in LNS format (torch.int64).
    base : torch.Tensor
        The base for the LNS representation.
    bias_defined : bool
        Whether the bias tensor is defined.
    stride_d : int
        The depth stride of the convolution.
    stride_h : int
        The height stride of the convolution.
    stride_w : int
        The width stride of the convolution.
    padding_d : int
        The depth padding applied to the input tensor.
    padding_h : int
        The height padding applied to the input tensor.
    padding_w : int
        The width padding applied to the input tensor.
    dilation_d : int
        The depth dilation factor for the convolution.
    dilation_h : int
        The height dilation factor for the convolution.
    dilation_w : int
        The width dilation factor for the convolution.
    groups : int
        The number of groups for the convolution.
    """
    _ensure_built()
    return _C.conv3d_backward(grad, x, weight, base, bias_defined,
                              stride_d, stride_h, stride_w,
                              padding_d, padding_h, padding_w,
                              dilation_d, dilation_h, dilation_w, groups)

def avg_pool1d_forward(
        x: torch.Tensor,
        kernel_size: int,
        base: torch.Tensor,
        stride: Optional[int],
        padding: int,
        ceil_mode: bool,
        count_include_pad: bool,
) -> torch.Tensor:
    """
    Perform LNS 1D average pooling.

    Parameters
    ----------
    x : torch.Tensor
        The input tensor in LNS format (torch.int64).
    kernel_size : int
        The size of the pooling kernel.
    base : torch.Tensor
        The base for the LNS representation.
    stride : int, optional
        The stride of the pooling operation. Defaults to kernel_size if None.
    padding : int
        The padding applied to the input tensor.
    ceil_mode : bool
        If True, will use ceil instead of floor to compute the output shape.
    count_include_pad : bool
        If True, will include the zero-padding in the averaging calculation.

    Returns
    -------
    torch.Tensor
        The output tensor in LNS format (torch.int64) after average pooling.
    """
    _ensure_built()
    return _C.avg_pool1d_forward(x, kernel_size, base, stride, padding, ceil_mode, count_include_pad)

def avg_pool1d_backward(
        grad: torch.Tensor,
        x: torch.Tensor,
        kernel_size: int,
        base: torch.Tensor,
        stride: Optional[int],
        padding: int,
        ceil_mode: bool,
        count_include_pad: bool,
) -> torch.Tensor:
    """
    Perform the backward pass for LNS 1D average pooling.

    Parameters
    ----------
    grad : torch.Tensor
        The gradient tensor in LNS format (torch.int64).
    x : torch.Tensor
        The input tensor in LNS format (torch.int64).
    kernel_size : int
        The size of the pooling kernel.
    base : torch.Tensor
        The base for the LNS representation.
    stride : int, optional
        The stride of the pooling operation. Defaults to kernel_size if None.
    padding : int
        The padding applied to the input tensor.
    ceil_mode : bool
        If True, will use ceil instead of floor to compute the output shape.
    count_include_pad : bool
        If True, will include the zero-padding in the averaging calculation.

    Returns
    -------
    torch.Tensor
        The output gradient tensor in LNS format (torch.int64).
    """
    _ensure_built()
    return _C.avg_pool1d_backward(grad, x, kernel_size, base, stride, padding, ceil_mode, count_include_pad)

__all__ = [
    "load_backend",

    "float_to_lns_forward",
    "float_to_lns_backward",
    "change_base_forward",
    "change_base_backward",
    "set_default_sbdb_implementation",
    "get_table",
    "add_forward",
    "sum_forward",
    "matmul_forward",
    "matmul_backward",
    "conv1d_forward",
    "conv1d_backward",
    "conv2d_forward",
    "conv2d_backward",
    "conv3d_forward",
    "conv3d_backward",
    "avg_pool1d_forward",
    "avg_pool1d_backward",
]
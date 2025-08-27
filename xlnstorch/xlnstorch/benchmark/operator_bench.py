from typing import Callable, Tuple, Dict, Union, Optional
import torch
from xlnstorch import randn, randn_like
from . import Benchmark

class UnaryBench(Benchmark):
    """
    A benchmark for unary operations in xlnstorch or torch.
    Inputs are generated from the standard normal distribution.

    Arguments
    ---------
    func : Callable
        The unary function to benchmark, e.g., `torch.sign` or `torch.relu`.
    shape : Tuple
        The shape of the input tensor.
    lns : bool, optional
        If True, uses `xlnstorch.randn` for generating inputs, otherwise uses `torch.randn`.
    f : int, optional
        The precision parameter for the input LNSTensor.
    b : float, optional
        The base parameter for the input LNSTensor.
    backward : bool, optional
        If True, the input tensor will require gradients for backward pass.
    device : torch.device or str, optional
        The device on which to create the input tensor (default is "cpu").
    kwargs : Dict, optional
        Additional keyword arguments to pass to the function being benchmarked.
    """

    def __init__(
            self,
            func: Callable,
            shape: Tuple,
            lns: bool = True,
            f: Optional[int] = None,
            b: Optional[float] = None,
            backward: bool = False,
            device: Union[torch.device, str] = "cpu",
            kwargs: Optional[Dict] = None
        ):
        self.func = func
        self.shape = shape
        self.lns = lns
        self.f = f
        self.b = b
        self.backward = backward
        self.device = device
        self.kwargs = kwargs if kwargs is not None else {}

    def make_inputs(self):
        if self.lns:
            a = randn(*self.shape, f=self.f, b=self.b, device=self.device, requires_grad=self.backward)
        else:
            a = torch.randn(*self.shape, device=self.device, requires_grad=self.backward)
        return (a,)

    def forward(self, x):
        return self.func(x, **self.kwargs)

class BinaryBench(Benchmark):
    """
    A benchmark for binary operations in xlnstorch or torch.
    Inputs are generated from the standard normal distribution.

    Arguments
    ---------
    func : Callable
        The unary function to benchmark, e.g., `torch.sign` or `torch.relu`.
    shape : Tuple
        The shape of the input tensor.
    lns : bool, optional
        If True, uses `xlnstorch.randn` for generating inputs, otherwise uses `torch.randn`.
    f : int, optional
        The precision parameter for the input LNSTensor.
    b : float, optional
        The base parameter for the input LNSTensor.
    backward : bool, optional
        If True, the input tensor will require gradients for backward pass.
    device : torch.device or str, optional
        The device on which to create the input tensor (default is "cpu").
    kwargs : Dict, optional
        Additional keyword arguments to pass to the function being benchmarked.
    """

    def __init__(
            self,
            func: Callable,
            shape: Tuple,
            lns: bool = True,
            f: Optional[int] = None,
            b: Optional[float] = None,
            backward: bool = False,
            device: Union[torch.device, str] = "cpu",
            kwargs: Optional[Dict] = None
        ):
        self.func = func
        self.shape = shape
        self.lns = lns
        self.f = f
        self.b = b
        self.backward = backward
        self.device = device
        self.kwargs = kwargs if kwargs is not None else {}

    def make_inputs(self):
        if self.lns:
            a = randn(*self.shape, f=self.f, b=self.b, device=self.device, requires_grad=self.backward)
            b = randn_like(a, requires_grad=self.backward)
        else:
            a = torch.randn(*self.shape, device=self.device, requires_grad=self.backward)
            b = torch.randn_like(a, requires_grad=self.backward)
        return a, b

    def forward(self, x, y):
        return self.func(x, y, **self.kwargs)
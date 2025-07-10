from typing import Callable, Tuple, Dict
import torch
from .. import randn, randn_like
from . import Benchmark

class UnaryBench(Benchmark):
    """
    A benchmark for unary operations in xlnstorch.
    """

    def __init__(
            self,
            func: Callable,
            shape: Tuple,
            backward: bool = False,
            device: torch.device | str = "cpu",
            kwargs: Dict | None = None
        ):
        self.func = func
        self.shape = shape
        self.backward = backward
        self.device = device
        self.kwargs = kwargs if kwargs is not None else {}

    def make_inputs(self):
        a = randn(*self.shape, device=self.device, requires_grad=self.backward)
        return (a,)

    def forward(self, x):
        return self.func(x, **self.kwargs)

class BinaryBench(Benchmark):
    """
    A benchmark for binary operations in xlnstorch.
    """

    def __init__(
            self,
            func: Callable,
            shape: Tuple,
            backward: bool = False,
            device: torch.device | str = "cpu",
            kwargs: Dict | None = None
        ):
        self.func = func
        self.shape = shape
        self.backward = backward
        self.device = device
        self.kwargs = kwargs if kwargs is not None else {}

    def make_inputs(self):
        a = randn(*self.shape, device=self.device, requires_grad=self.backward)
        b = randn_like(a, requires_grad=self.backward)
        return a, b

    def forward(self, x, y):
        return self.func(x, y, **self.kwargs)
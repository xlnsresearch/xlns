import torch
from .. import LNSTensor, lnstensor
from ..operators import (
    lns_sub,
    lns_mul,
)

class LNSSGD(torch.optim.Optimizer):

    def __init__(self, params, lr=0.1):

        if isinstance(lr, float):
            lr = lnstensor(lr)

        if not isinstance(lr, LNSTensor):
            raise TypeError("Learning rate should be an instance of LNSTensor")

        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")

        defaults = dict(lr=lr)
        super(LNSSGD, self).__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step."""

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:

            lr = group["lr"]
            base = group["base"]
            for p in group["params"]:

                if p.grad is None:
                    continue

                p.data = lns_sub(p, lns_mul(p.grad, lr._lns), base)

        return loss

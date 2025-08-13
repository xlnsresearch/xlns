import torch
from xlnstorch import LNSTensor, lnstensor, LNS_ZERO, LNS_ONE, align_lnstensor_bases
from xlnstorch.operators import (
    lns_mul,
    lns_sign,
    lns_eq,
    lns_lt,
    lns_abs,
    lns_div,
    lns_add,
    lns_reciprocal,
    lns_maximum,
)
from . import LNSOptimizer

def _as_lnstensor(x):
    if isinstance(x, LNSTensor):
        return x
    else:
        return lnstensor(x)

class LNSHybridMul(LNSOptimizer):
    r"""
    Implements a hybrid multiplication algorithm for LNSTensor
    parameters. This optimizer uses a heuristic to decide between
    using a standard multiplicative update, a sign-based update
    or a gradient descent-like update.
    """

    def __init__(
            self,
            params,
            lr=0.01,
    ):

        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")

        defaults = dict(
            lr=_as_lnstensor(lr),
        )
        super(LNSHybridMul, self).__init__(params, defaults)

        for group in self.param_groups:
            lr_ = group["lr"]
            group["signmul_term"] = 2.0 ** lr_

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step."""

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            signmul_term = group["signmul_term"]
            base = group["base"]

            # Align the parameters to the base of the group.
            lr, signmul_term = align_lnstensor_bases(lr, signmul_term, base=base)

            for p in group["params"]:

                if p.grad is None:
                    continue

                grad = p.grad

                same_sign = lns_eq(lns_sign(grad, base), lns_sign(p, base))
                small_values = lns_lt(lns_abs(grad), lr._lns) | lns_lt(lns_abs(p.data), lr._lns)
                mul_mask = same_sign | small_values

                lr_mul_grad = lns_mul(lr._lns, lns_abs(grad))
                mul_update = lns_add(LNS_ONE, lr_mul_grad, base)
                gd_update = lns_add(LNS_ONE, lns_div(lr_mul_grad, lns_abs(p), base), base)
                mul_term = torch.where(mul_mask, lns_reciprocal(mul_update, base),
                                       lns_maximum(signmul_term._lns, gd_update, base))

            p.data = lns_mul(p.data, mul_term)
import torch
from xlnstorch import LNSTensor, lnstensor, LNS_ZERO, LNS_ONE, align_lnstensor_bases
from xlnstorch.operators import (
    lns_mul,
    lns_sign,
    lns_eq,
)
from . import LNSOptimizer

def _as_lnstensor(x):
    if isinstance(x, LNSTensor):
        return x
    else:
        return lnstensor(x)

class LNSSignMul(LNSOptimizer):
    """
    """

    def __init__(
            self,
            params,
            lr=0.01,
            use_pow=False,
            *,
            maximize=False
        ):

        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")

        defaults = dict(
            lr=_as_lnstensor(lr),
            use_pow=use_pow,
            maximize=maximize
        )
        super(LNSSignMul, self).__init__(params, defaults)

        # precompute 1 + lr and 1 / (1 + lr)
        for group in self.param_groups:
            lr_ = group["lr"]
            use_pow_ = group["use_pow"]

            if use_pow_:
                group["mul_term"] = 2.0 ** lr_
            else:
                group["mul_term"] = 1.0 + lr_

            group["inv_mul_term"] = 1.0 / group["mul_term"]

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step."""

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            mul_term = group["mul_term"]
            inv_mul_term = group["inv_mul_term"]
            maximize = group["maximize"]
            base = group["base"]

            # Align the parameters to the base of the group.
            lr, mul_term, inv_mul_term = align_lnstensor_bases(lr, mul_term, inv_mul_term, base=base)

            for p in group["params"]:

                if p.grad is None:
                    continue

                grad = p.grad

                same_sign = lns_eq(lns_sign(grad, base), lns_sign(p, base))
                mul_update = torch.where(same_sign ^ maximize, inv_mul_term._lns, mul_term._lns)

                p.data = lns_mul(p.data, mul_update)
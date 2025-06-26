import torch
from .. import LNSTensor, lnstensor, LNS_ZERO, align_lnstensor_bases, zeros_like
from ..operators import (
    lns_sub,
    lns_equal,
    lns_neg,
    lns_mul,
    lns_add,
    lns_div,
    lns_sqrt,
    lns_pow,
    lns_maximum,
    lns_gt,
)

def _as_lnstensor(x):
    if isinstance(x, LNSTensor):
        return x
    else:
        return lnstensor(x)

class LNSASGD(torch.optim.Optimizer):
    """
    """

    def __init__(
            self,
            params,
            lr=0.01,
            lambd=0.0001,
            alpha=0.75,
            t0=1000000.0,
            weight_decay=0.0,
            *,
            maximize=False
        ):

        if lr <= 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")

        if lambd < 0.0:
            raise ValueError(f"Invalid lambda value: {lambd}")

        if not (0.0 < alpha <= 1.0):
            raise ValueError(f"Invalid alpha value: {alpha}")

        if t0 < 0.0:
            raise ValueError(f"Invalid t0 value: {t0}")

        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        
        defaults = dict(
            lr=_as_lnstensor(lr),
            lambd=_as_lnstensor(lambd),
            alpha=_as_lnstensor(alpha),
            t0=_as_lnstensor(t0),
            weight_decay=_as_lnstensor(weight_decay),
            maximize=maximize,
        )
        super(LNSASGD, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            lambd = group["lambd"]
            alpha = group["alpha"]
            t0 = group["t0"]
            weight_decay = group["weight_decay"]
            maximize = group["maximize"]
            base = group["base"]

            # Align the parameters to the base of the group.
            lr, lambd, alpha, t0, weight_decay = align_lnstensor_bases(
                lr, lambd, alpha, t0, weight_decay, base=base)

            one = LNSTensor.get_internal_tensor(1.0, base)

            for p in group["params"]:

                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                if maximize:
                    grad = lns_neg(grad)

                if not lns_equal(weight_decay._lns, LNS_ZERO):
                    grad = lns_add(grad, lns_mul(p.data, weight_decay._lns), base)

                if len(state) == 0:
                    # First time we see this parameter
                    state["step"] = LNS_ZERO.clone()
                    state["averaging_coef"] = one.clone()
                    state["averaged_param"] = p.data.clone()

                # Retrieve running stats
                step = lns_add(state["step"], one, base)
                averaging_coef = state["averaging_coef"]
                averaged_param = state["averaged_param"]

                # 1. learning-rate schedule
                denom = lns_add(one, lns_mul(lambd._lns, lns_mul(lr._lns, step)), base)
                denom = lns_pow(denom, alpha.value, base)
                current_lr = lns_div(lr._lns, denom, base)

                # 2. update averaged parameter
                decay = lns_sub(one, lns_mul(lambd._lns, current_lr), base)
                p.data = lns_mul(p.data, decay)
                p.data = lns_sub(p.data, lns_mul(grad, current_lr), base)

                # 3. update averaged parameter
                if lns_gt(step, t0._lns):
                    denom = lns_maximum(one, lns_sub(step, t0._lns, base), base)
                    averaging_coef = lns_div(one, denom, base)

                diff = lns_sub(p.data, averaged_param, base)
                averaged_param = lns_add(averaged_param, lns_mul(diff, averaging_coef), base)

                state["step"] = step
                state["averaging_coef"] = averaging_coef
                state["averaged_param"] = averaged_param
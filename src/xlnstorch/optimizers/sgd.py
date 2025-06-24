import torch
from .. import LNSTensor, lnstensor, LNS_ZERO
from ..operators import (
    lns_equal,
    lns_sub,
    lns_mul,
    lns_add,
)

def _as_internal_tensor(x):
    if isinstance(x, LNSTensor):
        return x._lns
    else:
        return lnstensor(x)._lns

class LNSSGD(torch.optim.Optimizer):
    """
    Implements stochastic gradient descent (SGD) with support for momentum,
    dampening, weight decay, and nesterov momentum.

    This optimizer is analogous to PyTorch's :py:class:`torch.optim.SGD`, but
    is designed to work with LNSTensor objects.

    """

    def __init__(
            self,
            params,
            lr=0.1,
            momentum=0.0,
            dampening=0.0,
            weight_decay=0.0,
            nesterov=False,
            *,
            maximize=False
        ):

        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")

        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")

        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {dampening}")

        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")

        defaults = dict(
            lr=_as_internal_tensor(lr),
            momentum=_as_internal_tensor(momentum),
            dampening=_as_internal_tensor(dampening),
            weight_decay=_as_internal_tensor(weight_decay),
            nesterov=nesterov,
            maximize=maximize
        )
        super(LNSSGD, self).__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step."""

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            dampening = group["dampening"]
            weight_decay = group["weight_decay"]
            nesterov = group["nesterov"]
            maximize = group["maximize"]
            base = group["base"]

            for p in group["params"]:

                if p.grad is None:
                    continue

                grad = p.grad # g_t
                state = self.state[p]

                # 1. weight_decay: g ← g + λθ
                if not lns_equal(weight_decay, LNS_ZERO):
                    grad = lns_add(grad, lns_mul(p.data, weight_decay), base)

                # 2. momentum buffering: b ← μb + (1 - τ)g
                if not lns_equal(momentum, LNS_ZERO):
                    buf = state.get("momentum_buffer", None)

                    if buf is None:
                        buf = grad.clone()
                        state["momentum_buffer"] = buf

                    else:
                        one_minus_tau = lns_sub(LNSTensor.get_internal_tensor(1.0, base), dampening, base)
                        buf = lns_add(
                            lns_mul(buf, momentum),
                            lns_mul(grad, one_minus_tau),
                            base
                        )
                        state["momentum_buffer"] = buf

                    # 3a. nesterov momentum: g ← g + μb
                    if nesterov:
                        grad = lns_add(grad, lns_mul(buf, momentum), base)
                    # 3b. classical momentum: g ← b 
                    else:
                        grad = buf

                # 4. parameter update: θ ← θ ± γg
                delta = lns_mul(grad, lr)
                if maximize:
                    p.data = lns_add(p.data, delta, base)
                else:
                    p.data = lns_sub(p.data, delta, base)

        return loss

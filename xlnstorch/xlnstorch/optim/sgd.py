import torch
from xlnstorch import LNS_ZERO, LNS_ONE
from xlnstorch.operators import (
    lns_equal,
    lns_sub,
    lns_mul,
    lns_add,
    lns_neg,
)
from . import LNSOptimizer

class LNSSGD(LNSOptimizer):
    """
    Implements stochastic gradient descent (SGD) with support for momentum,
    dampening, weight decay, and nesterov momentum.

    This optimizer is analogous to PyTorch's :py:class:`torch.optim.SGD`, but
    is designed to work with LNSTensor objects. See the PyTorch documentation
    for more details on the SGD algorithm.

    Parameters
    -----------
    params : iterable
        An iterable of parameters to optimize or dicts defining parameter groups.
        This should be obtained from a model's `lns_parameters()` method.
    lr : LNSTensor, float, optional
        Learning rate (default: 0.001). Must be a non-negative LNSTensor or float.
    momentum : LNSTensor, float, optional
        Momentum factor (default: 0.0). Must be a non-negative LNSTensor or float.
    dampening : LNSTensor, float, optional
        Dampening for momentum (default: 0.0). Must be a non-negative LNSTensor or float.
    weight_decay : LNSTensor, float, optional
        Weight decay (L2 penalty) (default: 0.0). Must be a non-negative LNSTensor or float.
    nesterov : bool, optional
        Enables Nesterov momentum if set to True (default: False).
    maximize : bool, optional
        If True, optimizes the parameters for maximization instead of minimization (default: False).

    Examples
    --------
    >>> optimizer = xlnstorch.optim.LNSSGD(model.lns_parameters(), lr=0.1, momentum=0.9)
    >>> optimizer.zero_grad() # Clear gradients before the step
    >>> loss_fn(model(input), target).backward() # Compute gradients
    >>> optimizer.step() # Update parameters based on gradients
    """

    def __init__(
            self,
            params,
            lr=0.001,
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
            lr=lr,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
            nesterov=nesterov,
            maximize=maximize
        )
        super(LNSSGD, self).__init__(params, defaults)
        self.make_lnstensor_params("lr", "momentum", "dampening", "weight_decay")

    @torch.no_grad()
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

                if maximize:
                    grad = lns_neg(grad)

                # 1. weight_decay: g ← g + λθ
                if not lns_equal(weight_decay, LNS_ZERO):
                    grad = lns_add(grad, lns_mul(p.data, weight_decay, base), base)

                # 2. momentum buffering: b ← μb + (1 - τ)g
                if not lns_equal(momentum, LNS_ZERO):
                    buf = state.get("momentum_buffer", None)

                    if buf is None:
                        buf = grad.clone()
                        state["momentum_buffer"] = buf

                    else:
                        one_minus_tau = lns_sub(LNS_ONE, dampening, base)
                        buf = lns_add(
                            lns_mul(buf, momentum, base),
                            lns_mul(grad, one_minus_tau, base),
                            base
                        )
                        state["momentum_buffer"] = buf

                    # 3a. nesterov momentum: g ← g + μb
                    if nesterov:
                        grad = lns_add(grad, lns_mul(buf, momentum, base), base)
                    # 3b. classical momentum: g ← b 
                    else:
                        grad = buf

                # 4. parameter update: θ ← θ ± γg
                delta = lns_mul(grad, lr, base)
                p.data = lns_sub(p.data, delta, base)

        return loss
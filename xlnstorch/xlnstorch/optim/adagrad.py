import torch
from xlnstorch import LNS_ZERO, LNS_ONE
from xlnstorch.operators import (
    lns_equal,
    lns_sub,
    lns_mul,
    lns_add,
    lns_div,
    lns_sqrt,
    lns_neg,
)
from . import LNSOptimizer

class LNSAdagrad(LNSOptimizer):
    """
    Implements the Adagrad algorithm with support for learning rate decay,
    weight decay, and an initial accumulator value.

    This optimizer is analogous to PyTorch's :py:class:`torch.optim.Adagrad`,
    but is designed to work with LNSTensor objects. See the PyTorch documentation
    for more details on the Adagrad algorithm.

    Parameters
    -----------
    params : iterable
        An iterable of parameters to optimize or dicts defining parameter groups.
        This should be obtained from a model's `lns_parameters()` method.
    lr : LNSTensor, float, optional
        Learning rate (default: 0.01). Must be a non-negative LNSTensor or float.
    lr_decay : LNSTensor, float, optional
        Learning rate decay factor (default: 0.0). Must be a non-negative LNSTensor
        or float.
    weight_decay : LNSTensor, float, optional
        Weight decay (L2 penalty) (default: 0.0). Must be a non-negative LNSTensor
        or float.
    initial_accumulator_value : LNSTensor, float, optional
        Initial value for the accumulator (default: 0). Must be a non-negative
        LNSTensor or float.
    eps : LNSTensor, float, optional
        Term added to the denominator for numerical stability (default: 1e-10).
    maximize : bool, optional
        If True, optimizes the parameters for maximization instead of minimization
        (default: False).
    """

    def __init__(
            self,
            params,
            lr=0.01,
            lr_decay=0.0,
            weight_decay=0.0,
            initial_accumulator_value=0,
            eps=1e-10,
            *,
            maximize=False
        ):

        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")

        if lr_decay < 0.0:
            raise ValueError(f"Invalid learning rate decay: {lr_decay}")

        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        if initial_accumulator_value < 0.0:
            raise ValueError(f"Invalid initial accumulator value: {initial_accumulator_value}")

        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")

        defaults = dict(
            lr=lr,
            lr_decay=lr_decay,
            weight_decay=weight_decay,
            initial_accumulator_value=initial_accumulator_value,
            eps=eps,
            maximize=maximize
        )
        super(LNSAdagrad, self).__init__(params, defaults)
        self.make_lnstensor_params("lr", "lr_decay", "weight_decay", "initial_accumulator_value", "eps")

    @torch.no_grad()
    def step(self, closure=None):

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            lr_decay = group["lr_decay"]
            weight_decay = group["weight_decay"]
            init_acc_val = group["initial_accumulator_value"]
            eps = group["eps"]
            maximize = group["maximize"]
            base = group["base"]

            for p in group["params"]:

                if p.grad is None:
                    continue

                grad = p.grad # g_t
                state = self.state[p]

                if maximize:
                    grad = lns_neg(grad)

                # 1. State initialisation (run the first time we see this parameter)
                if len(state) == 0:
                    state["step"] = LNS_ZERO.clone()
                    state["sum"] = torch.full_like(p, init_acc_val)

                state["step"] = lns_add(state["step"], LNS_ONE, base)
                step = state["step"] # t

                # 2. step lr: γ' ← γ / (1 + (t − 1) * η)
                if not lns_equal(lr_decay, LNS_ZERO):
                    denom = lns_add(
                        LNS_ONE,
                        lns_mul(lr_decay, lns_sub(
                            step, LNS_ONE, base), base), base)
                    lr_t = lns_div(lr, denom, base)

                else:
                    lr_t = lr

                # 3. weight decay: g_t ← g_t + λ*θ_{t-1}
                if not lns_equal(weight_decay, LNS_ZERO):
                    grad = lns_add(grad, lns_mul(p.data, weight_decay, base), base)

                # 4. Accumulator update: s_t ← s_{t-1} + g_t^2
                s_prev = state["sum"]
                s_t = lns_add(s_prev, lns_mul(grad, grad, base), base)
                state["sum"] = s_t

                # 5. Parameter update: θ_t ← θ_{t-1} ± γ' * g_t / (sqrt(s_t) + ε)
                sqrt_s_t = lns_sqrt(s_t, base)
                denom = lns_add(sqrt_s_t, eps, base)
                delta = lns_div(lns_mul(grad, lr_t, base), denom, base)
                p.data = lns_sub(p.data, delta, base)

        return loss
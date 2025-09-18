import torch
from xlnstorch import LNS_ZERO, LNS_ONE
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

        self.validate_param("lr", lambda lr: lr >= 0.0)
        self.validate_param("lr_decay", lambda lr_decay: lr_decay >= 0.0)
        self.validate_param("weight_decay", lambda weight_decay: weight_decay >= 0.0)
        self.validate_param("initial_accumulator_value", lambda init_acc_val: init_acc_val >= 0.0)
        self.validate_param("eps", lambda eps: eps >= 0.0)

    @torch.no_grad()
    def step(self, closure=None):

        loss = None
        if closure is not None:
            loss = closure()

        for group, ops in self.lns_param_groups():
            lr = group["lr"]
            lr_decay = group["lr_decay"]
            weight_decay = group["weight_decay"]
            init_acc_val = group["initial_accumulator_value"]
            eps = group["eps"]
            maximize = group["maximize"]

            for p in group["params"]:

                if p.grad is None:
                    continue

                grad = p.grad.view(torch.int64) # g_t
                data = p.data.view(torch.int64)
                state = self.state[p]

                if maximize:
                    grad = ops.neg(grad)

                # 1. State initialisation (run the first time we see this parameter)
                if len(state) == 0:
                    state["step"] = LNS_ZERO.clone()
                    state["sum"] = torch.full_like(data, init_acc_val)

                state["step"] = ops.add(state["step"], LNS_ONE)
                step = state["step"] # t

                # 2. step lr: γ' ← γ / (1 + (t − 1) * η)
                if not ops.equal(lr_decay, LNS_ZERO):
                    denom = ops.add(
                        LNS_ONE,
                        ops.mul(lr_decay, ops.sub(
                            step, LNS_ONE)))
                    lr_t = ops.div(lr, denom)

                else:
                    lr_t = lr

                # 3. weight decay: g_t ← g_t + λ*θ_{t-1}
                if not ops.equal(weight_decay, LNS_ZERO):
                    grad = ops.add(grad, ops.mul(data, weight_decay))

                # 4. Accumulator update: s_t ← s_{t-1} + g_t^2
                s_prev = state["sum"]
                s_t = ops.add(s_prev, ops.mul(grad, grad))
                state["sum"] = s_t

                # 5. Parameter update: θ_t ← θ_{t-1} ± γ' * g_t / (sqrt(s_t) + ε)
                sqrt_s_t = ops.sqrt(s_t)
                denom = ops.add(sqrt_s_t, eps)
                delta = ops.div(ops.mul(grad, lr_t), denom)
                p.data = ops.sub(data, delta).view(torch.float64)

        return loss
import torch
from xlnstorch import LNS_ZERO, LNS_ONE
from . import LNSOptimizer

class LNSAdamax(LNSOptimizer):
    """
    Implements the Adamax optimization algorithm for LNSTensor parameters,
    including optional weight-decay regularisation, and a “maximize” mode.

    This optimizer is analogous to PyTorch's :py:class:`torch.optim.Adamax`,
    but is designed to work with LNSTensor objects. See the PyTorch
    documentation for more details on the Adamax algorithm.

    Parameters
    ----------
    params : iterable
        An iterable of parameters to optimize or dicts defining parameter groups.
        This should be obtained from a model's `lns_parameters()` method.
    lr : LNSTensor, float, optional
        Learning rate (default: 0.002). Must be a non-negative LNSTensor or float.
    betas : Tuple[float, float] or Tuple[LNSTensor, LNSTensor], optional
        Coefficients used for computing running averages of gradient and its square
        (default: (0.9, 0.999)). Must be two non-negative LNSTensor or float values
        in the range [0.0, 1.0).
    eps : LNSTensor, float, optional
        Term added to the denominator for numerical stability (default: 1e-8).
    weight_decay : LNSTensor or float
        Weight decay (L2 penalty) (default: 0.0). Must be a non-negative LNSTensor or float.
    maximize : bool, optional
        If True, optimizes the parameters for maximization instead of minimization (default: False).
    """
    def __init__(
            self,
            params,
            lr=0.002,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.0,
            *,
            maximize=False
        ):
        defaults = dict(
            lr=lr,
            beta1=betas[0],
            beta2=betas[1],
            eps=eps,
            weight_decay=weight_decay,
            maximize=maximize,
        )
        super().__init__(params, defaults)
        self.make_lnstensor_params("lr", "beta1", "beta2", "eps", "weight_decay")

        self.validate_param("lr", lambda lr: lr >= 0.0)
        self.validate_param("eps", lambda eps: eps > 0.0)
        self.validate_param("beta1",lambda beta1: 0.0 <= beta1 < 1.0)
        self.validate_param("beta2",lambda beta2: 0.0 <= beta2 < 1.0)
        self.validate_param("weight_decay", lambda weight_decay: weight_decay >= 0.0)

    @torch.no_grad()
    def step(self, closure=None):
    
        loss = None
        if closure is not None:
            loss = closure()

        for group, ops in self.lns_param_groups():
            lr = group["lr"]
            beta1 = group["beta1"]
            beta2 = group["beta2"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            maximize = group["maximize"]

            one_minus_beta1 = ops.sub(LNS_ONE, beta1)

            for p in group["params"]:

                if p.grad is None:
                    continue

                grad = p.grad.view(torch.int64) # g_t
                data = p.data.view(torch.int64)

                # 1. flip sign if we want to maximise
                if maximize:
                    grad = ops.sub(LNS_ZERO, grad) # −∇f

                # 2. weight decay: g ← g + λθ
                if not ops.equal(weight_decay, LNS_ZERO):
                    grad = ops.add(grad, ops.mul(data, weight_decay))

                state = self.state[p]
                if len(state) == 0:
                    # First time we see this parameter
                    state["step"] = 0
                    state["exp_avg"] = ops.zeros_like(data) # m_0
                    state["inf_norm"] = ops.zeros_like(data) # u_0

                # Retrieve running stats
                exp_avg = state["exp_avg"] # m_{t-1}
                inf_norm = state["inf_norm"] # u_{t-1}
                state["step"] += 1
                t = state["step"]

                # 3. m_t ← β_1*m_{t-1} + (1 − β_1)*g
                exp_avg = ops.add(
                    ops.mul(exp_avg, beta1),
                    ops.mul(grad, one_minus_beta1)
                )

                # 4. u_t ← max(β_2*u_{t-1}, |g_t| + ε)
                inf_norm = ops.maximum(
                    ops.mul(inf_norm, beta2),
                    ops.add(ops.abs(grad), eps)
                )

                # 5. θ ← θ − γ*m / (sqrt(1 - b_1^t) * u)
                t_tensor = torch.tensor(t, dtype=torch.int64)
                one_minus_beta1_t = ops.sub(LNS_ONE, ops.pow(beta1, t_tensor))
                denom = ops.mul(one_minus_beta1_t, inf_norm)
                step_size = ops.mul(lr, ops.div(exp_avg, denom))
                p.data = ops.sub(data, step_size).view(torch.float64)

                state["exp_avg"] = exp_avg
                state["inf_norm"] = inf_norm

        return loss
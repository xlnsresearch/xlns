import torch
from xlnstorch import LNS_ZERO, LNS_ONE
from . import LNSOptimizer

class LNSRAdam(LNSOptimizer):
    """
    Implements the rectified Adam optimization algorithm for LNSTensor
    parameters, including decoupled weight decay, and a "maximize" mode.

    This optimizer is analogous to PyTorch's :py:class:`torch.optim.RAdam`,
    but is designed to work with LNSTensor objects. See the PyTorch
    documentation for more details on the RAdam algorithm.

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
    decoupled_weight_decay : bool, optional
        If True, applies decoupled weight decay (default: False).
    maximize : bool, optional
        If True, optimizes the parameters for maximization instead of minimization (default: False).
    """

    def __init__(
            self,
            params,
            lr=0.001,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.0,
            decoupled_weight_decay=False,
            *,
            maximize=False
        ):

        if lr <= 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")

        if not (0.0 <= betas[0] < 1.0):
            raise ValueError(f"Invalid beta1 value: {betas[0]}")

        if not (0.0 <= betas[1] < 1.0):
            raise ValueError(f"Invalid beta2 value: {betas[1]}")

        if eps <= 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")

        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(
            lr=lr,
            beta1=betas[0],
            beta2=betas[1],
            eps=eps,
            weight_decay=weight_decay,
            decoupled_weight_decay=decoupled_weight_decay,
            maximize=maximize,
        )
        super(LNSRAdam, self).__init__(params, defaults)
        self.make_lnstensor_params("lr", "beta1", "beta2", "eps", "weight_decay")

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
            decoupled_weight_decay = group["decoupled_weight_decay"]
            maximize = group["maximize"]

            two = ops.to_lns(2.0)
            four = ops.to_lns(4.0)
            five = ops.to_lns(5.0)

            one_minus_beta1 = ops.sub(LNS_ONE, beta1)
            one_minus_beta2 = ops.sub(LNS_ONE, beta2)
            rho_inf = ops.sub(ops.div(two, ops.sub(LNS_ONE, beta2)), LNS_ONE)

            for p in group["params"]:

                if p.grad is None:
                    continue

                grad = p.grad.view(torch.int64) # g_t
                data = p.data.view(torch.int64)
                state = self.state[p]

                # 1. flip sign if we want to maximise
                if maximize:
                    grad = ops.neg(grad)

                # 2. weight decay:
                if not ops.equal(weight_decay, LNS_ZERO):
                    if decoupled_weight_decay:
                        # θ ← θ − γ λ θ
                        wd_step = ops.mul(lr, weight_decay)
                        data = ops.sub(data, ops.mul(data, wd_step))
                    else:
                        # g ← g + λ θ
                        grad = ops.add(grad, ops.mul(weight_decay, data))

                if len(state) == 0:
                    # First time we see this parameter
                    state["step"] = torch.tensor(0, dtype=torch.int64)
                    state["exp_avg"] = ops.zeros_like(data) # m_0
                    state["exp_avg_sq"] = ops.zeros_like(data) # v_0

                # Retrieve running stats
                t = state["step"] + 1
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]

                # 3. first and second moments:
                # m_t ← β_1*m_{t-1} + (1-β_1)*g_t
                # v_t ← β_2*v_{t-1} + (1-β_2)*g_t^2
                exp_avg = ops.add(
                    ops.mul(exp_avg, beta1),
                    ops.mul(grad, one_minus_beta1)
                )
                grad_sq = ops.mul(grad, grad)
                exp_avg_sq = ops.add(
                    ops.mul(exp_avg_sq, beta2),
                    ops.mul(grad_sq, one_minus_beta2)
                )

                # 5. bias-corrected first moment: m'_t ← m_t / (1 - β_1^t)
                beta1_pow = ops.pow(beta1, t)
                one_minus_beta1_pow = ops.sub(LNS_ONE, beta1_pow)
                exp_avg_hat = ops.div(exp_avg, one_minus_beta1_pow)

                # 6. ρ_t ← ρ_∞ - 2t*β_2^t / (1 - β_2^t)
                t_lns = ops.to_lns(t)
                beta2_pow = ops.pow(beta2, t)
                one_minus_beta2_pow = ops.sub(LNS_ONE, beta2_pow)
                corr_term = ops.div(
                    ops.mul(two, ops.mul(t_lns, beta2_pow)),
                    one_minus_beta2_pow
                )
                rho_t = ops.sub(rho_inf, corr_term)

                # 7. update rule based on ρ_t:
                if ops.gt(rho_t, five):
                    # l_t ← sqrt(1-β_2^t) / (sqrt(v_t) + ε)
                    l_t = ops.div(
                        ops.sqrt(one_minus_beta2_pow),
                        ops.add(ops.sqrt(exp_avg_sq), eps)
                    )
                    # r_t ← sqrt((ρ_t−4)(ρ_t−2)ρ_∞ / ((ρ_∞−4)(ρ_∞−2)ρ_t))
                    r_t_num = ops.mul(rho_inf, ops.mul(ops.sub(rho_t, four), ops.sub(rho_t, two)))
                    r_t_den = ops.mul(rho_t, ops.mul(ops.sub(rho_inf, four), ops.sub(rho_inf, two)))
                    r_t = ops.sqrt(ops.div(r_t_num, r_t_den))
                    # step ← γ * m'_t * l_t * r_t
                    step = ops.mul(lr, ops.mul(exp_avg_hat, ops.mul(l_t, r_t)))
                else:
                    # step ← γ * m'_t
                    step = ops.mul(lr, exp_avg_hat)

                # 8. Update parameters: θ ← θ − step
                p.data = ops.sub(data, step).view(torch.float64)

                state["step"] = t
                state["exp_avg"] = exp_avg
                state["exp_avg_sq"] = exp_avg_sq

        return loss
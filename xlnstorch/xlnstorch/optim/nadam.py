import torch
from xlnstorch import LNS_ZERO, LNS_ONE
from . import LNSOptimizer

class LNSNAdam(LNSOptimizer):
    """
    Implements the Adam optimization algorithm for LNSTensor parameters,
    including decoupled weight decay, momentum decay, and a "maximize" mode.

    This optimizer is analogous to PyTorch's :py:class:`torch.optim.NAdam`,
    but is designed to work with LNSTensor objects. See the PyTorch
    documentation for more details on the NAdam algorithm.

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
    momentum_decay : LNSTensor or float, optional
        Decay factor for the momentum term (default: 0.004). Must be a non-negative LNSTensor or float.
    decoupled_weight_decay : bool, optional
        If True, applies decoupled weight decay (default: False).
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
            momentum_decay=0.004,
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

        if momentum_decay < 0.0:
            raise ValueError(f"Invalid momentum_decay value: {momentum_decay}")

        defaults = dict(
            lr=lr,
            beta1=betas[0],
            beta2=betas[1],
            eps=eps,
            weight_decay=weight_decay,
            momentum_decay=momentum_decay,
            decoupled_weight_decay=decoupled_weight_decay,
            maximize=maximize,
        )
        super(LNSNAdam, self).__init__(params, defaults)
        self.make_lnstensor_params("lr", "beta1", "beta2", "eps", "weight_decay", "momentum_decay")

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
            momentum_decay = group["momentum_decay"]
            decoupled_weight_decay = group["decoupled_weight_decay"]
            maximize = group["maximize"]

            half = ops.to_lns(0.5)
            point_nine_six = ops.to_lns(0.96)

            one_minus_beta1 = ops.sub(LNS_ONE, beta1)
            one_minus_beta2 = ops.sub(LNS_ONE, beta2)

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
                        data  = ops.sub(data, ops.mul(data, wd_step))
                    else:
                        # g ← g + λ θ
                        grad = ops.add(grad, ops.mul(weight_decay, data))

                if len(state) == 0:
                    # First time we see this parameter
                    state["step"] = torch.tensor(0, dtype=torch.int64)
                    state["exp_avg"] = ops.zeros_like(data) # m_0
                    state["exp_avg_sq"] = ops.zeros_like(data) # v_0
                    state["mu_product"] = LNS_ONE.clone() # Πμ

                # Retrieve running stats
                t = state["step"] + 1
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                mu_product = state["mu_product"]

                # 3. compute μ_t and μ_{t+1}
                momentum_decay_fp = ops.from_lns(momentum_decay)
                pow_t = ops.pow(point_nine_six, t * momentum_decay_fp)
                pow_next = ops.pow(point_nine_six, (t + 1) * momentum_decay_fp)
                mu = ops.mul(beta1, ops.sub(LNS_ONE, ops.mul(half, pow_t)))
                mu_next = ops.mul(beta1, ops.sub(LNS_ONE, ops.mul(half, pow_next)))

                # 4. first and second moments:
                # m_t ← β_1*m_{t-1} + (1 − β_1)*g_t
                # v_t ← β_2*v_{t-1} + (1 − β_2)*g_t^2
                exp_avg = ops.add(
                    ops.mul(exp_avg, beta1),
                    ops.mul(grad, one_minus_beta1)
                )
                grad_sq = ops.mul(grad, grad)
                exp_avg_sq = ops.add(
                    ops.mul(exp_avg_sq, beta2),
                    ops.mul(grad_sq, one_minus_beta2)
                )

                # 5. calculate next mu product: Π_{t+1}
                mu_product = ops.mul(mu_product, mu)
                mu_product_next = ops.mul(mu_product, mu_next)
                one_minus_mu_product = ops.sub(LNS_ONE, mu_product)
                one_minus_mu_product_next = ops.sub(LNS_ONE, mu_product_next)

                # 6. bias correction: m'_t = m_t / (1 − Π_{t+1})
                term1 = ops.div(ops.mul(mu_next, exp_avg), one_minus_mu_product_next)
                one_minus_mu_t = ops.sub(LNS_ONE, mu)
                term2 = ops.div(ops.mul(one_minus_mu_t, grad), one_minus_mu_product)
                exp_avg_hat = ops.add(term1, term2)

                # 7. bias correction: v'_t = v_t / (1 − β_2^t)
                beta_2_pow = ops.pow(beta2, t)
                exp_avg_sq_hat = ops.div(exp_avg_sq, ops.sub(LNS_ONE, beta_2_pow))

                # 8. Update parameters: θ ← θ − γ*m' / (sqrt(v') + ε)
                denom = ops.add(ops.sqrt(exp_avg_sq_hat), eps)
                step_size = ops.mul(lr, ops.div(exp_avg_hat, denom))
                p.data = ops.sub(data, step_size).view(torch.float64)

                state["step"] = t
                state["exp_avg"] = exp_avg
                state["exp_avg_sq"] = exp_avg_sq
                state["mu_product"] = mu_product

        return loss
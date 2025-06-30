import torch
from .. import LNSTensor, lnstensor, LNS_ZERO, align_lnstensor_bases, zeros_like
from ..operators import (
    lns_equal,
    lns_sub,
    lns_mul,
    lns_add,
    lns_pow,
    lns_div,
    lns_sqrt,
    lns_neg,
)
from . import LNSOptimizer

def _as_lnstensor(x):
    if isinstance(x, LNSTensor):
        return x
    else:
        return lnstensor(x)

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
        This should be obtained from a model's `parameter_groups()` method.
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
            lr=_as_lnstensor(lr),
            beta1=_as_lnstensor(betas[0]),
            beta2=_as_lnstensor(betas[1]),
            eps=_as_lnstensor(eps),
            weight_decay=_as_lnstensor(weight_decay),
            momentum_decay=_as_lnstensor(momentum_decay),
            decoupled_weight_decay=decoupled_weight_decay,
            maximize=maximize,
        )
        super(LNSNAdam, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1 = group["beta1"]
            beta2 = group["beta2"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            momentum_decay = group["momentum_decay"]
            decoupled_weight_decay = group["decoupled_weight_decay"]
            maximize = group["maximize"]
            base = group["base"]

            # Align the parameters to the base of the group.
            lr, beta1, beta2, eps, weight_decay, momentum_decay = align_lnstensor_bases(
                lr, beta1, beta2, eps, weight_decay, momentum_decay, base=base)
            
            one = LNSTensor.get_internal_tensor(1.0, base)
            half = LNSTensor.get_internal_tensor(0.5, base)
            point_nine_six = LNSTensor.get_internal_tensor(0.96, base)

            one_minus_beta1 = lns_sub(one, beta1._lns, base)
            one_minus_beta2 = lns_sub(one, beta2._lns, base)

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad # g_t
                state = self.state[p]

                # 1. flip sign if we want to maximise
                if maximize:
                    grad = lns_neg(grad)

                # 2. weight decay:
                if not lns_equal(weight_decay._lns, LNS_ZERO):
                    if decoupled_weight_decay:
                        # θ ← θ − γ λ θ
                        wd_step = lns_mul(lr._lns, weight_decay._lns)
                        p.data  = lns_sub(p.data, lns_mul(p.data, wd_step), base)
                    else:
                        # g ← g + λ θ
                        grad = lns_add(grad, lns_mul(weight_decay._lns, p.data), base)

                if len(state) == 0:
                    # First time we see this parameter
                    state["step"] = torch.tensor(0, dtype=torch.int64)
                    state["exp_avg"] = zeros_like(p.data, b=base)._lns # m_0
                    state["exp_avg_sq"] = zeros_like(p.data, b=base)._lns # v_0
                    state["mu_product"] = one.clone() # Πμ

                # Retrieve running stats
                t = state["step"] + 1
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                mu_product = state["mu_product"]

                # 3. compute μ_t and μ_{t+1}
                pow_t = lns_pow(point_nine_six, t * momentum_decay.value, base)
                pow_next = lns_pow(point_nine_six, (t + 1) * momentum_decay.value, base)
                mu = lns_mul(beta1._lns, lns_sub(one, lns_mul(half, pow_t), base))
                mu_next = lns_mul(beta1._lns, lns_sub(one, lns_mul(half, pow_next), base))

                # 4. first and second moments:
                # m_t ← β_1*m_{t-1} + (1 − β_1)*g_t
                # v_t ← β_2*v_{t-1} + (1 − β_2)*g_t^2
                exp_avg = lns_add(
                    lns_mul(exp_avg, beta1._lns),
                    lns_mul(grad, one_minus_beta1),
                    base
                )
                grad_sq = lns_mul(grad, grad)
                exp_avg_sq = lns_add(
                    lns_mul(exp_avg_sq, beta2._lns),
                    lns_mul(grad_sq, one_minus_beta2),
                    base
                )

                # 5. calculate next mu product: Π_{t+1}
                mu_product = lns_mul(mu_product, mu)
                mu_product_next = lns_mul(mu_product, mu_next)
                one_minus_mu_product = lns_sub(one, mu_product, base)
                one_minus_mu_product_next = lns_sub(one, mu_product_next, base)

                # 6. bias correction: m'_t = m_t / (1 − Π_{t+1})
                term1 = lns_div(lns_mul(mu_next, exp_avg), one_minus_mu_product_next, base)
                one_minus_mu_t = lns_sub(one, mu, base)
                term2 = lns_div(lns_mul(one_minus_mu_t, grad), one_minus_mu_product, base)
                exp_avg_hat = lns_add(term1, term2, base)

                # 7. bias correction: v'_t = v_t / (1 − β_2^t)
                beta_2_pow = lns_pow(beta2._lns, t, base)
                exp_avg_sq_hat = lns_div(exp_avg_sq, lns_sub(one, beta_2_pow, base), base)

                # 8. Update parameters: θ ← θ − γ*m' / (sqrt(v') + ε)
                denom = lns_add(lns_sqrt(exp_avg_sq_hat, base), eps._lns, base)
                step_size = lns_mul(lr._lns, lns_div(exp_avg_hat, denom, base))
                p.data = lns_sub(p.data, step_size, base)

                state["step"] = t
                state["exp_avg"] = exp_avg
                state["exp_avg_sq"] = exp_avg_sq
                state["mu_product"] = mu_product

        return loss
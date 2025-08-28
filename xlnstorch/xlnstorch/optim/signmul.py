import torch
from xlnstorch import LNSTensor, lnstensor, LNS_ZERO, LNS_ONE
from xlnstorch.operators import (
    lns_mul,
    lns_sign,
    lns_eq,
    lns_sqrt,
    lns_div,
    lns_sum,
    lns_neg,
    lns_clamp,
)
from . import LNSOptimizer

class LNSSignMul(LNSOptimizer):
    r"""
    Implements a simple sign multiplication algorithm for LNSTensor
    parameters.

    .. math::
        \begin{aligned}
            &\rule{120mm}{0.4pt}                                                \\
            &\textbf{input} : \gamma \text{ (lr)},\;
                              \theta_{0} \text{ (params)},\;
                              f(\theta) \text{ (objective)},\;
                              \textit{use_pow},\;                               \\
            &\hspace{17mm}    \textit{maximize},\;
                              \sigma \text{ (weight scale)}                     \\
            &\textbf{initialize} :                                              \\
            &\hspace{5mm}\alpha \; = \;
                \begin{cases}
                    2^{\gamma}, & \text{if } \textit{use_pow}                   \\
                    1 + \gamma, & \text{otherwise}
                \end{cases}
                \quad\text{(primary multiplier)},                               \\
            &\hspace{5mm}  \alpha^{-1} \; = \; 1 / \alpha
                \quad\text{(inverse multiplier)},                               \\
            &\hspace{5mm} \sigma^{*} \leftarrow \sigma \cdot
                          \operatorname{RMS}\left(\theta_{0}\right)
                          \text{ (max weight)}                                  \\[-1.ex]
            &\rule{120mm}{0.4pt}                                                \\
            &\textbf{for } t = 1 \textbf{ to } \ldots \textbf{ do}              \\
            &\hspace{5mm}\textbf{if } \textit{maximize}:                        \\
            &\hspace{10mm} g_t \leftarrow
                           -\nabla_{\theta} f_t \left(\theta_{t-1}\right)       \\
            &\hspace{5mm}\textbf{else}:                                         \\
            &\hspace{10mm} g_t \leftarrow
                           \nabla_{\theta} f_t \left(\theta_{t-1}\right)        \\
            &\hspace{5mm} u_t \leftarrow
                \begin{cases}
                    1,           &g_t = 0                                       \\
                    \alpha^{-1}, &\operatorname{sign} \bigl(\theta_{t-1}\bigr)
                                 = \operatorname{sign} \left(g_t\right)         \\
                    \alpha,      &\text{otherwise}
                \end{cases}                                                     \\
            &\hspace{5mm} \theta_t \leftarrow \theta_{t-1} \cdot u_t            \\
            &\hspace{5mm} \theta_t \leftarrow
                          \operatorname{clamp_{\sigma^{*}}}
                          \left(\theta_t\right)                                 \\[-1.ex]
            &\rule{120mm}{0.4pt}                                                \\[-1.ex]
            &\textbf{return } \theta_t                                          \\[-1.ex]
            &\rule{120mm}{0.4pt}                                                \\
        \end{aligned}

    Parameters
    ----------
    params : iterable
        An iterable of parameters to optimize or dicts defining parameter groups.
        This should be obtained from a model's `lns_parameters()` method.
    lr : LNSTensor, float, optional
        Learning rate (default: 0.01). Must be a non-negative LNSTensor or float.
    use_pow : bool, optional
        If True, uses a power-based multiplier (default: False).
    maximize : bool, optional
        If True, optimizes the parameters for maximization instead of minimization (default: False).
    """

    def __init__(
            self,
            params,
            lr=0.01,
            p_scale=10.0,
            use_pow=False,
            *,
            maximize=False
        ):

        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")

        defaults = dict(
            lr=lr,
            p_scale=p_scale,
            use_pow=use_pow,
            maximize=maximize
        )
        super(LNSSignMul, self).__init__(params, defaults)
        self.make_lnstensor_params("lr", "p_scale")

        # precompute 1 + lr and 1 / (1 + lr)
        for group in self.param_groups:
            base = group["base"]
            lr_ = lnstensor(group["lr"], from_lns=True, b=base)
            use_pow_ = group["use_pow"]

            if use_pow_:
                mul_term = 2.0 ** lr_
            else:
                mul_term = 1.0 + lr_

            group["mul_term"] = mul_term._lns
            group["inv_mul_term"] = (1.0 / mul_term)._lns

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step."""

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            p_scale = group["p_scale"]
            mul_term = group["mul_term"]
            inv_mul_term = group["inv_mul_term"]
            maximize = group["maximize"]
            base = group["base"]

            for p in group["params"]:

                if p.grad is None:
                    continue

                grad = p.grad

                state = self.state[p]
                if len(state) == 0:
                    # First time we see this parameter
                    rms = lns_sqrt(lns_div(lns_sum(lns_mul(p, p, base), base),
                                           LNSTensor.get_internal_tensor(p.numel(), base),
                                           base), base)
                    state['max'] = lns_mul(p_scale, rms, base)

                # retrieve running stats
                max = state['max']

                grad_sign = lns_sign(grad, base)
                p_sign = lns_sign(p, base)

                mul_update = torch.where(
                    lns_eq(grad, LNS_ZERO), LNS_ONE,
                    torch.where(lns_eq(grad_sign, p_sign) ^ maximize,
                    inv_mul_term, mul_term
                ))
                p.data = lns_mul(p.data, mul_update, base)
                p.data = lns_clamp(p.data, lns_neg(max), max)

        return loss
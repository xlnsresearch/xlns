import torch
from xlnstorch import LNS_ONE
from . import LNSOptimizer

class LNSMadam(LNSOptimizer):
    r"""
    Implements the Madam optimizer for LNSTensor parameters. See

    - `LNS-Madam: Low-Precision Training in Logarithmic Number System using
      Multiplicative Weight Update <https://arxiv.org/pdf/2106.13914>`__
    - `Learning compositional functions via multiplicative weight updates
      <https://arxiv.org/pdf/2006.14560>`__

    for more details on the algorithm.

    .. math::
        \begin{aligned}
            &\rule{130mm}{0.4pt}                                                \\
            &\textbf{input} : \gamma \text{ (lr)},\;
                              \beta \text{ (beta)},\;
                              \epsilon \text{ (epsilon)},\;
                              \mu \text{ (max perturbation)},\;                 \\
            &\hspace{17mm}    \sigma \text{ (weight scale)},\;
                              \theta_{0} \text{ (params)},\;
                              f(\theta) \text{ (objective)},\;                  \\
            &\hspace{17mm}    \textit{use_pow},\;
                              \textit{maximize}                                 \\
            &\textbf{initialize} : \sigma^{*} \leftarrow \sigma \cdot
                                   \operatorname{RMS}\left(\theta_{0}\right)
                                   \text{ (max weight)},\;                      \\
            &\hspace{26mm}         v_0 \leftarrow 0 \text{ (second moment)}     \\[-1.ex]
            &\rule{130mm}{0.4pt}                                                \\
            &\textbf{for } t = 1 \textbf{ to } \ldots \textbf{ do}              \\
            &\hspace{5mm} g_t \leftarrow
                          \nabla_{\theta} f_t \left(\theta_{t-1}\right)         \\
            &\hspace{5mm} \rho_t \leftarrow 1 - \beta^{t}                       \\
            &\hspace{5mm} v_t \leftarrow (1 - \beta) g_t^2 + \beta v_{t-1}      \\
            &\hspace{5mm} g^{*}_t \leftarrow g_t /
                          \sqrt{v_t / \rho_t + \epsilon}                        \\
            &\hspace{5mm} \chi_t \leftarrow -\gamma
                          \operatorname{sign} \bigl(\theta_{t-1}\bigr)
                          \operatorname{clamp_{\mu}} \bigl(g^{*}_t\bigr)        \\
            &\hspace{5mm} \textbf{if } \textit{maximize}:                       \\
            &\hspace{10mm} \chi_t \leftarrow -\chi_t                            \\
            &\hspace{5mm} \textbf{if } \textit{use_pow}:                        \\
            &\hspace{10mm} \theta_t \leftarrow \theta_{t-1} \cdot
                           \exp \left(\chi_t\right)                             \\
            &\hspace{5mm} \textbf{else}:                                        \\
            &\hspace{10mm} \theta_t \leftarrow \theta_{t-1} \cdot
                            \left(1 + \chi_t\right)                             \\
            &\hspace{5mm} \theta_t \leftarrow
                          \operatorname{clamp_{\sigma^{*}}}
                          \left(\theta_t\right)                                 \\[-1.ex]
            &\rule{130mm}{0.4pt}                                                \\[-1.ex]
            &\textbf{return } \theta_t                                          \\[-1.ex]
            &\rule{130mm}{0.4pt}                                                \\
        \end{aligned}

    Parameters
    ----------
    params : iterable
        An iterable of parameters to optimize or dicts defining parameter groups.
        This should be obtained from a model's `lns_parameters()` method.
    lr : LNSTensor, float, optional
        Learning rate (default: 0.01). Must be a non-negative LNSTensor or float.
    beta : LNSTensor, float, optional
        Coefficient used for computing the running average of the gradient (default: 0.999).
        Must be a non-negative LNSTensor or float in the range (0.0, 1.0).
    eps : LNSTensor, float, optional
        Term added to the denominator for numerical stability (default: 1e-8).
    p_scale : LNSTensor, float, optional
        Scaling factor for the parameter update (default: 3.0). Must be a non-negative
        LNSTensor or float.
    g_bound : LNSTensor, float, optional
        Bound for the gradient norm (default: 10.0). Must be a non-negative LNSTensor
        or float.
    use_pow : bool, optional
        If True, uses a power-based multiplier (default: False).
    maximize : bool, optional
        If True, optimizes the parameters for maximization instead of minimization (default: False).
    """

    def __init__(
            self,
            params,
            lr=0.01,
            beta=0.999,
            eps=1e-8,
            p_scale=3.0,
            g_bound=10.0,
            use_pow=False,
            *,
            maximize=False
    ):

        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")

        if eps <= 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")

        if not (0.0 < beta < 1.0):
            raise ValueError(f"Invalid beta value: {beta}")

        defaults = dict(
            lr=lr,
            beta=beta,
            eps=eps,
            p_scale=p_scale,
            g_bound=g_bound,
            use_pow=use_pow,
            maximize=maximize
        )
        super(LNSMadam, self).__init__(params, defaults)
        self.make_lnstensor_params("lr", "beta", "eps", "p_scale", "g_bound")

    @torch.no_grad()
    def step(self, closure=None):

        loss = None
        if closure is not None:
            loss = closure()

        for group, ops in self.lns_param_groups():
            lr = group["lr"]
            beta = group["beta"]
            eps = group["eps"]
            p_scale = group["p_scale"]
            g_bound = group["g_bound"]
            use_pow = group["use_pow"]
            maximize = group["maximize"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad.view(torch.int64) # g_t
                data = p.data.view(torch.int64)

                state = self.state[p]
                if len(state) == 0:
                    # First time we see this parameter
                    rms = ops.sqrt(ops.div(ops.sum(ops.mul(data, data)),
                                           ops.to_lns(data.numel())))
                    state['max'] = ops.mul(p_scale, rms)
                    state['step'] = 0
                    state['exp_avg_sq'] = ops.zeros_like(data)

                # retrieve running stats
                max = state['max']
                step = state['step'] + 1
                exp_avg_sq = state['exp_avg_sq']

                bias_correction = ops.sub(LNS_ONE, ops.pow(beta, torch.tensor(step)))
                exp_avg_sq = ops.add(
                    ops.mul(beta, exp_avg_sq),
                    ops.mul(ops.sub(LNS_ONE, beta), ops.mul(grad, grad))
                )
                corrected_exp_avg_sq = ops.add(ops.div(exp_avg_sq, bias_correction), eps)

                g_normed = ops.div(grad, ops.sqrt(corrected_exp_avg_sq))
                g_normed = ops.clamp(g_normed, ops.neg(g_bound), g_bound)

                if use_pow:
                    if maximize:
                        exponent = ops.mul(lr, ops.mul(g_normed, ops.sign(data)))
                    else:
                        exponent = ops.mul(ops.neg(lr), ops.mul(g_normed, ops.sign(data)))
                    data = ops.mul(data, ops.exp(exponent))

                else:
                    if maximize:
                        mul_term = ops.add(LNS_ONE, ops.mul(lr, ops.mul(g_normed, ops.sign(data))))
                    else:
                        mul_term = ops.sub(LNS_ONE, ops.mul(lr, ops.mul(g_normed, ops.sign(data))))
                    data = ops.mul(data, mul_term)

                p.data = ops.clamp(data, ops.neg(max), max).view(torch.float64)

                # update running stats
                state['step'] = step
                state['exp_avg_sq'] = exp_avg_sq

        return loss
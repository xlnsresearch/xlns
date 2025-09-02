import torch
from xlnstorch import LNS_ZERO, LNS_ONE
from . import LNSOptimizer

class LNSRMSprop(LNSOptimizer):
    """
    Implements the RMSprop algorithm with support for weight decay,
    momentum, and centered variants.

    This optimizer is analogous to PyTorch's :py:class:`torch.optim.RMSprop`,
    but is designed to work with LNSTensor objects. See the PyTorch documentation
    for more details on the RMSprop algorithm.

    Parameters
    -----------
    params : iterable
        An iterable of parameters to optimize or dicts defining parameter groups.
        This should be obtained from a model's `lns_parameters()` method.
    lr : LNSTensor, float, optional
        Learning rate (default: 0.01). Must be a non-negative LNSTensor or float.
    alpha : LNSTensor, float, optional
        Smoothing constant (default: 0.99). Must be a non-negative LNSTensor or float
        in the range [0.0, 1.0).
    eps : LNSTensor, float, optional
        Term added to the denominator to improve numerical stability (default: 1e-08).
        Must be a non-negative LNSTensor or float.
    weight_decay : LNSTensor, float, optional
        Weight decay (L2 penalty) (default: 0.0). Must be a non-negative LNSTensor or float.
    momentum : LNSTensor, float, optional
        Momentum factor (default: 0.0). Must be a non-negative LNSTensor or float.
    centered : bool, optional
        If True, computes the centered RMSprop variant (default: False).
    maximize : bool, optional
        If True, optimizes the parameters for maximization instead of minimization (default: False).
    """

    def __init__(
            self,
            params,
            lr=0.01,
            alpha=0.99,
            eps=1e-08,
            weight_decay=0.0,
            momentum=0.0,
            centered=False,
            *,
            maximize=False,
        ):

        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")

        if not (0.0 <= alpha < 1.0):
            raise ValueError(f"Invalid alpha value: {alpha}")

        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")

        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")

        defaults = dict(
            lr=lr,
            alpha=alpha,
            eps=eps,
            weight_decay=weight_decay,
            momentum=momentum,
            centered=centered,
            maximize=maximize
        )
        super(LNSRMSprop, self).__init__(params, defaults)
        self.make_lnstensor_params("lr", "alpha", "eps", "weight_decay", "momentum")

    @torch.no_grad()
    def step(self, closure=None):

        loss = None
        if closure is not None:
            loss = closure()

        for group, ops in self.lns_param_groups():
            lr = group["lr"]
            alpha = group["alpha"]
            eps = group["eps"]
            weight_decay= group["weight_decay"]
            momentum = group["momentum"]
            centered = group["centered"]
            maximize = group["maximize"]

            one_minus_alpha = ops.sub(LNS_ONE, alpha)

            for p in group["params"]:

                if p.grad is None:
                    continue

                grad  = p.grad.view(torch.int64) # g_t
                data = p.data.view(torch.int64)
                state = self.state[p]

                if maximize:
                    grad = ops.neg(grad)

                if not ops.equal(weight_decay, LNS_ZERO):
                    grad = ops.add(grad, ops.mul(data, weight_decay))

                if len(state) == 0:
                    # First time we see this parameter
                    zeros = ops.zeros_like(data)
                    state["square_avg"] = zeros.clone()
                    state["grad_avg"] = zeros.clone()
                    state["momentum_buffer"] = zeros.clone()

                # Retrieve running stats
                square_avg = state["square_avg"]
                grad_avg = state["grad_avg"]
                buf = state["momentum_buffer"]

                # 1. square_avg: v_t ← α v_{t-1} + (1-α) * g_t²
                grad_sq = ops.mul(grad, grad)
                square_avg = ops.add(
                    ops.mul(square_avg, alpha), # α v_{t-1}
                    ops.mul(grad_sq, one_minus_alpha), # (1-α) g_t²
                )

                # 2. centered:
                # g_avg ← α g_avg + (1-α) * g_t
                # v'_t = v_t - (g_avg) ^ 2
                if centered:
                    grad_avg = ops.add(
                        ops.mul(grad_avg, alpha), # α g_avg
                        ops.mul(grad, one_minus_alpha), # (1-α) * g_t
                    )
                    avg_sq = ops.mul(grad_avg, grad_avg)
                    denom = ops.sub(square_avg, avg_sq)

                else:
                    denom = square_avg

                # denominator: sqrt((v'_t) + ε)
                denom = ops.add(ops.sqrt(denom), eps)

                # 3. momentum buffering:
                if not ops.equal(momentum, LNS_ZERO):
                    # b_t ← μ b_{t-1} + g_t / denom
                    buf_div = ops.div(grad, denom)
                    buf = ops.add(ops.mul(buf, momentum), buf_div)

                    # θ ← θ - γ * b_t
                    delta = ops.mul(buf, lr)

                else:
                    # θ ← θ - γ * g_t / denom
                    step_dir = ops.div(grad, denom)
                    delta = ops.mul(step_dir, lr)

                # 4. parameter update: θ ← θ - γ * b_t
                p.data = ops.sub(data, delta).view(torch.float64)

                state["square_avg"] = square_avg
                state["grad_avg"] = grad_avg
                state["momentum_buffer"] = buf

        return loss
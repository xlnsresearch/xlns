import torch
from xlnstorch import LNS_ZERO, LNS_ONE, lnstensor
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

        self.validate_param("lr", lambda lr: lr >= 0.0)
        self.validate_param("momentum", lambda momentum: momentum >= 0.0)
        self.validate_param("dampening", lambda dampening: dampening >= 0.0)
        self.validate_param("weight_decay", lambda weight_decay: weight_decay >= 0.0)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step."""

        loss = None
        if closure is not None:
            loss = closure()

        for group, ops in self.lns_param_groups():
            lr = group["lr"]
            momentum = group["momentum"]
            dampening = group["dampening"]
            weight_decay = group["weight_decay"]
            nesterov = group["nesterov"]
            maximize = group["maximize"]

            for p in group["params"]:

                if p.grad is None:
                    continue

                grad = p.grad.view(torch.int64) # g_t
                data = p.data.view(torch.int64)
                state = self.state[p]

                if maximize:
                    grad = ops.neg(grad)

                # 1. weight_decay: g ← g + λθ
                if not ops.equal(weight_decay, LNS_ZERO):
                    grad = ops.add(grad, ops.mul(data, weight_decay))

                # 2. momentum buffering: b ← μb + (1 - τ)g
                if not ops.equal(momentum, LNS_ZERO):
                    buf = state.get("momentum_buffer", None)

                    if buf is None:
                        buf = grad.clone()
                        state["momentum_buffer"] = buf

                    else:
                        one_minus_tau = ops.sub(LNS_ONE, dampening)
                        buf = ops.add(
                            ops.mul(buf, momentum),
                            ops.mul(grad, one_minus_tau)
                        )
                        state["momentum_buffer"] = buf

                    # 3a. nesterov momentum: g ← g + μb
                    if nesterov:
                        grad = ops.add(grad, ops.mul(buf, momentum))
                    # 3b. classical momentum: g ← b 
                    else:
                        grad = buf

                # 4. parameter update: θ ← θ ± γg
                delta = ops.mul(grad, lr)
                p.data = ops.sub(data, delta).view(torch.float64)

        return loss
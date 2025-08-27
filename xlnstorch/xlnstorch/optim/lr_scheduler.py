from typing import Callable, List, Union
from typing_extensions import override
from bisect import bisect_right
import torch
from xlnstorch import LNSTensor, lnstensor, LNS_ONE
from . import LNSOptimizer
from xlnstorch.operators import (
    lns_mul,
    lns_pow,
    lns_div,
    lns_add,
    lns_sub,
    lns_minimum,
    lns_maximum,
    lns_gt,
)

def _lns(value: Union[float, LNSTensor], base) -> LNSTensor:
    if isinstance(value, LNSTensor):
        return lnstensor(value, b=base)._lns
    return LNSTensor.get_internal_tensor(value, base)

def get_lr_bases(optimizer: LNSOptimizer) -> List[torch.Tensor]:
    """
    Returns a list of base values for the learning rates of each
    parameter group in the optimizer.

    Parameters
    ----------
    optimizer : LNSOptimizer
        The optimizer from which to extract the base learning rates.

    Returns
    -------
    List[torch.Tensor]
        A list of base learning rates for each parameter group.

    Raises
    ------
    TypeError
        If the provided optimizer is not an instance of `LNSOptimizer`.
    """
    if not isinstance(optimizer, LNSOptimizer):
        raise TypeError(f"{type(optimizer).__name__} is not an LNSOptimizer")

    return [group["base"] for group in optimizer.param_groups]

class LNSLambdaLR(torch.optim.lr_scheduler.LambdaLR):
    """
    An LNS learning rate scheduler that sets the learning rate of each parameter group
    to the initial learning rate multipled by a given function of the epoch.

    See also: :class:`torch.optim.lr_scheduler.LambdaLR`

    Parameters
    ----------
    optimizer : LNSOptimizer
        Wrapped optimizer.
    lr_lambda : Callable[[int], float | LNSTensor] or List[Callable[[int], float | LNSTensor]]
        A function or a list of functions which computes a multiplicative factor given an integer parameter
        `epoch`, which is the index of the current epoch.
    last_epoch : int, optional
        The index of last epoch. Default: -1.
    """

    def __init__(
            self,
            optimizer: LNSOptimizer,
            lr_lambda: Union[Callable[[int], Union[float, LNSTensor]], List[Callable[[int], Union[float, LNSTensor]]]],
            last_epoch: int = -1,
        ):
        self.lns_lr_bases = get_lr_bases(optimizer)
        super().__init__(optimizer, lr_lambda, last_epoch)

    @override
    def get_lr(self) -> List[torch.Tensor]:
        torch.optim.lr_scheduler._warn_get_lr_called_within_step(self)

        return [
            lns_mul(base_lr, _lns(lmbda(self.last_epoch), base), base)
            for lmbda, base_lr, base in zip(self.lr_lambdas, self.base_lrs, self.lns_lr_bases)
        ]

class LNSMultiplicativeLR(torch.optim.lr_scheduler.MultiplicativeLR):
    """
    An LNS learning rate scheduler that sets the learning rate of each parameter group
    to the previous learning rate multipled by a given multiplicative factor.

    See also: :class:`torch.optim.lr_scheduler.MultiplicativeLR`

    Parameters
    ----------
    optimizer : LNSOptimizer
        Wrapped optimizer.
    lr_lambda : Callable[[int], float | LNSTensor] or List[Callable[[int], float | LNSTensor]]
        A function or a list of functions which computes a multiplicative factor given an integer parameter
        `epoch`, which is the index of the current epoch.
    last_epoch : int, optional
        The index of last epoch. Default: -1.
    """

    def __init__(
            self,
            optimizer: LNSOptimizer,
            lr_lambda: Union[Callable[[int], Union[float, LNSTensor]], List[Callable[[int], Union[float, LNSTensor]]]],
            last_epoch: int = -1,
        ):
        self.lns_lr_bases = get_lr_bases(optimizer)
        super().__init__(optimizer, lr_lambda, last_epoch)

    @override
    def get_lr(self) -> List[torch.Tensor]:
        torch.optim.lr_scheduler._warn_get_lr_called_within_step(self)

        if self.last_epoch > 0:
            return [
                lns_mul(group["lr"], _lns(lmbda(self.last_epoch), base), base)
                for lmbda, group, base in zip(self.lr_lambdas, self.optimizer.param_groups, self.lns_lr_bases)
            ]

        return [group["lr"] for group in self.optimizer.param_groups]

class LNSStepLR(torch.optim.lr_scheduler.StepLR):
    """
    An LNS learning rate scheduler that decays the learning rate of each parameter
    group by a factor of `gamma` every `step_size` epochs.

    See also: :class:`torch.optim.lr_scheduler.StepLR`

    Parameters
    ----------
    optimizer : LNSOptimizer
        Wrapped optimizer.
    step_size : int
        Period of learning rate decay.
    gamma : float | LNSTensor
        Multiplicative factor of learning rate decay.
    last_epoch : int, optional
        The index of last epoch. Default: -1.
    """

    def __init__(
            self,
            optimizer: LNSOptimizer,
            step_size: int,
            gamma: Union[float, LNSTensor] = 0.1,
            last_epoch: int = -1,
        ):
        self.lns_lr_bases = get_lr_bases(optimizer)
        self.gammas = [_lns(gamma, base) for base in self.lns_lr_bases]
        super().__init__(optimizer, step_size, gamma, last_epoch)

    @override
    def get_lr(self) -> List[torch.Tensor]:
        torch.optim.lr_scheduler._warn_get_lr_called_within_step(self)

        if self.last_epoch == 0 or self.last_epoch % self.step_size != 0:
            return [group["lr"] for group in self.optimizer.param_groups]

        return [
            lns_mul(group["lr"], gamma, base)
            for group, gamma, base in zip(self.optimizer.param_groups, self.gammas, self.lns_lr_bases)
        ]

    def _get_closed_form_lr(self) -> List[torch.Tensor]:
        return [
            lns_mul(base_lr, lns_pow(gamma, torch.tensor(self.last_epoch // self.step_size), base), base)
            for base_lr, gamma, base in zip(self.base_lrs, self.gammas, self.lns_lr_bases)
        ]

class LNSMultiStepLR(torch.optim.lr_scheduler.MultiStepLR):
    """
    An LNS learning rate scheduler that decays the learning rate of each parameter
    group by a factor of `gamma` at specified epochs.

    See also: :class:`torch.optim.lr_scheduler.MultiStepLR`

    Parameters
    ----------
    optimizer : LNSOptimizer
        Wrapped optimizer.
    milestones : List[int]
        List of epoch indices where the learning rate should be decayed.
    gamma : float | LNSTensor
        Multiplicative factor of learning rate decay.
    last_epoch : int, optional
        The index of last epoch. Default: -1.
    """

    def __init__(
            self,
            optimizer: LNSOptimizer,
            milestones: List[int],
            gamma: Union[float, LNSTensor] = 0.1,
            last_epoch: int = -1,
        ):
        self.lns_lr_bases = get_lr_bases(optimizer)
        self.gammas = [_lns(gamma, base) for base in self.lns_lr_bases]
        super().__init__(optimizer, milestones, gamma, last_epoch)

    @override
    def get_lr(self) -> List[torch.Tensor]:
        torch.optim.lr_scheduler._warn_get_lr_called_within_step(self)

        if self.last_epoch not in self.milestones:
            return [group["lr"] for group in self.optimizer.param_groups]

        return [
            lns_mul(group["lr"], lns_pow(gamma, torch.tensor(self.milestones[self.last_epoch]), base), base)
            for group, gamma, base in zip(self.optimizer.param_groups, self.gammas, self.lns_lr_bases)
        ]

    def _get_closed_form_lr(self):
        milestones = sorted(self.milestones.elements())
        return [
            lns_mul(base_lr, lns_pow(gamma, torch.tensor(bisect_right(milestones, self.last_epoch)), base), base)
            for base_lr, gamma, base in zip(self.base_lrs, self.gammas, self.lns_lr_bases)
        ]

class LNSConstantLR(torch.optim.lr_scheduler.ConstantLR):
    """
    An LNS learning rate scheduler that sets the learning rate of each parameter group
    to a constant value until a pre-determined number of epochs is reached.

    See also: :class:`torch.optim.lr_scheduler.ConstantLR`

    Parameters
    ----------
    optimizer : LNSOptimizer
        Wrapped optimizer.
    factor : float | LNSTensor
        Multiplicative factor of the learning rate.
    total_iters : int, optional
        The number of iterations for which the learning rate will be constant.
        Default: 0.
    last_epoch : int, optional
        The index of last epoch. Default: -1.
    """

    def __init__(
            self,
            optimizer: LNSOptimizer,
            factor: Union[float, LNSTensor] = 1.0 / 3,
            total_iters: int = 0,
            last_epoch: int = -1,
        ):
        self.lns_lr_bases = get_lr_bases(optimizer)
        self.factors = [_lns(factor, base) for base in self.lns_lr_bases]
        super().__init__(optimizer, factor, total_iters, last_epoch)

    @override
    def get_lr(self) -> List[torch.Tensor]:
        torch.optim.lr_scheduler._warn_get_lr_called_within_step(self)

        if self.last_epoch == 0:
            return [
                lns_mul(base_lr, factor, base)
                for base_lr, factor, base in zip(self.base_lrs, self.factors, self.lns_lr_bases)
            ]

        if self.last_epoch != self.total_iters:
            return [group["lr"] for group in self.optimizer.param_groups]

        return [
            lns_div(group["lr"], factor, base)
            for group, factor, base in zip(self.optimizer.param_groups, self.factors, self.lns_lr_bases)
        ]

class LNSLinearLR(torch.optim.lr_scheduler.LinearLR):
    """
    An LNS learning rate scheduler that sets the learning rate of each parameter group
    to a linearly decaying value.

    See also: :class:`torch.optim.lr_scheduler.LinearLR`

    Parameters
    ----------
    optimizer : LNSOptimizer
        Wrapped optimizer.
    start_factor : float | LNSTensor
        The initial factor for the learning rate.
    end_factor : float | LNSTensor
        The final factor for the learning rate.
    total_iters : int, optional
        The number of iterations over which the learning rate will decay.
        Default: 0.
    last_epoch : int, optional
        The index of last epoch. Default: -1.
    """

    def __init__(
            self,
            optimizer: LNSOptimizer,
            start_factor: Union[float, LNSTensor] = 1.0 / 3,
            end_factor: Union[float, LNSTensor] = 1.0,
            total_iters: int = 0,
            last_epoch: int = -1,
        ):
        self.lns_lr_bases = get_lr_bases(optimizer)
        self.start_factor_lns = [_lns(start_factor, base) for base in self.lns_lr_bases]
        self.end_factor_lns = [_lns(end_factor, base) for base in self.lns_lr_bases]
        self.total_iters_lns = [_lns(total_iters, base) for base in self.lns_lr_bases]
        super().__init__(optimizer, start_factor, end_factor, total_iters, last_epoch)

    @override
    def get_lr(self) -> List[torch.Tensor]:
        torch.optim.lr_scheduler._warn_get_lr_called_within_step(self)

        if self.last_epoch == 0:
            return [
                lns_mul(group["lr"], start_factor, base)
                for group, start_factor, base in zip(self.optimizer.param_groups, self.start_factor_lns, self.lns_lr_bases)
            ]

        if self.last_epoch > self.total_iters:
            return [group["lr"] for group in self.optimizer.param_groups]

        return [
            lns_mul(group["lr"],
                    lns_add(LNS_ONE,
                            lns_div(
                                lns_sub(end_factor, start_factor, base),
                                lns_add(
                                    lns_mul(total_iters, start_factor, base),
                                    lns_mul(
                                        lns_sub(LNSTensor.get_internal_tensor(self.last_epoch, base),
                                                LNS_ONE, base),
                                        lns_sub(end_factor, start_factor, base),
                                        base), base), base), base), base)
            for group, start_factor, end_factor, total_iters, base in zip(
                self.optimizer.param_groups, self.start_factor_lns, self.end_factor_lns,
                self.total_iters_lns, self.lns_lr_bases)
        ]

    def _get_closed_form_lr(self) -> List[torch.Tensor]:
        return [
            lns_mul(
                base_lr,
                lns_add(
                    start_factor,
                    lns_div(
                        lns_mul(
                            lns_sub(end_factor, start_factor, base),
                            lns_minimum(
                                total_iters, LNSTensor.get_internal_tensor(self.last_epoch, base), base),
                            base), total_iters, base), base), base)
            for base_lr, start_factor, end_factor, total_iters, base in zip(
                self.base_lrs, self.start_factor_lns, self.end_factor_lns,
                self.total_iters_lns, self.lns_lr_bases
            )
        ]

class LNSExponentialLR(torch.optim.lr_scheduler.ExponentialLR):
    """
    An LNS learning rate scheduler that decays the learning rate of each parameter group
    by gamma each epoch.

    See also: :class:`torch.optim.lr_scheduler.ExponentialLR`

    Parameters
    ----------
    optimizer : LNSOptimizer
        Wrapped optimizer.
    gamma : float | LNSTensor
        Multiplicative factor of learning rate decay.
    last_epoch : int, optional
        The index of last epoch. Default: -1.
    """

    def __init__(
            self,
            optimizer: LNSOptimizer,
            gamma: Union[float, LNSTensor],
            last_epoch: int = -1
        ):
        self.lns_lr_bases = get_lr_bases(optimizer)
        self.gammas = [_lns(gamma, base) for base in self.lns_lr_bases]
        super().__init__(optimizer, gamma, last_epoch)

    @override
    def get_lr(self) -> List[torch.Tensor]:
        torch.optim.lr_scheduler._warn_get_lr_called_within_step(self)

        if self.last_epoch == 0:
            return [group["lr"] for group in self.optimizer.param_groups]

        return [
            lns_mul(group["lr"], gamma, base)
            for group, gamma, base in zip(self.optimizer.param_groups, self.gammas, self.lns_lr_bases)
        ]

    @override
    def _get_closed_form_lr(self) -> List[torch.Tensor]:
        return [
            lns_mul(base_lr, lns_pow(gamma, torch.tensor(self.last_epoch), base), base)
            for base_lr, gamma, base in zip(self.base_lrs, self.gammas, self.lns_lr_bases)
        ]

class LNSPolynomialLR(torch.optim.lr_scheduler.PolynomialLR):
    """
    An LNS learning rate scheduler that decays the learning rate of each parameter group
    by a polynomial factor in the given total_iters.

    See also: :class:`torch.optim.lr_scheduler.PolynomialLR`

    Parameters
    ----------
    optimizer : LNSOptimizer
        Wrapped optimizer.
    total_iters : int
        The number of iterations over which the learning rate will decay.
    power : float | LNSTensor
        The power of the polynomial decay.
    last_epoch : int, optional
        The index of last epoch. Default: -1.
    """

    def __init__(
            self,
            optimizer: LNSOptimizer,
            total_iters: int = 5,
            power: Union[float, LNSTensor] = 0.9,
            last_epoch: int = -1,
        ):
        self.lns_lr_bases = get_lr_bases(optimizer)
        self.total_iters_lns = [_lns(total_iters, base) for base in self.lns_lr_bases]
        power = power.value.item() if isinstance(power, LNSTensor) else power
        super().__init__(optimizer, total_iters, power, last_epoch)

    @override
    def get_lr(self) -> List[torch.Tensor]:
        torch.optim.lr_scheduler._warn_get_lr_called_within_step(self)

        if self.last_epoch == 0 or self.last_epoch > self.total_iters:
            return [group["lr"] for group in self.optimizer.param_groups]

        return [
            lns_mul(
                group["lr"],
                lns_pow(
                    lns_div(
                        lns_sub(LNS_ONE, lns_div(
                            LNSTensor.get_internal_tensor(self.last_epoch, base),
                            total_iters, base), base),
                        lns_sub(LNS_ONE, lns_div(
                            lns_sub(LNSTensor.get_internal_tensor(self.last_epoch, base),
                                    LNS_ONE, base), total_iters, base), base),
                    base), torch.tensor(self.power), base), base)
            for group, total_iters, base in zip(self.optimizer.param_groups, self.total_iters_lns, self.lns_lr_bases)
        ]

    @override
    def _get_closed_form_lr(self) -> List[torch.Tensor]:
        return [
            lns_mul(
                base_lr,
                lns_pow(
                    lns_sub(LNS_ONE,
                            lns_div(
                                lns_minimum(total_iters, LNSTensor.get_internal_tensor(self.last_epoch, base), base),
                                total_iters, base
                            ), base), torch.tensor(self.power), base), base)
            for base_lr, total_iters, base in zip(self.base_lrs, self.total_iters_lns, self.lns_lr_bases)
        ]

class LNSReduceLROnPlateau(torch.optim.lr_scheduler.ReduceLROnPlateau):
    """
    An LNS learning rate scheduler that reduces the learning rate of each parameter group
    when a metric has stopped improving.

    See also: :class:`torch.optim.lr_scheduler.ReduceLROnPlateau`

    Parameters
    ----------
    optimizer : LNSOptimizer
        Wrapped optimizer.
    mode : str, optional
        One of `min`, `max`. In `min` mode, the learning rate will be reduced when the quantity
        monitored has stopped decreasing; in `max` mode, it will be reduced when the quantity
        monitored has stopped increasing. Default: `min`.
    factor : float | LNSTensor, optional
        Factor by which the learning rate will be reduced. new_lr = lr * factor. Default: 0.1.
    patience : int, optional
        Number of epochs with no improvement after which learning rate will be reduced.
        Default: 10.
    threshold : float | LNSTensor, optional
        Threshold for measuring the new optimum, to only focus on significant changes.
        Default: 1e-4.
    threshold_mode : str, optional
        One of `rel`, `abs`. In `rel` mode, the threshold is a relative change;
        in `abs` mode, it is an absolute change. Default: `rel`.
    cooldown : int, optional
        Number of epochs to wait before resuming normal operation after lr has been reduced.
        Default: 0.
    min_lr : float | LNSTensor | List[float] | List[LNSTensor], optional
        A scalar or a list of scalars defining the lower bound on the learning rate
        of each parameter group. Default: 0.0.
    eps : float | LNSTensor, optional
        Minimal decay applied to lr. If the difference between new and old lr is smaller than eps,
        the update is ignored. Default: 1e-8.
    """

    def __init__(
            self,
            optimizer: LNSOptimizer,
            mode: str = "min",
            factor: Union[float, LNSTensor] = 0.1,
            patience: int = 10,
            threshold: Union[float, LNSTensor] = 1e-4,
            threshold_mode: str = "rel",
            cooldown: int = 0,
            min_lr: Union[float, LNSTensor, List[float], List[LNSTensor]] = 0.0,
            eps: Union[float, LNSTensor] = 1e-8,
        ):
        self.lns_lr_bases = get_lr_bases(optimizer)
        self.factor_lns = [_lns(factor, base) for base in self.lns_lr_bases]
        self.eps_lns = [_lns(eps, base) for base in self.lns_lr_bases]
        super().__init__(optimizer, mode, factor, patience, threshold, threshold_mode, cooldown, min_lr, eps)

        if isinstance(min_lr, (list, tuple)):
            self.min_lrs = [_lns(lr, base) for lr, base in zip(min_lr, self.lns_lr_bases)]
        else:
            self.min_lrs = [_lns(min_lr, base) for base in self.lns_lr_bases]

    def _reduce_lr(self, epoch):
        if len(self.optimizer.param_groups) != len(self.min_lrs):
            if self.default_min_lr is None:
                raise RuntimeError("The number of param groups in the optimizer must match the number of min_lrs.")
            else:
                self.min_lrs = [_lns(self.default_min_lr, base) for base in self.lns_lr_bases]

        for i, param_group in enumerate(self.optimizer.param_groups):
            base = param_group["base"]
            old_lr = param_group["lr"]
            new_lr = lns_maximum(
                lns_mul(old_lr, self.factor_lns[i], base),
                self.min_lrs[i], self.lns_lr_bases[i])
            if lns_gt(lns_sub(old_lr, new_lr, base), self.eps_lns[i]):
                param_group["lr"] = new_lr

class LNSChainedScheduler(torch.optim.lr_scheduler.ChainedScheduler):
    """
    A scheduler that chains multiple schedulers together.

    Note that this scheduler is a subclass of torch's `ChainedScheduler`,
    and is implemented for completeness. You can use the torch version
    directly with LNS optimizers.

    See also: :class:`torch.optim.lr_scheduler.ChainedScheduler`

    Parameters
    ----------
    schedulers : List[torch.optim.lr_scheduler.LRScheduler]
        List of schedulers to chain.
    """

    def __init__(
            self,
            schedulers: List[torch.optim.lr_scheduler.LRScheduler],
            optimizer: LNSOptimizer,
        ):
        super().__init__(schedulers, optimizer)

class LNSSequentialLR(torch.optim.lr_scheduler.SequentialLR):
    """
    A scheduler that applies a sequence of schedulers in order.

    Note that this scheduler is a subclass of torch's `SequentialLR`,
    and is implemented for completeness. You can use the torch version
    directly with LNS optimizers.

    See also: :class:`torch.optim.lr_scheduler.SequentialLR`

    Parameters
    ----------
    optimizer : LNSOptimizer
        Wrapped optimizer.
    schedulers : List[torch.optim.lr_scheduler.LRScheduler]
        List of schedulers to apply sequentially.
    milestones : List[int]
        List of epochs at which to switch to the next scheduler.
    last_epoch : int, optional
        The index of last epoch. Default: -1.
    """

    def __init__(
            self,
            optimizer: LNSOptimizer,
            schedulers: List[torch.optim.lr_scheduler.LRScheduler],
            milestones: List[int],
            last_epoch: int = -1,
        ):
        super().__init__(optimizer, schedulers, milestones, last_epoch)
from typing import Callable, List
from typing_extensions import override
import torch
from xlnstorch import LNSTensor, lnstensor
from . import LNSOptimizer
from xlnstorch.operators import (
    lns_mul,
)

def _lns(value: float | LNSTensor, base) -> LNSTensor:
    if isinstance(value, LNSTensor):
        return lnstensor(value, b=base)._lns
    return LNSTensor.get_internal_tensor(value, base)

class LNSLRScheduler(torch.optim.lr_scheduler.LRScheduler):
    """Base class for all LNS learning rate schedulers."""

    def __init__(
            self,
            optimizer: LNSOptimizer,
            last_epoch: int = -1,
        ):
        if not isinstance(optimizer, LNSOptimizer):
            raise TypeError(f"{type(optimizer).__name__} is not an LNSOptimizer")

        self.lns_lr_bases: list[torch.Tensor] = [
            group["base"] for group in optimizer.param_groups
        ]

        super().__init__(optimizer, last_epoch)

class LNSLambdaLR(torch.optim.lr_scheduler.LambdaLR, LNSLRScheduler):
    """
    An LNS learning rate scheduler that sets the learning rate of each parameter group
    to the initial learning rate multipled by a given function of the epoch.

    See also: :class:`torch.optim.lr_scheduler.LambdaLR`

    Parameters
    ----------
    optimizer : LNSOptimizer
        Wrapped optimizer.
    lr_scheduler : Callable[[int], float | LNSTensor] or List[Callable[[int], float | LNSTensor]]
        A function or a list of functions which computes a multiplicative factor given an integer parameter
        `epoch`, which is the index of the current epoch.
    last_epoch : int, optional
        The index of last epoch. Default: -1.
    """

    def __init__(
            self,
            optimizer: LNSOptimizer,
            lr_scheduler: Callable[[int], float | LNSTensor] | List[Callable[[int], float | LNSTensor]],
            last_epoch: int = -1,
        ):
        super().__init__(optimizer, lr_scheduler, last_epoch)

    @override
    def get_lr(self) -> List[torch.Tensor]:
        torch.optim.lr_scheduler._warn_get_lr_called_within_step(self)

        return [
            lns_mul(base_lr, _lns(lmbda(self.last_epoch), base), base)
            for lmbda, base_lr, base in zip(self.lr_lambdas, self.base_lrs, self.lns_lr_bases)
        ]
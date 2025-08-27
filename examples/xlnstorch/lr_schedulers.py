import argparse
from collections import defaultdict

import matplotlib.pyplot as plt
import xlnstorch as xlt
import xlnstorch.optim.lr_scheduler as lr_sched

def _get_lr(opt):
    """A helper function to convert the learning rate to float"""
    return xlt.lnstensor(opt.param_groups[0]["lr"], from_lns=True, b=opt.param_groups[0]["base"]).item()

def get_dummy_components(lr: float, total_steps: int):
    """Return (model, optimizer, schedulers_dict) ready to log LR."""

    # create a new identical optimizer for each scheduler
    model = xlt.nn.LNSLinear(10, 1)
    opt = lambda: xlt.optim.LNSSGD(model.lns_parameters(), lr=lr, momentum=0.9)

    schedulers = {
        "StepLR": lr_sched.LNSStepLR(opt(), step_size=30, gamma=0.5),
        "ExponentialLR": lr_sched.LNSExponentialLR(opt(), gamma=0.95),
        "PolynomialLR": lr_sched.LNSPolynomialLR(opt(), total_iters=20),
    }
    return schedulers


def log_lrs(epochs: int, base_lr: float = 1e-2):
    """Run through `epochs` fake training iterations and collect LRs."""
    sched_dict = get_dummy_components(base_lr, epochs)
    lr_history = defaultdict(list)

    for epoch in range(epochs):

        # Perform a single training step here (if needed)

        for name, sched in sched_dict.items():
            sched.optimizer.step() # to silence warnings
            sched.step()
            lr_history[name].append(_get_lr(sched.optimizer))

    return lr_history


def plot(lr_history):
    plt.figure(figsize=(10, 6))
    for name, lrs in lr_history.items():
        plt.plot(lrs, label=name)
    plt.xlabel("Epoch")
    plt.ylabel("Learning rate")
    plt.title("PyTorch LR scheduler comparison")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


parser = argparse.ArgumentParser(description="PyTorch LR Scheduler demo")
parser.add_argument("--epochs", type=int, default=100,
                    help="Number of epochs / scheduler steps (default: 100)")
parser.add_argument("--base_lr", type=float, default=1e-2,
                    help="Base learning rate for optimizers (default: 1e-2)")
args = parser.parse_args()

lr_hist = log_lrs(epochs=args.epochs, base_lr=args.base_lr)
plot(lr_hist)
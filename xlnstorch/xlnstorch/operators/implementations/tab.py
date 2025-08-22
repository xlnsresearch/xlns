"""
A torch port of the tab ufunc implementation for addition. See

https://github.com/xlnsresearch/xlns/blob/main/src/xlnsconf/tab_ufunc.py
"""
import os
import warnings
import torch
import numpy as np
import xlnstorch.csrc
import xlnstorch
from xlnstorch import implements_sbdb
from xlnstorch.operators.addition_ops import sbdb_ideal
from xlnstorch.tensor_utils import get_base_from_precision

# constants for the tab implementation
MAX_PREC = 23 # maximum precision for tables
initialized = False
tab_base = tab_ez = tab_sbdb = tab_mismatch = None

# tensor constants
_one = torch.tensor(1, dtype=torch.int64)
_zero = torch.tensor(0, dtype=torch.int64)

def get_table(filestem: str, f=None, b=None):
    if f is not None:
        base = get_base_from_precision(f)
    elif b is not None:
        if torch.is_tensor(b):
            base = b.detach().to(torch.float64)
        else:
            base = torch.tensor(b, dtype=torch.float64)

    global initialized, tab_base, tab_ez, tab_sbdb, tab_mismatch
    tab_mismatch = False
    tab_base = base

    filename = f"./{filestem}_{str(tab_base.item())[2:]}.npz"

    if os.path.isfile(filename):
        print(f"Loading table from {filename}")
        tablefile = np.load(filename)
        tab_ez = torch.tensor(tablefile['tab_ez'], dtype=torch.int64)
        tab_sbdb = torch.tensor(tablefile['tab_sbdb'], dtype=torch.int64)
        tablefile.close()
        initialized = True

        if xlnstorch.CSRC_AVAILABLE:
            xlnstorch.csrc.get_table(tab_ez, tab_sbdb, tab_base)

    elif tab_base >= get_base_from_precision(MAX_PREC):
        print(f"Creating ideal table as {filename}")
        tab_ez = sbdb_ideal(_one, _one, tab_base).to(torch.int64)
        zrange = torch.arange(tab_ez, 0)

        sbt = sbdb_ideal(zrange, _zero, tab_base).to(torch.int64)
        dbt = sbdb_ideal(zrange, _one, tab_base).to(torch.int64)
        tab_sbdb = torch.vstack((sbt, dbt))

        np.savez(filename, tab_ez=tab_ez.numpy(), tab_sbdb=tab_sbdb.numpy())
        initialized = True

        if xlnstorch.CSRC_AVAILABLE:
            xlnstorch.csrc.get_table(tab_ez, tab_sbdb, tab_base)

    else:
        warnings.warn(
            f"Table for base {tab_base.item()} is too"
            f"large to create. Max precision is {MAX_PREC}"
        )
        tab_base = None

@implements_sbdb("tab")
def sbdb_ufunc_tab(z, s, base):
    if not initialized:
        raise RuntimeError("Tab ufunc implementation not initialized. Call `make_table` first.")

    if base == tab_base:
        return tab_sbdb[s, torch.maximum(tab_ez, torch.where(z == 0, -1, z))]

    else:
        # only warn on the first mismatch
        global tab_mismatch
        if tab_mismatch == False:
            tab_mismatch = True
            warnings.warn(
                f"Table generated for base {tab_base.item()} but called with"
                f"base {base.item()}. Using ideal sbdb instead."
            )

        return sbdb_ideal(z, s, base)
from typing import Optional
import importlib

try:
    import xlnstorch._csrc as _C
except Exception:
    _C = None

def _ensure_built() -> None:
    if _C is None:
        raise ImportError(
            "xlnstorch C++ extension is not built. "
            "Install a prebuilt wheel or build from source with a C++17 compiler."
        )

def float_to_lns_forward(*args):
    _ensure_built()
    return _C.float_to_lns_forward(*args)

def float_to_lns_backward(*args):
    _ensure_built()
    return _C.float_to_lns_backward(*args)

def change_base_forward(*args):
    _ensure_built()
    return _C.change_base_forward(*args)

def change_base_backward(*args):
    _ensure_built()
    return _C.change_base_backward(*args)

def set_default_sbdb_implementation(*args):
    _ensure_built()
    return _C.set_default_sbdb_implementation(*args)

def get_table(*args):
    _ensure_built()
    return _C.get_table(*args)

def add_forward(*args):
    _ensure_built()
    return _C.add_forward(*args)

def sum_forward(*args):
    _ensure_built()
    return _C.sum_forward(*args)

def matmul_forward(*args):
    _ensure_built()
    return _C.matmul_forward(*args)

def matmul_backward(*args):
    _ensure_built()
    return _C.matmul_backward(*args)

def conv1d_forward(*args):
    _ensure_built()
    return _C.conv1d_forward(*args)

def conv1d_backward(*args):
    _ensure_built()
    return _C.conv1d_backward(*args)

def conv2d_forward(*args):
    _ensure_built()
    return _C.conv2d_forward(*args)

def conv2d_backward(*args):
    _ensure_built()
    return _C.conv2d_backward(*args)

def conv3d_forward(*args):
    _ensure_built()
    return _C.conv3d_forward(*args)

def conv3d_backward(*args):
    _ensure_built()
    return _C.conv3d_backward(*args)

__all__ = [
    "float_to_lns_forward",
    "float_to_lns_backward",
    "change_base_forward",
    "change_base_backward",
    "set_default_sbdb_implementation",
    "get_table",
    "add_forward",
    "sum_forward",
    "matmul_forward",
    "matmul_backward",
    "conv1d_forward",
    "conv1d_backward",
    "conv2d_forward",
    "conv2d_backward",
    "conv3d_forward",
    "conv3d_backward"
]
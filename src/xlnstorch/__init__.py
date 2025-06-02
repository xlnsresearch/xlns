

try:
    import torch
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "xlnstorch requires PyTorch but it is not installed.\n"
        "See https://pytorch.org/get-started/locally for instructions."
    ) from e
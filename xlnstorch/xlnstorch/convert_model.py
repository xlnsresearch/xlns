from collections import OrderedDict
import inspect
from typing import Union
import torch.nn as nn
import xlnstorch.nn as xlns_nn

__all__ = [
    "parse_sequential",
    "build_lns_sequential",
]

def _parameter_to_bool(x):
    # bias=True  -> layer.bias is a torch.nn.Parameter
    # bias=False -> layer.bias is None
    return None if x is None else True

SPECIAL_HANDLING = {
    nn.Linear: (
        "in_features",
        "out_features",
        ("bias", _parameter_to_bool),
    ),
    nn.Bilinear: (
        "in1_features",
        "in2_features",
        "out_features",
        ("bias", _parameter_to_bool),
    ),
    nn.LazyLinear: (
        "out_features",
        ("bias", _parameter_to_bool),
    ),
    nn.Conv1d: (
        "in_channels",
        "out_channels",
        "kernel_size",
        "stride",
        "padding",
        "dilation",
        "groups",
        ("bias", _parameter_to_bool),
        "padding_mode",
    ),
    nn.Conv2d: (
        "in_channels",
        "out_channels",
        "kernel_size",
        "stride",
        "padding",
        "dilation",
        "groups",
        ("bias", _parameter_to_bool),
        "padding_mode",
    ),
    nn.Conv3d: (
        "in_channels",
        "out_channels",
        "kernel_size",
        "stride",
        "padding",
        "dilation",
        "groups",
        ("bias", _parameter_to_bool),
        "padding_mode",
    ),
    nn.RNN: (
        "input_size",
        "hidden_size",
        "num_layers",
        "nonlinearity",
        "bias",
        "batch_first",
        "dropout",
        "bidirectional",
    ),
    nn.RNNCell: (
        "input_size",
        "hidden_size",
        "bias",
        "nonlinearity",
    ),
    nn.LSTM: (
        "input_size",
        "hidden_size",
        "num_layers",
        "bias",
        "batch_first",
        "dropout",
        "bidirectional",
        "proj_size",
    ),
    nn.LSTMCell: (
        "input_size",
        "hidden_size",
        "bias",
    ),
    nn.GRU: (
        "input_size",
        "hidden_size",
        "num_layers",
        "bias",
        "batch_first",
        "dropout",
        "bidirectional",
    ),
    nn.GRUCell: (
        "input_size",
        "hidden_size",
        "bias",
    ),
    nn.MultiheadAttention: (
        "embed_dim",
        "num_heads",
        "dropout",
        "bias",
        "add_bias_kv",
        "add_zero_attn",
        "kdim",
        "vdim",
        "batch_first",
    ),
    nn.Dropout: (
        "p",
        "inplace",
    ),
    nn.Dropout1d: (
        "p",
        "inplace",
    ),
    nn.Dropout2d: (
        "p",
        "inplace",
    ),
    nn.Dropout3d: (
        "p",
        "inplace",
    ),
    nn.AvgPool1d: (
        "kernel_size",
        "stride",
        "padding",
        "ceil_mode",
        "count_include_pad",
        "divisor_override",
    ),
    nn.AvgPool2d: (
        "kernel_size",
        "stride",
        "padding",
        "ceil_mode",
        "count_include_pad",
        "divisor_override",
    ),
    nn.AvgPool3d: (
        "kernel_size",
        "stride",
        "padding",
        "ceil_mode",
        "count_include_pad",
        "divisor_override",
    ),
    nn.AdaptiveAvgPool1d: (
        "output_size",
    ),
    nn.AdaptiveAvgPool2d: (
        "output_size",
    ),
    nn.AdaptiveAvgPool3d: (
        "output_size",
    ),
    nn.MaxPool1d: (
        "kernel_size",
        "stride",
        "padding",
        "dilation",
        "return_indices",
        "ceil_mode",
    ),
    nn.MaxPool2d: (
        "kernel_size",
        "stride",
        "padding",
        "dilation",
        "return_indices",
        "ceil_mode",
    ),
    nn.MaxPool3d: (
        "kernel_size",
        "stride",
        "padding",
        "dilation",
        "return_indices",
        "ceil_mode",
    ),
    nn.BatchNorm1d: (
        "num_features",
        "eps",
        "momentum",
        "affine",
        "track_running_stats",
    ),
    nn.BatchNorm2d: (
        "num_features",
        "eps",
        "momentum",
        "affine",
        "track_running_stats",
    ),
    nn.BatchNorm3d: (
        "num_features",
        "eps",
        "momentum",
        "affine",
        "track_running_stats",
    ),
    nn.LayerNorm: (
        "normalized_shape",
        "eps",
        "elementwise_affine",
        ("bias", _parameter_to_bool)
    ),
    nn.ReLU: (
        "inplace",
    ),
}

def parse_sequential(model, parent_name=""):
    """
    Parses a PyTorch nn.Sequential model or an equivalent nn.Module that
    defines nn.Module children in a similar way.

    Parameters
    ----------
    model : Union[torch.nn.Sequential, torch.nn.Module]
        The model to parse, typically an instance of nn.Sequential or a custom
        nn.Module that contains nn.Module children.
    parent_name : str, optional
        The name of the parent module, used for nested modules. Defaults to "".

    Returns
    -------
    list of tuples
        Each tuple contains:
        - name: str, the full name of the module
        - cls: str, the class name of the module
        - args: OrderedDict, the parameters of the module
        This allows for easy introspection of the model structure and parameters.
    """
    result = []

    for name, module in model._modules.items():
        full_name = f"{parent_name}.{name}" if parent_name else name
        cls_type  = type(module)
        cls_name  = cls_type.__name__

        # special case handling
        if cls_type in SPECIAL_HANDLING:
            argspec = OrderedDict()

            for item in SPECIAL_HANDLING[cls_type]:
                if isinstance(item, tuple): # needs post-processing
                    attr, func = item
                    argspec[attr] = func(getattr(module, attr))
                else:
                    argspec[item] = getattr(module, item)

            result.append((full_name, cls_name, argspec))

        # generic fallback
        else:
            sig = inspect.signature(module.__init__)
            argspec = OrderedDict()

            for p in sig.parameters.values():
                if p.name == "self":
                    continue
                if hasattr(module, p.name):
                    argspec[p.name] = getattr(module, p.name)

            result.append((full_name, cls_name, argspec))

        if isinstance(module, nn.Sequential):
            result.extend(parse_sequential(module, full_name))

    return result

def _resolve_class(class_name: str):
    """
    Resolves a class name to the actual class object. This
    function checks both `xlnstorch.nn` and `torch.nn`
    namespaces to find the class.

    Parameters
    ----------
    class_name : str
        The name of the class to resolve.

    Returns
    -------
    type
        The resolved class object.
    """
    if hasattr(xlns_nn, "LNS" + class_name):
        return getattr(xlns_nn, "LNS" + class_name)

    if hasattr(nn, class_name):
        return getattr(nn, class_name)

    raise RuntimeError(
        f"Could not find a layer called '{class_name}' in either "
        "xlnstorch.nn or torch.nn."
    )

def build_lns_sequential(model, *, keep_containers=False):
    """
    Create a new `xlnstorch.nn.LNSSequential` object where each layer is
    instantiated from `xlnstorch.nn` if available, otherwise from `torch.nn`.

    Parameters
    ----------
    model : torch.nn.Sequential | torch.nn.Module
        The original model to replicate.
    keep_containers : bool, optional
        If True, the function will also replicate the explicit
        Sequential containers that appear in `parse_sequential`
        (they have no arguments).  If False (default) those entries are
        ignored and a flat ordering of the leaf layers is returned.

    Returns
    -------
    xlnstorch.nn.LNSSequential
        A new `xlnstorch.nn.LNSSequential` object containing the layers from the original model.

    Raises
    ------
    RuntimeError
        If a layer in the original model is not an instance of `xlnstorch.nn.LNSModule`
        but has parameters, indicating that it cannot be converted to LNS.
    """
    new_layers = []

    for _, cls_name, arg_dict in parse_sequential(model):

        if cls_name == "Sequential" and not keep_containers:
            continue

        cls = _resolve_class(cls_name)
        try:
            layer = cls(**arg_dict)
        except Exception as e:
            # Check if this is an LNS layer or regular torch layer to provide better error message
            if hasattr(xlns_nn, "LNS" + cls_name):
                raise RuntimeError(
                    f"Failed to instantiate LNS layer '{cls_name}' with arguments {arg_dict}. "
                    f"This suggests an issue with the LNS implementation or argument compatibility. "
                    f"Original error: {type(e).__name__}: {e}"
                ) from e
            else:
                raise RuntimeError(
                    f"Failed to instantiate torch.nn layer '{cls_name}' with arguments {arg_dict}. "
                    f"This layer may not be supported in LNS conversion or has incompatible arguments. "
                    f"Original error: {type(e).__name__}: {e}"
                ) from e

        # torch modules with parameters are incompatible with LNS
        # here we use a sentinel to check this
        sentinel = object()
        if not (isinstance(layer, xlns_nn.LNSModule) or next(layer.parameters(), sentinel) is sentinel):
            raise RuntimeError(
                f"Layer '{cls_name}' is not an instance of xlnstorch.nn.LNSModule, "
                "but it has parameters. This is not supported."
            )

        new_layers.append(layer)

    return xlns_nn.LNSSequential(*new_layers)
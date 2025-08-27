from collections import OrderedDict
from typing import Any, Dict, Iterable, List, Optional, Set, Union
import torch
from xlnstorch import LNSTensor, lnstensor

# This module is heavily based on pytorchviz. I have adapted it to
# work with LNSTensor objects and to only import graphviz when
# this function is called (so that it is not a hard dependency)
# among other things.
# 
# Original source:
# https://github.com/szagoruyko/pytorchviz

__all__ = [
    "make_autograd_graph"
]

default_node_attr = {
    "style": "filled",
    "shape": "box",
    "align": "left",
    "fontsize": "9",
    "fontname": "monospace",
    "ranksep": "0.1",
    "height": "0.2",
}
default_edge_attr = {
    "arrowhead": "vee",
}

def _get_size(tensor: torch.Tensor) -> str:
    if tensor.dim() == 0:
        return "(scalar)"
    return "(" + ",".join(map(str, tensor.size())) + ")"

def _get_label(tensor: torch.Tensor, names: Dict[int, str]) -> str:
    label = names.get(id(tensor), "")
    if label:
        label += "\n"
    return label + _get_size(tensor)

def _make_node_table(fn: torch.autograd.Function, show_saved: bool) -> str:
    """
    Return a string representation of the saved attributes in the function.
    If no saved attributes are present, return an empty string.
    """
    table_str = type(fn).__name__

    if not show_saved:
        return table_str

    attrs = []
    names_len = 0
    values_len = 0

    # for standard pytorch functions, saved tensors are stored as attributes
    # with the prefix "_saved_".
    for attr in dir(fn):
        if attr.startswith("_saved_"):

            name = attr[7:]  # remove the "_saved_" prefix
            value = getattr(fn, attr)

            if isinstance(value, torch.Tensor):
                value = _get_size(value)
            else:
                value = repr(value)

            attrs.append((name, value))
            names_len = max(names_len, len(name))
            values_len = max(values_len, len(value))

    # for custom functions, tensors are saved in the `saved_tensors` attribute
    # and other variables in the `__dict__` attribute.
    if hasattr(fn, 'saved_tensors'):
        for tensor in fn.saved_tensors:

            name = "saved_tensor"
            value = _get_size(tensor)

            attrs.append((name, value))
            names_len = max(names_len, len(name))
            values_len = max(values_len, len(value))

    if hasattr(fn, '__dict__'):
        for name, value in fn.__dict__.items():
            value = repr(value)
            attrs.append((name, value))
            names_len = max(names_len, len(name))
            values_len = max(values_len, len(value))

    # if no attributes are found, return the function name
    if len(attrs) == 0:
        return table_str

    max_len = max(len(table_str), names_len + values_len + 3)
    table_str += "\n" + "-" * max_len + "\n"
    table_str += "\n".join(
        name.ljust(names_len) + " : " + value.rjust(max_len - names_len - 3)
        for name, value in attrs
    )

    return table_str

def _check_import_graphviz():
    """
    Check if the 'graphviz' package is installed and available.
    If not, raise an ImportError with a helpful message.
    """
    try:
        import graphviz
    except ImportError as exc:
        raise ImportError(
            "You asked for an autograd graph but the 'graphviz' package is not "
            "installed. Install it with `pip install graphviz` (and make sure the "
            "graphviz system libraries are available)."
        ) from exc

def flatten(items: Iterable[Any]) -> List[Any]:
    """
    Flatten a nested iterable (list or tuple) into a single list.
    """
    flat: List[Any] = []
    for it in items:
        if isinstance(it, (list, tuple)):
            flat.extend(flatten(it))
        else:
            flat.append(it)
    return flat

def is_tensor_like(obj: Any) -> bool:
    "True for torch.Tensor or a wrapper exposing a ._lns torch.Tensor."
    return torch.is_tensor(obj) or isinstance(obj, LNSTensor)

def unwrap(obj: Union[torch.Tensor, LNSTensor]) -> torch.Tensor:
    """
    Unwrap an LNSTensor object to its underlying torch.Tensor.
    """
    if isinstance(obj, LNSTensor):
        return obj._lns
    return obj

def make_autograd_graph(
        *vars: Union[torch.Tensor, LNSTensor],
        graph_name: str = "Autograd Graph",
        show_saved: bool = False,
        leaf_color: str = "orange",
        node_color: str = "lightgrey",
        output_color: str = "yellow",
        params: Optional[Dict[str, Union[torch.Tensor, LNSTensor]]] = None,
        node_attr: Optional[Dict[str, str]] = None,
        edge_attr: Optional[Dict[str, str]] = None,
    ):
    """
    Build (and return) a `graphviz.Digraph` object that visualizes
    the PyTorch autograd graph for the given variables.

    Parameters
    ----------
    vars : torch.Tensor | LNSTensor
        The output variables for which to build the autograd graph.
        Typically just the one output of the loss function.
    graph_name : str, optional
        The name of the graph to be displayed in the visualization.
        Defaults to "Autograd Graph".
    show_saved : bool, optional
        If `True`, saved tensors and variables will be shown in the
        graph (inside their respective function nodes).
    leaf_color : str, optional
        The color to use for leaf nodes (i.e., tensors that are not
        outputs of any autograd function). Defaults to "orange".
    node_color : str, optional
        The color to use for function nodes in the graph. Defaults
        to "lightgrey".
    output_color : str, optional
        The color to use for the final output tensor node in the
        graph. Defaults to "yellow".
    params : Dict[str, torch.Tensor | LNSTensor], optional
        An optional mapping ``parameter_name -> value`` where *value*
        is either a ``torch.Tensor`` or an ``LNSTensor`` instance.
        If supplied, the corresponding nodes will be highlighted and
        annotated with the user-provided name, which makes it much
        easier to see where model parameters occur in the graph.
    node_attr : Dict[str, str], optional
        Additional attributes to apply to all nodes in the graph.
        This can be used to set styles, colors, or other properties
        that should be consistent across all nodes.
    edge_attr : Dict[str, str], optional
        Additional attributes to apply to all edges in the graph.
        This can be used to set styles, colors, or other properties
        that should be consistent across all edges.

    Returns
    -------
    graphviz.Digraph
        A directed graph object representing the autograd graph.
        This can be rendered using `graphviz.render` or similar methods.    
    """

    # 1. only import graphviz if we actually need it (i.e. if this function is called)
    try:
        import graphviz
    except ImportError as exc:
        raise ImportError(
            "You asked for an autograd graph but the 'graphviz' package is not "
            "installed. Install it with `pip install graphviz` (and make sure the "
            "graphviz system libraries are available)."
        ) from exc

    # 2. collect and flatten all root variables
    roots_raw: List[Any] = flatten(vars)
    if not roots_raw:
        raise ValueError("No variables passed to make_autograd_graph().")

    roots = []
    for root in roots_raw:
        if not is_tensor_like(root):
            raise TypeError(
                f"Expected a torch.Tensor or LNSTensor, got {type(root).__name__}."
            )
        roots.append(unwrap(root))

    # 3. create param node lookup table
    param_id_to_name: Dict[int, str] = {}
    if params is not None:
        if not isinstance(params, dict):
            raise TypeError("`params` must be a dict mapping names to tensors.")
        for name, value in params.items():
            if not is_tensor_like(value):
                raise TypeError(
                    f"Parameter '{name}' is not a tensor / LNSTensor (got {type(value)})"
                )
            if name.endswith("_lns"):
                name = name[:-4]  # remove the "_lns" suffix if present
            param_id_to_name[id(unwrap(value))] = name

    # 4. initialize node and edge attributes
    node_attributes = default_node_attr.copy()
    if node_attr:
        node_attributes.update(node_attr)

    edge_attributes = default_edge_attr.copy()
    if edge_attr:
        edge_attributes.update(edge_attr)

    # 5. construct the graph
    dot = graphviz.Digraph(graph_name, node_attr=node_attributes, edge_attr=edge_attributes)
    seen: Set[int] = set()

    def add_node(obj: Any) -> None:
        """
        Recursively add nodes + edges starting from *obj*.
        """
        obj_id = id(obj)
        if obj_id in seen:
            return
        seen.add(obj_id)

        # distinguish between leaf tensors, saved tensors, and function nodes
        if torch.is_tensor(obj):
            # plain tensor (leaf or saved value)
            dot.node(str(obj_id), label=_get_label(obj, param_id_to_name), fillcolor=leaf_color)

        elif isinstance(obj, torch._C._functions.AccumulateGrad):
            # GradAccumulate node - skip it and add the tensor it accumulates to the graph
            tensor = obj.variable
            dot.node(str(obj_id), label=_get_label(tensor, param_id_to_name), fillcolor=leaf_color)

        else:
            # autograd function node
            label = _make_node_table(obj, show_saved)
            dot.node(str(obj_id), label, fillcolor=node_color)

        if hasattr(obj, "next_functions"):
            for next_obj, _ in obj.next_functions:
                if next_obj is not None:
                    dot.edge(str(id(next_obj)), str(obj_id))
                    add_node(next_obj)

    # add all root nodes
    for root in roots:
        if root.grad_fn is not None:
            add_node(root.grad_fn)
            # Also show the final output tensor as a small extra node
            dot.node(str(id(root)), _get_label(root, param_id_to_name), fillcolor=output_color)
            dot.edge(str(id(root.grad_fn)), str(id(root)))
        else:
            add_node(root)

    return dot
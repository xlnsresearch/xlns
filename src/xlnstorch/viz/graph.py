import warnings
from typing import Any, Dict, Iterable, List, Set
import torch
from .. import LNSTensor, lnstensor

__all__ = [
    "make_autograd_graph"
]

default_node_attr = {
    "style": "filled",
    "shape": "box",
    "align": "left",
    "fontsize": "9",
    "ranksep": "0.1",
    "height": "0.2",
}
default_edge_attr = {
    "arrowhead": "vee",
}

def _get_size(tensor: torch.Tensor) -> str:
    return ",".join(map(str, tensor.size()))

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

def unwrap(obj: torch.Tensor | LNSTensor) -> torch.Tensor:
    """
    Unwrap an LNSTensor object to its underlying torch.Tensor.
    """
    if isinstance(obj, LNSTensor):
        return obj._lns
    return obj

def make_autograd_graph(
        *vars: torch.Tensor | LNSTensor,
        params: Dict[str, torch.Tensor | LNSTensor] | None = None,
        graph_name: str = "Autograd Graph",
        node_attr: Dict[str, str] | None = None,
        edge_attr: Dict[str, str] | None = None,
    ):
    """
    Build (and return) a `graphviz.Digraph` object that vizualizes
    the PyTorch autograd graph for the given variables.

    Parameters
    ----------
    vars : torch.Tensor | LNSTensor
        The output variables for which to build the autograd graph.
        Typically just the one output of the loss function.
    params : Dict[str, torch.Tensor | LNSTensor], optional
        An optional mapping ``parameter_name -> value`` where *value*
        is either a ``torch.Tensor`` or an ``LNSTensor`` instance.
        If supplied, the corresponding nodes will be highlighted and
        annotated with the user-provided name, which makes it much
        easier to see where model parameters occur in the graph.
    graph_name : str, optional
        The name of the graph to be displayed in the visualization.
        Defaults to "Autograd Graph".
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
            param_id_to_name[id(unwrap(value))] = name

    # 4. construct the graph
    if node_attr:
        node_attr = default_node_attr.copy().update(node_attr)
    else:
        node_attr = default_node_attr.copy()

    if edge_attr:
        edge_attr = default_edge_attr.copy().update(edge_attr)
    else:
        edge_attr = default_edge_attr.copy()

    dot = graphviz.Digraph(graph_name, node_attr=default_node_attr, edge_attr=default_edge_attr)
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
            tensor: torch.Tensor = obj
            name = param_id_to_name.get(obj_id, "")
            if name:
                name += "\n"
            label = f"{name}({_get_size(tensor)})"
            fillcolor = "lightblue" if name else "orange"
            dot.node(str(obj_id), label=label, fillcolor=fillcolor)

        elif isinstance(obj, torch._C._functions.AccumulateGrad):
            # GradAccumulate node
            tensor = obj.variable
            tensor_id = id(tensor)
            name = param_id_to_name.get(tensor_id, "")
            if name:
                name += "\n"
            label = f"{name}({_get_size(tensor)})"
            dot.node(str(obj_id), label=label, fillcolor="brown")

        else:
            # autograd function node
            dot.node(str(obj_id), type(obj).__name__, fillcolor="lightgrey")

        if hasattr(obj, "next_functions"):
            for next_obj, _ in obj.next_functions:
                if next_obj is not None:
                    dot.edge(str(id(next_obj)), str(obj_id))
                    add_node(next_obj)

        # if hasattr(obj, "saved_tensors"):
        #     for t in obj.saved_tensors:
        #         dot.edge(str(id(t)), str(obj_id))
        #         add_node(t)

    # add all root nodes
    for root in roots:
        if root.grad_fn is not None:
            add_node(root.grad_fn)
            # Also show the final output tensor as a small extra node
            name = param_id_to_name.get(id(root), "")
            if name:
                name += "\n"
            label = f"{name}({_get_size(root)})"
            dot.node(str(id(root)), label, fillcolor="yellow")
            dot.edge(str(id(root.grad_fn)), str(id(root)))
        else:
            add_node(root)

    return dot
import torch
from collections import deque
from typing import List, Dict, Iterable, Set, Any, TYPE_CHECKING, Union

if TYPE_CHECKING:
    from xlnstorch.tensor import LNSTensor

# Lazy import cache to avoid repeated imports
_tensor_module = None

__all__ = [
    "LNSFunction",
    "has_fanout",
    "find_fanout",
    "raise_fanout_error"
]

def _get_tensor_module():
    """Lazy import of tensor module to avoid circular imports."""
    global _tensor_module
    if _tensor_module is None:
        from . import tensor
        _tensor_module = tensor
    return _tensor_module

class LNSFunction(torch.autograd.Function):
    """
    Base class for LNS operations that require custom forward and backward methods.
    This class should be subclassed for specific LNS operations.
    """

    @staticmethod
    def forward(ctx, *args, **kwargs):
        """
        Forward pass for the LNS operation.
        Should be implemented in subclasses.
        """
        raise NotImplementedError("Forward method must be implemented in subclasses.")

    @staticmethod
    def backward(ctx, *grad_outputs):
        """
        Backward pass for the LNS operation.
        Should be implemented in subclasses.
        """
        raise NotImplementedError("Backward method must be implemented in subclasses.")

    @classmethod
    def apply(cls, *args, **kwargs):
        """
        Applies the LNS operation defined by this class.
        This method is used to call the forward and backward methods.

        Note that any keyword arguments passed to this method will raise an error,
        as `torch.autograd.Function` does not support keyword arguments. Instead,
        use positional arguments only.

        In addition, this method converts any `LNSTensor` arguments to their
        internal representation (i.e., the underlying tensor) before calling the
        forward method. This is necessary for LNSTensor internal behavior.
        """
        tensor_module = _get_tensor_module()

        # This check is also performed in the base class, but we do it here too
        # in case PyTorch decides to change the behavior of the apply method.
        if kwargs:
            raise ValueError("torch.autograd.Function does not support keyword arguments. Please use positional arguments only.")

        # Convert LNSTensor arguments to internal representation. This is necessary
        # because the autograd.Function expects tensors, not LNSTensor objects.
        internal_args = []
        for arg in args:
            if isinstance(arg, tensor_module.LNSTensor):
                internal_args.append(arg._lns)
            else:
                internal_args.append(arg)

        # call the forward method of the class with the internal arguments
        result = super().apply(*internal_args)

        # get all output tensors and store them in a tuple
        if isinstance(result, torch.Tensor):
            result_tuple = (result,)
        elif isinstance(result, (tuple, list)):
            result_tuple = tuple(result)
        else:
            result_tuple = tuple()

        # we register hooks on each input to each output for gradient accumulation
        for output in result_tuple:

            # only register hooks for tensors outputs that require gradients
            if not (isinstance(output, torch.Tensor) and output.requires_grad):
                continue

            # get the gradient edge for the output tensor.
            edge = torch.autograd.graph.get_gradient_edge(output)

            j = 0
            for i in range(len(args)):

                if not isinstance(args[i], (torch.Tensor, tensor_module.LNSTensor)):
                    continue
                j += 1

                # only register hooks for LNSTensor inputs with gradients
                if not (isinstance(args[i], tensor_module.LNSTensor) and args[i].requires_grad):
                    continue

                # track operation by registering a hook on the input tensor
                # to use custom addition logic each time it receives a gradient
                args[i]._track_operation(edge, j - 1)

            # experiment with breaking after first output
            # otherwise hooks are duplicated unnecessarily
            break

        return result

# This file contains functions to analyze the autograd graph in PyTorch.
# In particular, it can detect nodes with fan-out, i.e., nodes that have
# multiple parents. This was necessary since it broke the previous LNS
# autograd implementation but is not a problem for the new one. The functions
# aren't used in the library itself anymore but are kept here for reference.

def _children(fn: torch.autograd.Function) -> List[torch.autograd.Function]:
    """Returns a list of the function nodes reachable from `fn`."""
    if fn is None:
        return []
    return [n for n, _ in fn.next_functions if n is not None]

def has_fanout(root: Union[torch.Tensor, torch.autograd.Function]) -> bool:
    """
    Determines if the autograd graph starting from `root` has any fan-out nodes.

    Parameters
    ----------
    root : Union[torch.Tensor, torch.autograd.Function]
        A tensor whose `.grad_fn` is used as the graph root,
        or a `Function` node itself.

    Returns
    -------
    bool
        True if there are nodes in the graph that have more than one parent,
        False otherwise.
    """

    # make sure we start from a function node
    start = root.grad_fn if isinstance(root, torch.Tensor) else root
    if start is None:
        return [] # leaf tensor -> empty graph above it

    visited: Set[int] = set() # ids of nodes already expanded
    q: deque = deque([start])
    edge_counts: Dict[int, int] = {} # child_id -> number of incoming edges

    # breadth-first search through the graph
    while q:
        
        parent = q.popleft()
        parent_id = id(parent)

        # skip nodes we've already expanded
        if parent_id in visited:
            continue
        visited.add(parent_id)

        # detect intra-parent fan-out quickly:
        #   if the same child appears twice in parent.next_functions,
        #   we can return True immediately without touching any dict.
        children = _children(parent)
        child_ids = [id(child) for child in children]
        if len(child_ids) != len(set(child_ids)): # duplicate found
            return True
        
        # normal per-edge bookkeeping
        for child in children:
            child_id = id(child)

            edge_counts[child_id] = edge_counts.get(child_id, 0) + 1
            if edge_counts[child_id] > 1: # another parent already pointed to this child
                return True
            q.append(child)

    # traversal finished without finding any fan-out nodes
    return False

def find_fanout(root: Any) -> List[Dict[str, Any]]:
    """
    Detect every node in the autograd graph starting from `root`
    that has multiple incoming edges (i.e., multiple parents and
    therefore fan-out).
    
    Parameters
    ----------
    root : Union[torch.Tensor, torch.autograd.Function]
        A tensor whose `.grad_fn` is used as the graph root,
        or a `Function` node itself.

    Returns
    -------
    List[Dict[str, Any]]
        A list of dictionaries, each containing:
        - 'child': the child node with fan-out,
        - 'edge_count': the number of incoming edges to this child,
        - 'parents': a set of parent nodes that reference this child.
        If no fan-out nodes are found, an empty list is returned.
    """

    # make sure we start from a function node
    start = root.grad_fn if isinstance(root, torch.Tensor) else root
    if start is None:
        return [] # leaf tensor -> empty graph above it

    visited: Set[int] = set() # ids of nodes already expanded
    q: deque = deque([start])
    child_info: Dict[int, Dict[str, Any]] = {} # child_id -> info dict (will be filled on the fly)

    # breadth-first search through the graph
    while q:

        parent = q.popleft()
        parent_id = id(parent)

        # skip nodes we've already expanded
        if parent_id in visited:
            continue
        visited.add(parent_id)

        # explore all non-None children reachable from this parent
        for child in _children(parent):
            child_id = id(child)

            # lazily create the bookkeeping entry for this child
            if child_id not in child_info:
                child_info[child_id] = {
                    'child': child,
                    'edge_count': 0, # incremented below
                    'parents': set() # distinct parent nodes
                }

            info = child_info[child_id]
            info['edge_count'] += 1
            info['parents'].add(parent)

            # put the child into the queue so we can visit its children later
            q.append(child)

    # filter out children with only one edge (nodes may have one parent but multiple edges)
    return [info for info in child_info.values() if info['edge_count'] > 1]

def _obj_name(obj: Any) -> str:
    """Returns the class name of the object or "None" if the object is None."""
    return obj.__class__.__name__ if obj else "None"

def _node_repr(obj):
    """Returns a compact string representation of the object."""
    return f"{_obj_name(obj)}@{hex(id(obj))}"

def raise_fanout_error(offenders: Iterable[Dict[str, Any]]) -> None:
    """
    Turn the `offenders` list returned by `find_fanout` into a *human-readable*
    error message and raise a `RuntimeError`.

    Parameters
    ----------
    offenders : Iterable[Dict[str, Any]]
        Each dict must have exactly the keys inserted by `find_fanout`:
        - 'child' : the Function node with >1 incoming edges
        - 'edge_count' : total number of incoming edges (int)
        - 'parents' : *set* of distinct parent nodes

    Raises
    ------
    RuntimeError
        If the `offenders` list is empty, this function does nothing. If
        the list is not empty, it raises a `RuntimeError` with a detailed message
        about each offender node and its parents.
    """

    if not offenders:
        return

    error_message: list[str] = ["Fan-out in autograd graph is not allowed, found:"]

    for info in offenders:
        # Each `info` dict must have the keys inserted by `find_fanout`
        child = _node_repr(info['child'])
        parents_repr = ", ".join(_node_repr(p) for p in info['parents'])
        edge_count = info['edge_count']

        if len(info['parents']) == 1:
            # intra-parent fan-out: a single parent references its child >1 times
            parent_repr = _node_repr(next(iter(info['parents'])))
            error_message.append(
                f"{child} is referenced {edge_count} times by its single parent:\n  {parent_repr}"
            )

        else:
            # inter-parent fan-out: several parents share the same child
            error_message.append(
                f"{child} is shared by {len(info['parents'])} parents:\n  total incoming edges: {edge_count}.\n  {parents_repr}"
            )

    raise RuntimeError("\n".join(error_message))
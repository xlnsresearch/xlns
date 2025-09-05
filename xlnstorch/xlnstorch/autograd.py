import torch
import functools 
from collections import deque
import inspect
from typing import List, Dict, Iterable, Set, Any, TYPE_CHECKING, Union, Callable, Optional, Tuple

if TYPE_CHECKING:
    from xlnstorch.tensor import LNSTensor

# Lazy import cache to avoid repeated imports
_tensor_module = None
_ops_module = None

__all__ = [
    "LNSFunction",
    "LNSNonDifferentiableFunction",
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

def _get_ops_module():
    """Lazy import of ops module to avoid circular imports."""
    global _ops_module
    if _ops_module is None:
        from . import ops
        _ops_module = ops
    return _ops_module

def _cast_int64(x):
    return x.view(torch.int64) if isinstance(x, torch.Tensor) and x.dtype == torch.float64 else x

def _cast_float64(x):
    return x.view(torch.float64) if isinstance(x, torch.Tensor) and x.dtype == torch.int64 else x

def _cast_values(values, indices, cast_fn):

    all_indices = indices is None

    if isinstance(values, tuple):
        return tuple(
            cast_fn(values[i]) if all_indices or i in indices else values[i]
            for i in range(len(values))
        )

    elif isinstance(values, list):
        return [
            cast_fn(values[i]) if all_indices or i in indices else values[i]
            for i in range(len(values))
        ]

    else:
        return cast_fn(values) if all_indices or 0 in indices else values

# added to forward only if setup_context is not defined
def forward_ctx_decorator(func, cls):

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        ctx = args[0]
        lns_ops = args[1]
        inputs = args[2:]

        # add lns_ops to ctx for later use in backward
        ctx._lns_ops = lns_ops

        # store the indices of LNSTensor inputs on the ctx for backward
        input_indices = cls._lnstensor_inputs
        ctx._lnstensor_inputs = input_indices

        # call the original function with cast inputs
        cast_inputs = _cast_values(inputs, input_indices, _cast_int64)
        out = func(ctx, lns_ops, *cast_inputs, **kwargs)

        return _cast_values(out, cls._lnstensor_outputs, _cast_float64)

    return wrapper

def forward_no_ctx_decorator(func, cls):

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        ctx = args[0]
        inputs = args[1:]

        # call the original function with cast inputs
        cast_inputs = _cast_values(inputs, cls._lnstensor_inputs, _cast_int64)
        out = func(ctx, *cast_inputs, **kwargs)

        return _cast_values(out, cls._lnstensor_outputs, _cast_float64)

    return wrapper

# added to setup_context if it is defined
def setup_context_decorator(func, cls):

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # pop lns_ops from inputs
        ctx = args[0]
        lns_ops = args[1][0]
        inputs = args[1][1:]
        output = args[2]

        # add lns_ops to ctx for later use in backward
        ctx._lns_ops = lns_ops

        # store the indices of LNSTensor inputs on the ctx for backward
        input_indices = cls._lnstensor_inputs
        ctx._lnstensor_inputs = input_indices

        # call the original function with cast inputs and output
        cast_inputs = _cast_values(inputs, input_indices, _cast_int64)
        cast_output = _cast_values(output, cls._lnstensor_outputs, _cast_int64)

        # call the original function with lns_ops as the second argument
        return func(ctx, lns_ops, cast_inputs, cast_output, **kwargs)

    return wrapper

# always added to backward
def backward_decorator(func, cls):

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        ctx = args[0]
        grads = args[1:]

        # extract quantities from ctx
        lns_ops = ctx._lns_ops
        input_indices = ctx._lnstensor_inputs

        # cast grad_outputs
        cast_grads = _cast_values(grads, cls._lnstensor_outputs, _cast_int64)

        # call the original function with lns_ctx as second argument
        output = func(args[0], lns_ops, *cast_grads, **kwargs)
        cast_output = _cast_values(output, input_indices, _cast_float64)

        # To do: check if the number of gradients matches the number of inputs
        # since leaving this up to torch would correctly raise an error but
        # would claim there is an additional expected and received grad (for lns_ops
        # and its None grad)

        # return the output, adding None for the lns_ops position
        if isinstance(output, tuple):
            return (None,) + cast_output
        return None, cast_output

    return wrapper

class LNSFunction(torch.autograd.Function):
    """
    Base class for LNS operations that require custom forward and backward methods.
    This class should be subclassed for specific LNS operations.
    """

    _lnstensor_outputs: Optional[Tuple[int, ...]] = None

    @classmethod
    def _register_ops_decorator(cls, func_name, decorator):
        func = getattr(cls, func_name)

        if not getattr(func, "_is_lns_decorated", False):
            decorated = decorator(func, cls)
            decorated._is_lns_decorated = True
            setattr(cls, func_name, decorated)

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        # check if setup_context is defined in subclass or inherited
        # from torch.autograd.function._SingleLevelFunction. If it is
        # inherited, we only need to decorate forward and backward.
        # If it is defined, we need to decorate setup_context instead
        # of forward.
        if cls.setup_context is torch.autograd.function._SingleLevelFunction.setup_context:
            cls._register_ops_decorator("forward", forward_ctx_decorator)
        else:
            cls._register_ops_decorator("forward", forward_no_ctx_decorator)
            cls._register_ops_decorator("setup_context", setup_context_decorator)

        cls._register_ops_decorator("backward", backward_decorator)

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
    def apply(cls, *args, common_base: torch.Tensor = None, **kwargs):
        """
        Applies the LNS operation defined by this class.
        This method is used to call the forward and backward methods.

        The `common_base` parameter is used to explicitly specify the base
        for LNSTensor operations. If not provided, it will be inferred from
        the first LNSTensor argument.

        Note that any keyword arguments passed to this method except for `common_base`
        will raise an error, as `torch.autograd.Function` does not support keyword
        arguments. Instead, use positional arguments only.

        In addition, this method converts any `LNSTensor` arguments to their
        internal representation (i.e., the underlying tensor) before calling the
        forward method. This is necessary for LNSTensor internal behavior.
        """
        tensor_module = _get_tensor_module()
        ops_module = _get_ops_module()

        # This check is also performed in the base class, but we do it here too
        # in case PyTorch decides to change the behavior of the apply method.
        if kwargs:
            raise ValueError("torch.autograd.Function does not support keyword arguments. Please use positional arguments only.")

        # Convert LNSTensor arguments to internal representation. This is necessary
        # because the autograd.Function expects tensors, not LNSTensor objects.
        internal_args = []
        lnstensor_inputs = []
        for i in range(len(args)):
            arg = args[i]

            if common_base is None and isinstance(arg, tensor_module.LNSTensor):
                common_base = arg.base

            if isinstance(arg, tensor_module.LNSTensor):
                internal_args.append(arg._lns)
                lnstensor_inputs.append(i)
            else:
                internal_args.append(arg)

        ops = ops_module.LNSOps(common_base) if common_base is not None else None

        # store the indices of LNSTensor inputs on the class to be accessed
        # by forward decorator and (if defined) setup_context decorators
        cls._lnstensor_inputs = lnstensor_inputs

        # call the forward method of the class with the internal arguments
        result = super().apply(ops, *internal_args)

        # delete the stored input indices to avoid confusion later
        del cls._lnstensor_inputs

        # get all output tensors and store them in a tuple
        if isinstance(result, (list, tuple)):
            result_iter = result
        else:
            result_iter = [result]

        # we register hooks on each input to each output for gradient accumulation
        for output in result_iter:

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

        if isinstance(result, torch.Tensor):
            return _tensor_module.lnstensor(
                result, from_lns=True, b=common_base
            ) if cls._lnstensor_outputs is None or 0 in cls._lnstensor_outputs else result

        elif isinstance(result, list):
            return [
                _tensor_module.lnstensor(result[i], from_lns=True, b=common_base)
                if cls._lnstensor_outputs is None or i in cls._lnstensor_outputs else result[i]
                for i in range(len(result))
            ]

        elif isinstance(result, tuple):
            return tuple(
                _tensor_module.lnstensor(result[i], from_lns=True, b=common_base)
                if cls._lnstensor_outputs is None or i in cls._lnstensor_outputs else result[i]
                for i in range(len(result))
            )

        else:
            return result


class LNSNonDifferentiableFunction:

    _lnstensor_outputs: Optional[Tuple[int, ...]] = None

    @staticmethod
    def forward(ctx, *args, **kwargs):
        """
        Forward pass for the non-differentiable LNS operation.
        Should be implemented in subclasses.
        """
        raise NotImplementedError("Forward method must be implemented in subclasses.")

    @classmethod
    def apply(cls, *args, common_base: torch.Tensor = None, **kwargs):

        if kwargs:
            raise ValueError(
                "LNSNonDifferentiableFunction does not support keyword arguments. "
                "Please use positional arguments only."
            )

        tensor_module = _get_tensor_module()
        ops_module = _get_ops_module()

        internal_args = []
        for i in range(len(args)):
            arg = args[i]

            if common_base is None and isinstance(arg, tensor_module.LNSTensor):
                common_base = arg.base

            if isinstance(arg, tensor_module.LNSTensor):
                internal_args.append(arg.lns) # .lns views to int64
            else:
                internal_args.append(arg)

        ops = ops_module.LNSOps(common_base) if common_base is not None else None
        result = cls.forward(ops, *internal_args, **kwargs)

        if isinstance(result, torch.Tensor):
            return _tensor_module.lnstensor(
                result, from_lns=True, b=common_base
            ) if cls._lnstensor_outputs is None or 0 in cls._lnstensor_outputs else result

        elif isinstance(result, list):
            return [
                _tensor_module.lnstensor(result[i], from_lns=True, b=common_base)
                if cls._lnstensor_outputs is None or i in cls._lnstensor_outputs else result[i]
                for i in range(len(result))
            ]

        elif isinstance(result, tuple):
            return tuple(
                _tensor_module.lnstensor(result[i], from_lns=True, b=common_base)
                if cls._lnstensor_outputs is None or i in cls._lnstensor_outputs else result[i]
                for i in range(len(result))
            )

        else:
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
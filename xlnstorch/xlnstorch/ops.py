from __future__ import annotations
import inspect
from typing import Callable, Any, Iterable, Tuple, Union, Optional

import torch
from xlnstorch import apply_lns_op

_KIND_MAP: dict[str, inspect._ParameterKind] = {
    "po": inspect.Parameter.POSITIONAL_ONLY,
    "pk": inspect.Parameter.POSITIONAL_OR_KEYWORD,
    "ko": inspect.Parameter.KEYWORD_ONLY,
    "*": inspect.Parameter.VAR_POSITIONAL,
    "**": inspect.Parameter.VAR_KEYWORD,
    "positional_only": inspect.Parameter.POSITIONAL_ONLY,
    "positional_or_keyword": inspect.Parameter.POSITIONAL_OR_KEYWORD,
    "keyword_only": inspect.Parameter.KEYWORD_ONLY,
    "var_positional": inspect.Parameter.VAR_POSITIONAL,
    "var_keyword": inspect.Parameter.VAR_KEYWORD,
}

def _build_signature(
        params: Iterable[Tuple],
        return_type: Any = inspect._empty,
        include_self: bool = True,
    ):

    if include_self:
        params = (("self", "pk", None),) + tuple(params)

    parameters: list[inspect.Parameter] = []

    for entry in params:
        if len(entry) not in (3, 4):
            raise ValueError(
                "Each parameter tuple must have 3 or 4 items "
                "(name, kind, annotation [, default])."
            )

        name, kind, annotation, *rest = entry
        default = rest[0] if rest else inspect._empty

        if isinstance(kind, str):
            try:
                kind = _KIND_MAP[kind.lower()]
            except KeyError as e:
                raise ValueError(f"Unknown parameter kind string '{kind}'.") from e

        parameters.append(
            inspect.Parameter(
                name,
                kind,
                annotation=annotation,
                default=default,
            )
        )

    return inspect.Signature(parameters, return_annotation=return_type)

def _create_lns_op_func(
    op_name: str,
    torch_op: Callable,
    *,
    docstring: Optional[str] = None,
    signature: Optional[inspect.Signature] = None,
) -> Callable:
    """
    Factory function to create LNS operation functions. These functions are
    the operations that are performed on the internal representations of
    LNSTensors.

    Parameters
    ----------
    op_name : str
        Name of the operation (e.g., 'add', 'mul')
    torch_op : Callable
        The corresponding PyTorch operation function
    docstring : str, optional
        Docstring for the function
    signature : inspect.Signature, optional
        Signature of the function, if provided.
    return_type : Any, optional
        Return type of the function, if provided.

    Returns
    -------
    Callable
        The created LNS operation function
    """
    def func(*args: Any, **kwargs: Any):
        return apply_lns_op(torch_op, *args, **kwargs)

    func.__name__ = f"lns_{op_name}"

    if docstring:
        func.__doc__ = docstring
    else:
        func.__doc__ = (
            f"See docs for :py:func:`{torch_op.__module__}.{torch_op.__name__}` for "
            f"parameter/return details. Typically, torch.Tensor arguments are the "
            f"equivalent internal representations of LNStensors."
        )

    if signature is not None:
        func.__signature__ = signature

    return func

class LNSOps:

    def __init__(self, base: torch.Tensor):
        self.base = base
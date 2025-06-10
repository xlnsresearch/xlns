.. currentmodule:: xlnstorch

.. _operations-doc:

Operations
==========

The ``xlnstorch`` package provides a set of analogous operations to
PyTorch's built-in operations. These operations are registered with
PyTorch's internal dispatch mechanism, so that they can be used in
the same way as PyTorch's built-in operations. For example,

.. code-block:: python

    import xlnstorch as xltorch

    x = xltorch.lnstensor([1.0, 2.0])
    y = xltorch.lnstensor([3.0, 4.0])

    z = x + y
    print(z)
    # LNSTensor(value=tensor([4.0000, 6.0000], dtype=torch.float64), base=1.0000000826295863)

Custom Operations
-----------------

If you want to implement your own custom implementation of an operation,
or define a new operation, you can use the following functions.

.. autosummary::
    :toctree: generated
    :nosignatures:

    implements
    get_implementation
    set_default_implementation
    get_default_implementation_key
    override_implementation
    align_lnstensor_bases
    format_lnstensor_operands

    operators.implement_sbdb
    operators.sbdb

Internal Operations
-------------------

These are the internal operations performed on ``torch.Tensor`` internal
representations of ``LNSTensor`` objects related to arithmetic operations.
These operations are useful if you want to implement your own custom
operations. For the most part, these internal operator functions wrap the
``apply_lns_op()`` function.

Arithmetic Operations
~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: generated
    :nosignatures:

    operators.lns_add
    operators.lns_sub
    operators.lns_mul
    operators.lns_div
    operators.lns_neg
    operators.lns_abs
    operators.lns_sqrt
    operators.lns_square
    operators.lns_pow
    operators.lns_reciprocal
    operators.lns_sign
    operators.lns_positive

Comparison Operations
~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: generated
    :nosignatures:

    operators.lns_equal
    operators.lns_eq
    operators.lns_ne
    operators.lns_ge
    operators.lns_gt
    operators.lns_le
    operators.lns_lt
    operators.lns_isclose
    operators.lns_allclose
    operators.lns_any
    operators.lns_all
    operators.lns_isin
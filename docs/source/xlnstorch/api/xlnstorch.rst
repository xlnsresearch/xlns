.. currentmodule:: xlnstorch

xlnstorch
=========

The xlnstorch package provides an LNSTensor class, analogous to PyTorch's Tensor,
which is designed to handle LNS arithmetic. The LNSTensor class is built on top
of PyTorch's Tensor, allowing it to leverage PyTorch's features such as automatic
differentiation, while also providing specialized methods for LNS operations.

Tensor Creation Ops
-------------------

.. autosummary::
    :toctree: generated
    :nosignatures:

    lnstensor
    zeros
    zeros_like
    ones
    ones_like
    full
    full_like
    rand
    rand_like
    randn
    randn_like
    empty
    empty_like

Constants
---------

.. data:: LNS_ZERO

    An int64 scalar tensor representing zero in the LNS. This is a
    special value defined to be
    :math:`\left( -2^{52} \ll 1 \right) \mid 1 = -9007199254740991`.
    It is independent of an LNS object's base.

.. data:: LNS_INF

    An int64 scalar tensor representing positive infinity in the LNS.
    This is a special value defined to be
    :math:`2^{53} = 9007199254740992`.
    It is independent of an LNS object's base.

.. data:: LNS_NEG_INF

    An int64 scalar tensor representing negative infinity in the LNS.
    This is a special value defined to be
    :math:`\left( 2^{53} \right) - 1 = 9007199254740991`.
    It is independent of an LNS object's base.

.. data:: LNS_ONE

    An int64 scalar tensor representing one in the LNS. For any
    LNS base :math:`B`, we have :math:`\log_B(1) = 0`, so this value
    is independent of base. It is defined to be
    :math:`\left( 0 \ll 1 \right) \mid 0 = 0`.

.. data:: LNS_NEG_ONE

    An int64 scalar tensor representing one in the LNS. For any
    LNS base :math:`B`, we have :math:`\log_B(1) = 0`, so this value
    is independent of base. It is defined to be
    :math:`\left( 0 \ll 1 \right) \mid 1 = 1`.

.. data:: CSRC_AVAILABLE

    A boolean flag indicating whether the C++ extension is available.
    When True, high-performance C++ implementations are used; when
    False, pure Python implementations are used as fallback. To find
    out why the C++ extension is not available, install the xlnstorch
    package with the ``--verbose`` flag to see the build logs. The
    internal C++ functions can be accessed via the ``csrc`` submodule.

Custom Operations
-----------------

The ``xlnstorch`` package provides a set of analogous operations to
PyTorch's built-in operations. These operations are registered with
PyTorch's internal dispatch mechanism, so that they can be used in
the same way as PyTorch's built-in operations. For example,

.. code-block:: python

    import xlnstorch as xltorch

    x = xltorch.lnstensor([1.0, 2.0], f=23)
    y = xltorch.lnstensor([3.0, 4.0], f=23)

    z = torch.add(x, y)
    print(z)
    # LNSTensor(value=[4.0000, 6.0000], prec=23)

If you want to implement your own custom implementation of an operation,
define a new operation, or use an alternative implementation of an
operation, you can use the following functions.

.. autosummary::
    :toctree: generated
    :nosignatures:

    implements
    get_implementation
    set_default_implementation
    get_default_implementation_key
    override_implementation

    implements_sbdb
    set_default_sbdb_implementation
    override_sbdb_implementation
    register_xlnsconf_implementation
    sbdb

    align_lnstensor_bases
    format_lnstensor_operands
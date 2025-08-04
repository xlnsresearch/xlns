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

    A float64 scalar tensor representing zero in the LNS. This is a
    special value defined to be
    :math:`\left( -2^{52} \ll 1 \right) \mid 1 = -9007199254740991`.
    It is independent of an LNS object's base.

.. data:: LNS_ONE

    A float64 scalar tensor representing one in the LNS. For any
    LNS base :math:`B`, we have :math:`\log_B(1) = 0`, so this value
    is independent of base. It is defined to be
    :math:`\left( 0 \ll 1 \right) \mid 0 = 0`.

.. data:: LNS_NEG_ONE

    A float64 scalar tensor representing one in the LNS. For any
    LNS base :math:`B`, we have :math:`\log_B(1) = 0`, so this value
    is independent of base. It is defined to be
    :math:`\left( 0 \ll 1 \right) \mid 1 = 1`.

.. data:: _C_AVAILABLE

    A boolean flag indicating whether the C++ extension is available.
    When True,  high-performance C++ implementations are used; when
    False, pure Python implementations are used as fallback. To find
    out why the C++ extension is not available, install the xlnstorch
    package with the ``--verbose`` flag to see the build logs.
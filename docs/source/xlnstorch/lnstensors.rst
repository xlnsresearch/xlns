.. currentmodule:: xlnstorch
.. automodule:: xlnstorch

.. _lnstensor-doc:

xlnstorch.LNSTensor
===================

An :class:`xlnstorch.LNSTensor` is a wrapper for a :class:`torch.Tensor`
that stores an 'internal representation' Tensor object along with an
LNS base.

Todo: Add more information/math.

LNSTensor class reference
-------------------------

.. class:: LNSTensor()

    There are two ways to create a tensor currently.

    - To create a tensor with pre-existing data, use :func:`xlnstorch.lnstensor`.
    - If necessary, use the ``xlnstorch.LNSTensor()`` constructor directly, although
      the use of this is discouraged.


.. autoattribute:: LNSTensor._lns
.. autoattribute:: LNSTensor.base

.. autosummary::
    :toctree: generated
    :nosignatures:

    LNSTensor.lns
    LNSTensor.value
    LNSTensor.backward
    LNSTensor.grad

.. autosummary::
    :toctree: generated
    :nosignatures:

    lnstensor
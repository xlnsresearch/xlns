.. currentmodule:: xlnstorch

.. _lnstensor-doc:

xlnstorch.LNSTensor
===================

An :class:`xlnstorch.LNSTensor` is a wrapper for a :class:`torch.Tensor`
that stores an 'internal representation' Tensor object along with an
LNS base.

Mathematical Context and Internal Representation
------------------------------------------------

We represent a non-zero real number :math:`x` in the LNS (Logarithmic Number System)
using the following scheme:

Let :math:`B` denote the chosen base and let

.. math::
    x = (-1) ^ {s_x} \cdot B^{X}

where:

.. math::
    X = \log_B |x| \text{ and } s_x = \begin{cases}
    0 & \text{if } x \ge 0 \\
    1 & \text{if } x \le 0
    \end{cases}

We can pack the logarithm and sign into a single ``int64`` value as follows:

.. math::
    x' = (\mathrm{round}(X) \ll 1) + s_x

Since we are quantising the logarithm, we can represent it as an integer value. It is
useful to choose a base close to 1, to ensure that the quantisation does not lose too
much precision. This is typically done by choosing a base of the form :math:`B = 2^{2^{-f}}`
for some integer :math:`f`, where :math:`f` is the number of fractional bits in the
LNS representation.

The internal representation of an LNSTensor encodes the LNS value as an integer, but for
compatibility with PyTorch's autograd system, it is stored in a tensor of type ``torch.float64``.
This allows gradients to be computed and propagated correctly during backpropagation, as
PyTorch's autograd does not support integer tensors for gradient computation.

Although the storage type is a floating point, the actual values represent
integers. This means that before performing any bitwise operations or
integer-specific manipulations, the values must be explicitly converted
back to integers. Failing to do so can result in incorrect behavior, since
bitwise operations on floating point types are not valid and may produce
unexpected results. Note that if you are not implementing custom functionality,
you typically do not need to worry about this, as the library handles these
conversions.

This design ensures that LNSTensor objects can fully participate in PyTorch's
computation graph and benefit from automatic differentiation, while still
maintaining the integrity of their integer-based internal encoding for LNS
arithmetic and bit-level operations.

LNSTensor class reference
-------------------------

.. class:: LNSTensor

    There are two ways to create a tensor currently.

    - To create a tensor with pre-existing data, use :func:`xlnstorch.lnstensor`.
    - If necessary, use the ``xlnstorch.LNSTensor()`` constructor directly, although
      the use of this is discouraged.

.. autosummary::
    :toctree: generated
    :nosignatures:

    LNSTensor.get_internal_tensor

    LNSTensor.base
    LNSTensor.lns
    LNSTensor.value
    LNSTensor.grad
    LNSTensor.shape
    LNSTensor.ndim
    LNSTensor.requires_grad

    LNSTensor.item
    LNSTensor.size
    LNSTensor.numel
    LNSTensor.backward
    LNSTensor.broadcast_to
    LNSTensor.dim
    LNSTensor.clone
    LNSTensor.squeeze
    LNSTensor.unsqueeze
    LNSTensor.detach
    LNSTensor.requires_grad_

Creation Operations
~~~~~~~~~~~~~~~~~~~

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
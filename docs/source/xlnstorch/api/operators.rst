.. currentmodule:: xlnstorch.ops

.. _operators-doc:

Operators
==========

As part of the xlnstorch package, we provide a set of operations
that are analogous to PyTorch's built-in operations. These operations
are registered with PyTorch's internal dispatch mechanism, allowing
them to be used in the same way as PyTorch's built-in operations.
These operations can be accessed via the traditional PyTorch mechanism,
such as ``torch.add()`` or ``torch.matmul()``.

C++ Implementations
-------------------

For operations that are computationally intensive, we provide C++ implementations
that can be used to accelerate the computation. These implementations are
available if the package is built with C++ extensions enabled and are enabled
by default. You can toggle the use of C++ implementations with the following function:

.. code-block:: python

    from xlnstorch.operators import toggle_cpp_implementations

    toggle_cpp_implementations(True) # enable C++ implementations
    toggle_cpp_implementations(False) # disable C++ implementations

Currently, there are C++ implementations for the following operations:

- Addition
- Summation
- Matrix Multiplication
- Convolution functions

Custom SBDB functions
---------------------

In LNS, numbers are represented by their logarithms. To implement addition
and subtraction, we work with their logarithmic forms and a special function,
commonly known as the "sum/difference in the log domain" (sbdb) function,
which is closely related to Gaussian logarithms.

Suppose two numbers :math:`X` and :math:`Y` are represented as:

.. math::

    x &= \log_B \lvert X \rvert \\
    y &= \log_B \lvert Y \rvert

To compute :math:`\log_B \left( \vert X \rvert + \lvert Y \rvert \right)` and
:math:`\log_B \left( \vert X \rvert - \lvert Y \rvert \right)`, we use the
identities:

.. math::

    \log_B \left( \lvert X \rvert + \lvert Y \rvert \right) &= x + s_B \left( y - x \right) \\
    \log_B \left( \lvert X \rvert - \lvert Y \rvert \right) &= x + d_B \left( y - x \right)

where the "sum" and "difference" helper functions are defined as:

.. math::

    s_B \left( z \right) &= \log_B \left( 1 + B ^ {z} \right) \\
    d_B \left( z \right) &= \log_B \lvert 1 - B ^ {z} \rvert

However, computing :math:`s_B(z)` and :math:`d_B(z)` directly is computationally
expensive, especially on hardware that does not support efficient logarithm and
exponentiation. For this reason, `xlnstorch` provides fast, approximate implementations
of these functions to accelerate LNS addition and subtraction.

By using these approximations, we achieve a good trade-off between numerical accuracy
and computational efficiency in LNS arithmetic within the package. See ``xlnsconf``
for more details on these implementations and the papers that describe them.

Tab
~~~

The 'tab' method provides fast LNS addition and subtraction by precomputing the values
of :math:`s_B(z)` and :math:`d_B(z)` and storing them in lookup tables. During computation,
these tables are used to quickly retrieve approximate results instead of calculating the
logarithms and exponentials directly. This approach enables rapid and efficient evaluation
of LNS arithmetic operations with minimal computational cost, at the expense of increased
memory usage and a fixed precision determined by the table resolution.

To use the 'tab' method, you must first initialize the lookup table with the desired base or
precision, and a filestem to store the table.

.. code-block:: python

    import xlnstorch as xltorch

    xltorch.operators.tab.get_table("filestem", f=10)
    xltorch.set_default_sbdb_implementation("tab")

    a = xltorch.lnstensor([1.0, 2.0], f=10)
    b = xltorch.lnstensor([3.0, 4.0], f=10)
    c = torch.add(a, b) # uses the tab implementation

Utah-Tayco
~~~~~~~~~~

The 'utah_tayco' method is an approximate implementation of the
sbdb function that uses unpartitioned linear Taylor interpolation
and/or cotransformation of the Gaussian logarithm.

To use the 'utah_tayco' method, there is no initialization required
and can be used as follows:

.. code-block:: python

    import xlnstorch as xltorch

    xltorch.set_default_sbdb_implementation("utah_tayco")

    a = xltorch.lnstensor([1.0, 2.0], f=10)
    b = xltorch.lnstensor([3.0, 4.0], f=10)
    c = torch.add(a, b) # uses the utah_tayco implementation

Internal Operators
-------------------

For advanced users, we provide a set of internal operator functions.
These are defined under the ``xlnstorch.ops.LNSOps`` object. Any types
denoted by ``torch.LongTensor`` refer to inputs that should be the
internal int64 representations of ``LNSTensor`` objects. Any types
denoted by just ``torch.Tensor`` refer to standard PyTorch tensors
(typically with dtype torch.float64 but this is context dependent).

If you are implementing custom functionality, you may find these
functions useful. However, for most users, the standard PyTorch
operations (e.g., ``torch.add()``, ``torch.matmul()``, etc.) should
be sufficient.

One example of using ``LNSOps`` is:

.. code-block:: python

    import xlnstorch as xltorch
    import torch

    a = torch.tensor(0, dtype=torch.int64) # internal representation of 1.0
    b = torch.tensor(1, dtype=torch.int64) # internal representation of -1.0

    ops = xltorch.ops.LNSOps(xltorch.tensor_utils.get_base_from_precision(10))

    c = ops.mul(a, b) # 1 i.e. the internal representation of -1.0

Helper Methods
~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/ops
   :nosignatures:

    LNSOps.to_lns
    LNSOps.from_lns
    LNSOps.zeros
    LNSOps.zeros_like
    LNSOps.ones
    LNSOps.ones_like
    LNSOps.full
    LNSOps.full_like
    LNSOps.sum_to_size

Arithmetic Operations
~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/ops
   :nosignatures:

    LNSOps.add
    LNSOps.sub
    LNSOps.mul
    LNSOps.div
    LNSOps.neg
    LNSOps.abs
    LNSOps.sqrt
    LNSOps.square
    LNSOps.pow
    LNSOps.exp
    LNSOps.log
    LNSOps.reciprocal
    LNSOps.sign
    LNSOps.positive
    LNSOps.sum
    LNSOps.prod
    LNSOps.mean
    LNSOps.var
    LNSOps.matmul
    LNSOps.transpose

Comparison Operations
~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/ops
   :nosignatures:

    LNSOps.equal
    LNSOps.eq
    LNSOps.ne
    LNSOps.ge
    LNSOps.gt
    LNSOps.le
    LNSOps.lt
    LNSOps.isclose
    LNSOps.allclose
    LNSOps.any
    LNSOps.all
    LNSOps.isin
    LNSOps.sort
    LNSOps.argsort
    LNSOps.kthvalue
    LNSOps.maximum
    LNSOps.minimum
    LNSOps.max
    LNSOps.argmax
    LNSOps.min
    LNSOps.argmin
    LNSOps.clamp

Miscellaneous Operations
~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/ops
   :nosignatures:

    LNSOps.broadcast_to
    LNSOps.clone
    LNSOps.squeeze
    LNSOps.unsqueeze
    LNSOps.stack
    LNSOps.cat
    LNSOps.chunk
    LNSOps.where
    LNSOps.pad

Loss Operations
~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/ops
   :nosignatures:

    LNSOps.mse_loss
    LNSOps.l1_loss
    LNSOps.binary_cross_entropy
    LNSOps.binary_cross_entropy_with_logits
    LNSOps.nll_loss
    LNSOps.poisson_nll_loss
    LNSOps.hinge_embedding_loss
    LNSOps.kl_div
    LNSOps.margin_ranking_loss
    LNSOps.gaussian_nll_loss
    LNSOps.huber_loss
    LNSOps.smooth_l1_loss
    LNSOps.cross_entropy

Activation Operations
~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/ops
   :nosignatures:

    LNSOps.relu
    LNSOps.leaky_relu
    LNSOps.threshold
    LNSOps.tanh
    LNSOps.sigmoid
    LNSOps.logsigmoid
    LNSOps.softmin
    LNSOps.softmax
    LNSOps.log_softmax
    LNSOps.hardtanh
    LNSOps.hardswish
    LNSOps.elu
    LNSOps.selu
    LNSOps.celu
    LNSOps.prelu
    LNSOps.rrelu
    LNSOps.glu
    LNSOps.hardshrink
    LNSOps.tanhshrink
    LNSOps.softsign
    LNSOps.softplus
    LNSOps.softshrink
    LNSOps.hardsigmoid
    LNSOps.silu
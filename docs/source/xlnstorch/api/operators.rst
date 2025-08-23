.. currentmodule:: xlnstorch.operators

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

    xltorch.operators.implementations.tab.get_table("filestem", f=10)
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

These are the internal operations performed on ``torch.Tensor`` internal
representations of ``LNSTensor`` objects related to arithmetic operations.
These operations are useful if you want to implement your own custom
operations. For the most part, these internal operator functions wrap the
``apply_lns_op()`` function.

Arithmetic Operations
~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: generated/operators
    :nosignatures:

    lns_add
    lns_sub
    lns_mul
    lns_div
    lns_neg
    lns_abs
    lns_sqrt
    lns_square
    lns_pow
    lns_exp
    lns_log
    lns_reciprocal
    lns_sign
    lns_positive
    lns_sum
    lns_prod
    lns_mean
    lns_var
    lns_matmul
    lns_transpose

Comparison Operations
~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: generated/operators
    :nosignatures:

    lns_equal
    lns_eq
    lns_ne
    lns_ge
    lns_gt
    lns_le
    lns_lt
    lns_isclose
    lns_allclose
    lns_any
    lns_all
    lns_isin
    lns_sort
    lns_argsort
    lns_kthvalue
    lns_maximum
    lns_minimum

Miscellaneous Operations
~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: generated/operators
    :nosignatures:

    lns_broadcast_to
    lns_clone
    lns_squeeze
    lns_unsqueeze
    lns_stack
    lns_cat
    lns_chunk
    lns_where

Loss Operations
~~~~~~~~~~~~~~~

Note that in practice, you should use the standard PyTorch
loss classes, such as ``torch.nn.MSELoss``, which are already
implemented to work with ``LNSTensor`` objects. However, if you
want to implement your own custom loss functions, or want more
control over the loss computation, you can use the following
functions.

.. autosummary::
    :toctree: generated/operators
    :nosignatures:

    lns_mse_loss
    lns_l1_loss
    lns_binary_cross_entropy
    lns_binary_cross_entropy_with_logits
    lns_nll_loss
    lns_poisson_nll_loss
    lns_hinge_embedding_loss
    lns_kl_div
    lns_margin_ranking_loss
    lns_gaussian_nll_loss
    lns_huber_loss
    lns_smooth_l1_loss
    lns_cross_entropy

Activation Operations
~~~~~~~~~~~~~~~~~~~~~

Again, in practice, you should use the standard PyTorch
activation classes, such as ``torch.nn.ReLU``, which are
already implemented to work with ``LNSTensor`` objects.

.. autosummary::
    :toctree: generated/operators
    :nosignatures:

    lns_relu
    lns_relu_
    lns_leaky_relu
    lns_leaky_relu_
    lns_threshold
    lns_threshold_
    lns_tanh
    lns_sigmoid
    lns_logsigmoid
    lns_softmin
    lns_softmax
    lns_log_softmax
    lns_hardtanh
    lns_hardswish
    lns_elu
    lns_selu
    lns_celu
    lns_prelu
    lns_rrelu
    lns_glu
    lns_hardshrink
    lns_tanhshrink
    lns_softsign
    lns_softplus
    lns_softshrink
    lns_hardsigmoid
    lns_silu

Layer Operations
~~~~~~~~~~~~~~~~

As per usual, you should use the standard PyTorch
layer classes, such as ``torch.nn.Linear``, which
support ``LNSTensor`` objects.

.. autosummary::
    :toctree: generated/operators
    :nosignatures:

    lns_linear
    lns_bilinear
    lns_dropout
    lns_dropout1d
    lns_dropout2d
    lns_dropout3d
    lns_conv1d
    lns_conv2d
    lns_conv3d
    lns_avg_pool1d
    lns_avg_pool2d
    lns_avg_pool3d
    lns_adaptive_avg_pool1d
    lns_adaptive_avg_pool2d
    lns_adaptive_avg_pool3d
    lns_batch_norm
    lns_layer_norm
    lns_max_pool1d
    lns_max_pool2d
    lns_max_pool3d
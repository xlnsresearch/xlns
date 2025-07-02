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
    operators.lns_exp
    operators.lns_log
    operators.lns_reciprocal
    operators.lns_sign
    operators.lns_positive
    operators.lns_sum
    operators.lns_matmul
    operators.lns_transpose

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
    operators.lns_sort
    operators.lns_argsort
    operators.lns_kthvalue
    operators.lns_maximum
    operators.lns_minimum

Loss Operations
~~~~~~~~~~~~~~~

Note that in practice, you should use the standard PyTorch
loss classes, such as ``torch.nn.MSELoss``, which are already
implemented to work with ``LNSTensor`` objects. However, if you
want to implement your own custom loss functions, or want more
control over the loss computation, you can use the following
functions.

.. autosummary::
    :toctree: generated
    :nosignatures:

    operators.lns_mse_loss
    operators.lns_l1_loss
    operators.lns_binary_cross_entropy
    operators.lns_binary_cross_entropy_with_logits
    operators.lns_nll_loss
    operators.lns_poisson_nll_loss
    operators.lns_hinge_embedding_loss
    operators.lns_kl_div
    operators.lns_margin_ranking_loss
    operators.lns_gaussian_nll_loss
    operators.lns_huber_loss
    operators.lns_smooth_l1_loss

Activation Operations
~~~~~~~~~~~~~~~~~~~~~

Again, in practice, you should use the standard PyTorch
activation classes, such as ``torch.nn.ReLU``, which are
already implemented to work with ``LNSTensor`` objects.

.. autosummary::
    :toctree: generated
    :nosignatures:

    operators.lns_relu
    operators.lns_relu_
    operators.lns_leaky_relu
    operators.lns_leaky_relu_
    operators.lns_threshold
    operators.lns_threshold_
    operators.lns_tanh
    operators.lns_sigmoid
    operators.lns_logsigmoid
    operators.lns_softmin
    operators.lns_softmax
    operators.lns_log_softmax
    operators.lns_hardtanh
    operators.lns_hardswish
    operators.lns_elu
    operators.lns_selu
    operators.lns_celu
    operators.lns_prelu
    operators.lns_rrelu
    operators.lns_glu
    operators.lns_hardshrink
    operators.lns_tanhshrink
    operators.lns_softsign
    operators.lns_softplus
    operators.lns_softshrink
    operators.lns_hardsigmoid
    operators.lns_silu

Layer Operations
~~~~~~~~~~~~~~~~

As per usual, you should use the standard PyTorch
layer classes, such as ``torch.nn.Linear``, which
support ``LNSTensor`` objects.

.. autosummary::
    :toctree: generated
    :nosignatures:

    operators.lns_linear
    operators.lns_bilinear
    operators.lns_dropout
    operators.lns_dropout1d
    operators.lns_dropout2d
    operators.lns_dropout3d
    operators.lns_conv1d
    operators.lns_conv2d
    operators.lns_conv3d
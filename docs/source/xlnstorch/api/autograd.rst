.. currentmodule:: xlnstorch.autograd

Autograd
========

The ``torch.autograd`` module provides support for automatic differentiation
in pytorch. In xlnstorch, the autograd system is extended to support LNS
operations on LNSTensor objects. This allows for gradients to be computed
with respect to LNS operations, enabling the use of LNSTensor in neural networks
and other machine learning models.

LNSFunction
-----------

To define a differentiable custom LNS operation, you can subclass
``xlnstorch.autograd.LNSFunction``. This class provides the necessary methods
to implement the forward and backward passes for your custom operation and is
analogous to PyTorch's ``torch.autograd.Function``.

.. autoclass:: LNSFunction

.. note::

    When using the LNSFunction class, you should pass the LNSTensor objects
    as inputs to the apply method, not their internal values. This allows
    the autograd system to correctly track the operations and compute gradients.
    However, in the forward method, the inputs will be the internal values
    of the LNSTensor objects, so that you can perform the necessary LNS
    operations directly. All differentiable outputs should be returned as
    float64 tensors. For example:

    .. code-block:: python

        import xlnstorch as xltorch

        class MyLNSFunction(xltorch.autograd.LNSFunction):

            @staticmethod
            def forward(x, y): # x and y are float64 tensors
                x_packed, y_packed = x.to(torch.int64), y.to(torch.int64)
                result = f(x_packed, y_packed).to(torch.float64)
                return result # result is a float64 tensor

            @staticmethod
            def setup_context(ctx, inputs, output):
                pass

            @staticmethod
            def backward(ctx, grad_output):
                return grad_output, grad_output

        a = xltorch.lnstensor([1.0, 2.0], f=23)
        b = xltorch.lnstensor([3.0, 4.0], f=23)
        c = MyLNSFunction.apply(a, b) # we pass the LNSTensor objects to apply

If you want all float64 inputs to be automatically bitcasted to int64 and
all int64 outputs to be automatically bitcasted to float64, you can use
the ``with_bitcast`` decorator.

.. autosummary::
    :toctree: generated

    with_bitcast

Fanout Functions
----------------

In early development of xlnstorch, the autograd system would break when fanout occured.
This is where a single LNSTensor is used in multiple operations, so that the same
value has multiple paths through the computation graph. To handle and detect this,
the following functions were used. This problem has been resolved but the functions
are still available for reference. Detecting fanout is equivalent to checking whether
the computation graph is a tree or not.

.. autosummary::
    :toctree: generated

    has_fanout
    find_fanout
    raise_fanout_error
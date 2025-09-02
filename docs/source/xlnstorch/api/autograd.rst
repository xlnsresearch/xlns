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
    However, in the forward method, the inputs will be the int64 internal values
    of the LNSTensor objects, so that you can perform the necessary LNS
    operations directly. The indices of LNSTensor outputs should be denoted
    in the ``_lnstensor_outputs`` list field of your function class. This
    allows the autograd system to correctly handle the outputs of your custom
    operation. Leaving the _lnstensor_outputs list empty indicates that
    all outputs are LNSTensors.

    .. code-block:: python

        import xlnstorch as xltorch

        class MyLNSFunction(xltorch.autograd.LNSFunction):

            # First output is an LNSTensor. This field is optional here
            # since by default all outputs are assumed to be LNSTensors.
            _lnstensor_outputs = [0]

            @staticmethod
            def forward(x, y): # x and y are int64 tensors
                result = f(x, y)
                return result # result is an int64 tensor

            @staticmethod
            def setup_context(ctx, inputs, output):
                pass

            @staticmethod
            def backward(ctx, grad_output):
                return grad_output, grad_output

        a = xltorch.lnstensor([1.0, 2.0], f=23)
        b = xltorch.lnstensor([3.0, 4.0], f=23)
        c = MyLNSFunction.apply(a, b) # we pass the LNSTensor objects to apply

We also provide an analogous class ``xlnstorch.autograd.LNSNonDifferentiableFunction``
for operations that are not differentiable. This class is similar to ``LNSFunction``
but does not require the implementation of backward or setup_context methods. This
is useful since it handles the pre and post processing of inputs and outputs for you.
For example,

.. code-block:: python

    import xlnstorch as xltorch

    class MyNonDiffFunction(xltorch.autograd.LNSNonDifferentiableFunction):

        @staticmethod
        def forward(x, y): # x and y are int64 tensors
            result = f(x, y)
            return result # result is an int64 tensor

    a = xltorch.lnstensor([1.0, 2.0], f=23)
    b = xltorch.lnstensor([3.0, 4.0], f=23)
    c = MyLNSNonDifferentiableFunction.apply(a, b) # we pass the LNSTensor objects to apply

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
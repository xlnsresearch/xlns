Quickstart
==========

xlnstorch attempts to provide a seamless experience for users familiar with PyTorch.
Its main object is the `LNSTensor`, which wraps a PyTorch tensor and provides
additional functionality for LNS arithmetic.

The Basics
----------

Similar to PyTorch, you can create an `LNSTensor` from array-like data.
For LNSTensors, we can also specify the precision, `f`, or base, `b`.
This determines how precise our representation of a number is. A base
closer to 1, or a larger precision will result in a more precise
representation.

.. note::

    The precision is related to the base by the formula :math:`b = 2 ^ {2 ^ {-f}}`.

    If no base or precision is specified, the default base is derived from the
    `xlns` package's `xlnsB` variable, which defaults to :math:`2 ^ {2 ^ {-23}}`.

.. code-block:: python

    import xlnstorch as xlt

    a = xlt.lnstensor([1.0, 2.0, 3.0], b=1.0001)
    b = xlt.lnstensor([4.0, 5.0, 6.0], f=10)  # b = 2 ^ (2 ^ -10) = 1.0006771306930664

Operations
----------

You can perform operations on `LNSTensor` objects just like you would with
PyTorch tensors. The operations will automatically handle the LNS arithmetic.

When performing operations with different precisions or bases, xlnstorch will
attempt to cast the LNSTensors to a common precision or base. The convention,
unless specified otherwise, is to cast to the base of the first LNSTensor involved
in the operation (when looking left to right).

For example, you can add two `LNSTensor` objects:

.. code-block:: python

    a = xlt.lnstensor([1.0, 2.0, 3.0], f=10)
    b = xlt.lnstensor([4.0, 5.0, 6.0], f=16)
    print(a)  # LNSTensor(value=[1.0000, 2.0000, 3.0000], prec=10)
    print(b)  # LNSTensor(value=[4.0000, 5.0000, 6.0000], prec=16)

    c = torch.add(a, b)
    print(c)  # LNSTensor(value=[5.0012, 7.0013, 9.0000], prec=10)

We can see here that the result `c` is not exactly correct, since
we are working with a low precision of 10.

Layers and Models
-----------------

You can also implement layers in xlnstorch, similar to how you would in PyTorch.
The `xlnstorch.nn.LNSModule` class is the base class for all LNS layers and is
analogous to `torch.nn.Module`. See an example below:

.. code-block:: python

    import torch
    import xlnstorch as xlt

    class MyLNSModel(xlt.nn.LNSModule):

        def __init__(self, in_features, hidden_features, out_features):
            super(MyLNSModel, self).__init__()
            self.fc1 = xlt.nn.LNSLinear(in_features, hidden_features)
            self.fc2 = xlt.nn.LNSLinear(hidden_features, out_features)
            self.dropout = torch.nn.Dropout(0.5)

        def forward(self, x):
            x = self.fc1(x)
            x = torch.nn.functional.relu(x)
            x = self.dropout(x)
            x = self.fc2(x)
            return torch.nn.functional.relu(x)

    model = MyLNSModel(10, 20, 1)
    input = xlt.randn(10)

    output = model(input)

Notice how we use the `torch.nn.Dropout` layer as usual, and the `xlnstorch.nn.LNSLinear`
layer. As a general rule, you can use any PyTorch layer that does not contain parameters.
These include layers like `torch.nn.Dropout`, `torch.nn.ReLU`, and others. If a layer has
parameters, you must use the corresponding LNS version, such as `xlnstorch.nn.LNSLinear`
if it exists.

Optimizers
----------

xlnstorch provides many of the same optimizers as PyTorch, such as `SGD`, `Adam`, and others.
These work with `LNSTensor` objects just as the PyTorch optimizers work with `torch.Tensor`
objects, but you must use xlnstorch's versions of the optimizers.

.. note::

    `xlnstorch.nn.LNSModule` provides a method `lns_parameters()` that is analogous to
    `torch.nn.Module.parameters()`. This method returns the LNS parameters of the model
    **and** their bases which is necessary for the optimizers to work correctly.

For example, you can use the `SGD` optimizer as follows:

.. code-block:: python

    import xlnstorch as xlt
    import torch

    model = xlt.nn.LNSLinear(10, 5)
    optimizer = xlt.optim.LNSSGD(model.lns_parameters(), lr=0.01)
    loss_fn = torch.nn.MSELoss()

    input = xlt.randn(10)
    target = xlt.ones(5)

    for i in range(10):
        optimizer.zero_grad()

        output = model(input)
        loss = loss_fn(output, target)

        loss.backward()
        optimizer.step()

        print(f"Iteration {i}, Loss: {loss.item():.2f}")
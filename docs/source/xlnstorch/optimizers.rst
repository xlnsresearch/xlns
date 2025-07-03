Optimizers
==========

The ``xlnstorch.optimizers`` module provides LNS-compatible optimization algorithms
for training neural networks with LNSTensor parameters. These optimizers are
designed to be analogous to the PyTorch optimizers. Currently, the optimizers are
missing support for ``foreach`` and ``fused`` operations, but they are fully functional
for standard LNS training tasks.

Overview
--------

All optimizers in xlnstorch follow the same interface as PyTorch optimizers but
are specifically designed to handle LNSTensor parameters. They support:

* LNSTensor and regular float learning rates and hyperparameters
* Automatic gradient handling for LNS arithmetic
* Parameter groups from LNS layers using ``model.parameter_groups()``
* Standard optimizer features like momentum, weight decay, and adaptive learning rates

Basic Usage
-----------

Here's a basic example of using an LNS optimizer:

.. code-block:: python

    import torch
    import xlnstorch as xltorch

    # Create model and data
    model = xltorch.layers.LNSLinear(3, 3, bias=True)
    input = xltorch.randn(3, requires_grad=True)
    target = xltorch.lnstensor([1.0, 1.0, 1.0])

    # Initialize optimizer with model parameters
    optimizer = xltorch.optimizers.LNSSGD(model.parameter_groups(), lr=0.1)
    loss_fn = torch.nn.MSELoss(reduction='mean')

    # Training loop
    for i in range(20):
        optimizer.zero_grad()

        output = model(input)
        loss = loss_fn(output, target)

        loss.backward()
        optimizer.step()

Parameter Groups
----------------

LNS optimizers work with parameter groups obtained from LNS layers using the
``parameter_groups()`` method. This method returns an iterable of parameter
dictionaries that contain both weights and biases in LNS format.

.. code-block:: python

    # Get parameter groups from a model
    model = xltorch.layers.LNSLinear(10, 5, bias=True)
    param_groups = model.parameter_groups()

    # Initialize optimizer with parameter groups
    optimizer = xltorch.optimizers.LNSAdam(param_groups, lr=0.001)

Learning Rate and Hyperparameters
----------------------------------

All LNS optimizers accept both regular Python floats and LNSTensor objects for
learning rates and other hyperparameters. Using LNSTensor hyperparameters allows
for control over the base of the LNSTensor since other hyperparameters will be
converted to LNSTensors with the default base.

.. code-block:: python

    # Using float learning rate
    optimizer1 = xltorch.optimizers.LNSAdam(params, lr=0.001)

    # Using LNSTensor learning rate
    lr_tensor = xltorch.lnstensor(0.001)
    optimizer2 = xltorch.optimizers.LNSAdam(params, lr=lr_tensor)

Available Optimizers
--------------------

xlnstorch provides LNS-compatible versions of popular PyTorch optimizers:

Quick Reference
~~~~~~~~~~~~~~~

* :ref:`LNSSGD <lnssgd>` - Stochastic Gradient Descent with momentum support
* :ref:`LNSAdam <lnsadam>` - Adam optimizer with bias correction
* :ref:`LNSAdamW <lnsadamw>` - Adam optimizer with decoupled weight decay
* :ref:`LNSAdamax <lnsadamax>` - Adamax optimizer (infinity norm variant of Adam)
* :ref:`LNSNAdam <lnsnadam>` - Nesterov-accelerated Adam optimizer
* :ref:`LNSRAdam <lnsradam>` - Rectified Adam optimizer
* :ref:`LNSAdagrad <lnsadagrad>` - Adaptive gradient algorithm
* :ref:`LNSAdadelta <lnsadadelta>` - Adadelta optimizer
* :ref:`LNSRMSprop <lnsrmsprop>` - RMSprop optimizer
* :ref:`LNSRprop <lnsrprop>` - Resilient backpropagation algorithm
* :ref:`LNSASGD <lnsasgd>` - Averaged Stochastic Gradient Descent

.. _lnssgd:

Stochastic Gradient Descent (SGD)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: xlnstorch.optimizers.LNSSGD
   :members:
   :undoc-members:
   :show-inheritance:

Adam Optimizers
~~~~~~~~~~~~~~~

.. _lnsadam:

.. autoclass:: xlnstorch.optimizers.LNSAdam
   :members:
   :undoc-members:
   :show-inheritance:

.. _lnsadamw:

.. autoclass:: xlnstorch.optimizers.LNSAdamW
   :members:
   :undoc-members:
   :show-inheritance:

.. _lnsadamax:

.. autoclass:: xlnstorch.optimizers.LNSAdamax
   :members:
   :undoc-members:
   :show-inheritance:

.. _lnsnadam:

.. autoclass:: xlnstorch.optimizers.LNSNAdam
   :members:
   :undoc-members:
   :show-inheritance:

.. _lnsradam:

.. autoclass:: xlnstorch.optimizers.LNSRAdam
   :members:
   :undoc-members:
   :show-inheritance:

Adaptive Learning Rate Optimizers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. _lnsadagrad:

.. autoclass:: xlnstorch.optimizers.LNSAdagrad
   :members:
   :undoc-members:
   :show-inheritance:

.. _lnsadadelta:

.. autoclass:: xlnstorch.optimizers.LNSAdadelta
   :members:
   :undoc-members:
   :show-inheritance:

.. _lnsrmsprop:

.. autoclass:: xlnstorch.optimizers.LNSRMSprop
   :members:
   :undoc-members:
   :show-inheritance:

Other Optimizers
~~~~~~~~~~~~~~~~

.. _lnsrprop:

.. autoclass:: xlnstorch.optimizers.LNSRprop
   :members:
   :undoc-members:
   :show-inheritance:

.. _lnsasgd:

.. autoclass:: xlnstorch.optimizers.LNSASGD
   :members:
   :undoc-members:
   :show-inheritance:
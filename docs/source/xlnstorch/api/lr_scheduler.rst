.. currentmodule:: xlnstorch.optim.lr_scheduler

Learning Rate Schedulers
========================

LNS optimizers can be used in conjunction with learning rate schedulers from the
``xlnstorch.optim.lr_scheduler`` module. These schedulers allow for dynamic
adjustment of the learning rate during training, which can help improve convergence
and performance.

LNS learning rate schedulers work in the same was as PyTorch schedulers but are
designed to handle LNSTensor learning rates. They can be applied to any LNS optimizer
and support both float and LNSTensor learning rates. For example:

.. code-block:: python

    import xlnstorch as xltorch

    func = lambda epoch: 1 / (epoch + 1)
    scheduler = xltorch.optim.lr_scheduler.LNSLambdaLR(optimizer, func)

    for epoch in range(100):
        train(...)
        validate(...)
        scheduler.step()

.. autoclass:: LNSLRScheduler
    :members:
    :show-inheritance:

.. autosummary::
    :toctree: generated/optim/lr_scheduler
    :nosignatures:

    LNSLambdaLR
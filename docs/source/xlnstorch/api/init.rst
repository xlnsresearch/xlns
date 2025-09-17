.. currentmodule:: xlnstorch.nn.init

Initializing Module Parameters
==============================

The ``xlnstorch.nn.init`` module provides functions to initialize the parameters
of layers in the ``xlnstorch.nn`` module. These functions modify the input
``LNSTensor`` in-place and are analogous to the ``torch.nn.init`` functions.

.. autosummary::
    :toctree: generated/nn/init

    uniform_
    normal_
    zeros_
    ones_
    constant_
    eye_
    xavier_uniform_
    xavier_normal_
    kaiming_uniform_
    kaiming_normal_
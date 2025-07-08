.. currentmodule:: xlnstorch

.. _layers-doc:

Layers
======

The ``xlnstorch`` package provides a set of layers that support ``LNSTensor``
parameters. Whilst you can use the built-in PyTorch layers with ``LNSTensor``
inputs, these layers will implement standard floating-point parameters.

Note that some of the layers in this module are only added for completeness
since they don't have any parameters. For example, the ``nn.LNSDropout``
layer is equivalent to the standard PyTorch ``torch.nn.Dropout`` layer, both
will work with ``LNSTensor`` inputs, but the ``nn.LNSDropout`` layer is
implemented for completeness. Any layers that do not have parameters will be
denoted in the documentation.

===============================    =====================
Linear Layers                      Note
===============================    =====================
:class:`nn.LNSIdentity`            No parameters
:class:`nn.LNSLinear`
:class:`nn.LNSBilinear`
:class:`nn.LNSLazyLinear`
===============================    =====================

===============================    =====================
Dropout Layers                     Note
===============================    =====================
:class:`nn.LNSDropout`             No parameters
:class:`nn.LNSDropout1d`           No parameters
:class:`nn.LNSDropout2d`           No parameters
:class:`nn.LNSDropout3d`           No parameters
===============================    =====================

===============================    =====================
Convolutional Layers               Note
===============================    =====================
:class:`nn.LNSConv1d`
:class:`nn.LNSConv2d`
:class:`nn.LNSConv3d`
===============================    =====================

.. hide the autosummary table from the main page but still
.. generate the stub files for the layers (this is a hack).

.. raw:: html

    <div style="display: none;">

.. autosummary::
    :toctree: generated
    :nosignatures:

    nn.LNSModule

    nn.LNSIdentity
    nn.LNSLinear
    nn.LNSBilinear
    nn.LNSLazyLinear

    nn.LNSDropout
    nn.LNSDropout1d
    nn.LNSDropout2d
    nn.LNSDropout3d

    nn.LNSConv1d
    nn.LNSConv2d
    nn.LNSConv3d

.. raw:: html

    </div>

Custom Layers
-------------

To implement your own custom layers that support ``LNSTensor`` parameters,
you can subclass the base layer class provided in this module :class:`nn.LNSModule`.

This base class is a subclass of the standard PyTorch ``torch.nn.Module`` and
provides the method :func:`nn.LNSModule.register_parameter` which is equivalent
to PyTorch's method of registering parameters.
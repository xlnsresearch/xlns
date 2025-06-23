.. currentmodule:: xlnstorch

.. _layers-doc:

Layers
======

The ``xlnstorch`` package provides a set of layers that support ``LNSTensor``
parameters. Whilst you can use the built-in PyTorch layers with ``LNSTensor``
inputs, these layers will implement standard floating-point parameters.

Note that some of the layers in this module are only added for completeness
since they don't have any parameters. For example, the ``layers.LNSDropout``
layer is equivalent to the standard PyTorch ``torch.nn.Dropout`` layer, both
will work with ``LNSTensor`` inputs, but the ``layers.LNSDropout`` layer is
implemented for completeness. Any layers that do not have parameters will be
denoted in the documentation.

===============================    =====================
Linear Layers                      Note
===============================    =====================
:class:`layers.LNSIdentity`        No parameters
:class:`layers.LNSLinear`
:class:`layers.LNSBilinear`
:class:`layers.LNSLazyLinear`
===============================    =====================

===============================    =====================
Dropout Layers                     Note
===============================    =====================
:class:`layers.LNSDropout`         No parameters
:class:`layers.LNSDropout1d`       No parameters
:class:`layers.LNSDropout2d`       No parameters
:class:`layers.LNSDropout3d`       No parameters
===============================    =====================

===============================    =====================
Convolutional Layers               Note
===============================    =====================
:class:`layers.LNSConv1d`
===============================    =====================

.. hide the autosummary table from the main page but still
.. generate the stub files for the layers (this is a hack).

.. raw:: html

   <div style="display: none;">

.. autosummary::
    :toctree: generated
    :nosignatures:

    layers.LNSModule

    layers.LNSIdentity
    layers.LNSLinear
    layers.LNSBilinear
    layers.LNSLazyLinear

    layers.LNSDropout
    layers.LNSDropout1d
    layers.LNSDropout2d
    layers.LNSDropout3d

    layers.LNSConv1d

.. raw:: html

   </div>

Custom Layers
-------------

To implement your own custom layers that support ``LNSTensor`` parameters,
you can subclass the base layer class provided in this module :class:`layers.LNSModule`.

This base class is a subclass of the standard PyTorch ``torch.nn.Module`` and
provides the method :func:`layers.LNSModule.register_parameter` which is equivalent
to PyTorch's method of registering parameters.
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
Layer                              Note
===============================    =====================
:class:`layers.LNSIdentity`        No parameters
:class:`layers.LNSLinear`
:class:`layers.LNSBilinear`
:class:`layers.LNSLazyLinear`

:class:`layers.LNSDropout`         No parameters
:class:`layers.LNSDropout1d`       No parameters
:class:`layers.LNSDropout2d`       No parameters
:class:`layers.LNSDropout3d`       No parameters

:class:`layers.LNSConv1d`
===============================    =====================

.. hide the autosummary table from the main page but still
.. generate the stub files for the layers (this is a hack).

.. raw:: html

   <div style="display: none;">

.. autosummary::
    :toctree: generated
    :nosignatures:

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
.. currentmodule:: xlnstorch.nn

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
:class:`LNSIdentity`               No parameters
:class:`LNSLinear`
:class:`LNSBilinear`
:class:`LNSLazyLinear`
===============================    =====================

===============================    =====================
Dropout Layers                     Note
===============================    =====================
:class:`LNSDropout`                No parameters
:class:`LNSDropout1d`              No parameters
:class:`LNSDropout2d`              No parameters
:class:`LNSDropout3d`              No parameters
===============================    =====================

===============================    =====================
Convolutional Layers               Note
===============================    =====================
:class:`LNSConv1d`
:class:`LNSConv2d`
:class:`LNSConv3d`
===============================    =====================

===============================    =====================
Normalization Layers               Note
===============================    =====================
:class:`LNSBatchNorm1d`
:class:`LNSBatchNorm2d`
:class:`LNSBatchNorm3d`
:class:`LNSLayerNorm`
===============================    =====================

===============================    =====================
Pooling Layers                     Note
===============================    =====================
:class:`LNSAvgPool1d`              No parameters
:class:`LNSAvgPool2d`              No parameters
:class:`LNSAvgPool3d`              No parameters
:class:`LNSAdaptiveAvgPool1d`      No parameters
:class:`LNSAdaptiveAvgPool2d`      No parameters
:class:`LNSAdaptiveAvgPool3d`      No parameters
:class:`LNSMaxPool1d`              No parameters
:class:`LNSMaxPool2d`              No parameters
:class:`LNSMaxPool3d`              No parameters
===============================    =====================

===============================    =====================
Recurrent Layers                   Note
===============================    =====================
:class:`LNSRNN`
:class:`LNSRNNCell`
:class:`LNSLSTM`
:class:`LNSLSTMCell`
:class:`LNSGRU`
:class:`LNSGRUCell`
===============================    =====================

.. hide the autosummary table from the main page but still
.. generate the stub files for the layers (this is a hack).

Custom Layers
-------------

To implement your own custom layers that support ``LNSTensor`` parameters,
you can subclass the base layer class provided in this module :class:`LNSModule`.

This base class is a subclass of the standard PyTorch ``torch.nn.Module`` and
provides the method :func:`LNSModule.register_parameter` which is equivalent
to PyTorch's method of registering parameters.

.. raw:: html

    <div style="display: none;">

.. autosummary::
    :toctree: generated/nn
    :nosignatures:

    LNSModule

    LNSIdentity
    LNSLinear
    LNSBilinear
    LNSLazyLinear

    LNSDropout
    LNSDropout1d
    LNSDropout2d
    LNSDropout3d

    LNSConv1d
    LNSConv2d
    LNSConv3d

    LNSBatchNorm1d
    LNSBatchNorm2d
    LNSBatchNorm3d

    LNSAvgPool1d
    LNSAvgPool2d
    LNSAvgPool3d
    LNSAdaptiveAvgPool1d
    LNSAdaptiveAvgPool2d
    LNSAdaptiveAvgPool3d
    LNSMaxPool1d
    LNSMaxPool2d
    LNSMaxPool3d

    LNSRNN
    LNSRNNCell
    LNSLSTM
    LNSLSTMCell
    LNSGRU
    LNSGRUCell

.. raw:: html

    </div>
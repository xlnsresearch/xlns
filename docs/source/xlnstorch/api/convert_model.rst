.. currentmodule:: xlnstorch.convert_model

Convert FP Models
=================

The ``xlnstorch.convert_model`` submodule provides utilities to convert
torch models and layers to their analogues in xlnstorch. This is an
experimental feature and is a work in progress. The range of supported
layers and models is limited, and the conversion may not always be perfect.

.. note::

    Currently, only sequential models (or normal modules that are defined
    squentially) are supported for conversion.

.. autosummary::
    :toctree: generated/convert_model
    :nosignatures:

    parse_sequential
    build_lns_sequential
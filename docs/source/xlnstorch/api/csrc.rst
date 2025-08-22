.. currentmodule:: xlnstorch.csrc

C++ Extensions
==============

The C++ extensions in ``xlnstorch.csrc`` provide optimized implementations
of various operations and utilities that are used in xlnstorch. These are
only available if you have built the C++ extensions which is done automatically
when you install xlnstorch (if you have a compatible C++ compiler). If the
extensions are not built, xlnstorch will fall back to pure Python implementations
for those operations, which may be slower but will still work correctly.

To JIT compile and load the C++ extensions at runtime, you can call the
`xlnstorch.csrc.load_backend()` function.

.. autosummary::
    :toctree: generated/csrc

    load_backend

You can verify if the C++ extensions are available by checking the
``xlnstorch.CSRC_AVAILABLE`` flag. If it is set to `True`, you can import
and use the ``xlnstorch.csrc`` module. To switch between the C++ and Python
implementations, you can call the `xlnstorch.operators.toggle_cpp_implementations()`
function.

The following functions are available in the `xlnstorch.csrc` module. They should
not be called directly by users, but if necessary, the following docs are provided
for reference:

.. autosummary::
    :toctree: generated/csrc
    
    float_to_lns_forward
    float_to_lns_backward
    change_base_forward
    change_base_backward
    set_default_sbdb_implementation
    get_table
    add_forward
    sum_forward
    matmul_forward
    matmul_backward
    conv1d_forward
    conv1d_backward
    conv2d_forward
    conv2d_backward
    conv3d_forward
    conv3d_backward
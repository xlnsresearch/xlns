C++ Extensions
==============

The C++ extensions in ``xlnstorch.csrc`` provide optimized implementations
of various operations and utilities that are used in xlnstorch. These are
only available if you have built the C++ extensions which is done automatically
when you install xlnstorch (if you have a compatible C++ compiler). If the
extensions are not built, xlnstorch will fall back to pure Python implementations
for those operations, which may be slower but will still work correctly.

You can verify if the C++ extensions are available by checking the
``xlnstorch.CSRC_AVAILABLE`` flag. If it is set to `True`, you can import
and use the ``xlnstorch.csrc`` module. To switch between the C++ and Python
implementations, you can call the `xlnstorch.operators.toggle_cpp_implementations()`
function.
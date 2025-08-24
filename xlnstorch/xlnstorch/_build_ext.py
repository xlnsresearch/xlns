"""
Custom build_ext command that turns a failed C++ build into a friendly
warning instead of a hard error.
"""
import textwrap
import warnings
from pathlib import Path

from torch.utils.cpp_extension import CppExtension, BuildExtension, include_paths

# The exact exception raised by a compiler/linker failure can vary
# between setuptools versions, so we catch the whole family.
try:
    # setuptools ≥ 68
    from setuptools.errors import (
        CompileError,
        LinkError,
        PlatformError,
    )
except ImportError:                          # setuptools < 68
    from distutils.errors import (           # type: ignore
        CompileError,
        LinkError,
        DistutilsPlatformError as PlatformError,
    )


class OptionalBuildExtension(BuildExtension):
    """
    BuildExtension that Globs all *.cpp in xlnstorch/_C and
    plugs them into ext.sources. If compilation/linking fails,
    shows a warning instead of aborting the installation.
    """

    _BOX = """
    ************************************************************
       xlnstorch: C++ backend could NOT be built
       falling back to the (slower) pure-python implementation
 
       Reason: {reason}                                        
    ************************************************************
    """

    def run(self) -> None:
        try:
            super().run()
        except (CompileError, LinkError, PlatformError) as exc:
            # Format and emit a visible message
            msg = textwrap.dedent(self._BOX).format(reason=exc)
            self.announce(msg, level=3)      # level=3 == WARNING
            # Also raise a RuntimeWarning so that it is visible when
            # pip is invoked with low verbosity.
            warnings.warn(
                "C++ backend for xlnstorch was not built; "
                "installation will succeed but will run in pure-python mode.",
                RuntimeWarning,
            )
            # swallow the error → installation continues

    def build_extensions(self):
        src_dir = Path(__file__).resolve().parent / "csrc"
        cpp_files = [str(p) for p in src_dir.glob("*.cpp")]

        torch_includes = include_paths()

        for ext in self.extensions:
            if ext.name == "xlnstorch._csrc": # only our extension
                ext.sources = cpp_files
            # add torch include directories to the extension
            ext.include_dirs = list(ext.include_dirs or []) + torch_includes

        super().build_extensions()
import importlib
import types

def test_package_import():
    """Package should be importable."""
    module = importlib.import_module("xlnstorch")
    assert isinstance(module, types.ModuleType)

def test_version_attribute():
    """
    If xlnstorch defines __version__, make sure it's a non-empty string.
    The test is skipped if the attribute is absent.
    """
    module = importlib.import_module("xlnstorch")

    if hasattr(module, "__version__"):
        assert isinstance(module.__version__, str) and module.__version__
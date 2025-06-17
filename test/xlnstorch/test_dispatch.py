import pytest
import torch
import xlnstorch as xltorch

def test_implementation_registration():
    """Test registering and retrieving implementations."""
    # Test registering a new implementation
    @xltorch.implements(torch.add, None, key="test_impl")
    def test_add_impl(x, y, *, alpha=1, out=None):
        return xltorch.lnstensor(x.value + alpha * y.value, b=x.base)

    # Test retrieving the implementation
    impl = xltorch.get_implementation(torch.add, "test_impl")
    assert impl is not None
    assert impl[0] is test_add_impl

    # Test setting as default
    with xltorch.override_implementation(torch.add, "test_impl"):
        default_key = xltorch.get_default_implementation_key(torch.add)
    assert default_key == "test_impl"

def test_implementation_override():
    """Test overriding existing implementations."""
    # Register initial implementation
    @xltorch.implements(torch.add, None, key="impl1")
    def add_impl1(x, y, *, alpha=1, out=None):
        return xltorch.lnstensor(x.value + y.value, b=x.base)

    # Override with new implementation
    @xltorch.implements(torch.add, None, key="impl1")
    def add_impl2(x, y, *, alpha=1, out=None):
        return xltorch.lnstensor(x.value + y.value, b=x.base)

    # Check that the override worked
    impl = xltorch.get_implementation(torch.add, "impl1")
    assert impl[0] is add_impl2

def test_apply_lns_op():
    """Test the apply_lns_op function."""
    data1 = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
    data2 = torch.tensor([4.0, 5.0, 6.0], dtype=torch.float64)
    lns1 = xltorch.lnstensor(data1, f=23)
    lns2 = xltorch.lnstensor(data2, f=23)

    # Test with default implementation
    result = xltorch.apply_lns_op(torch.add, lns1._lns, lns2._lns, lns1.base)
    result_lns = xltorch.lnstensor(result, from_lns=True, b=lns1.base)
    expected = lns1.value + lns2.value
    assert torch.allclose(result_lns.value, expected)

    # Test with specific implementation
    internal_comp = lambda x, y, base: torch.zeros_like(x) # dummy implementation
    @xltorch.implements(torch.add, internal_comp, key="test_impl")
    def test_add_impl(x, y, *, alpha=1, out=None):
        return xltorch.lnstensor(x.value + alpha * y.value, b=x.base)

    with xltorch.override_implementation(torch.add, "test_impl"):
        result = xltorch.apply_lns_op(torch.add, lns1._lns, lns2._lns, lns1.base)

def test_invalid_implementation():
    """Test behavior with invalid implementation keys."""
    with pytest.raises(ValueError):
        xltorch.get_implementation(torch.add, "nonexistent_key")

    with pytest.raises(ValueError):
        xltorch.set_default_implementation(torch.add, "nonexistent_key")
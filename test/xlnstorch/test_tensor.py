import pytest
import torch
import xlnstorch as xltorch

def test_lnstensor_creation():
    """Test basic LNSTensor creation and properties."""
    # Test creation from regular tensor
    data = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
    lns = xltorch.lnstensor(data, f=23)
    base = torch.tensor(2.0 ** (2.0 ** -23), dtype=torch.float64)

    assert isinstance(lns, xltorch.LNSTensor)
    assert lns.base == base
    assert torch.allclose(lns.value, data)

    # Test creation with different base
    base2 = torch.tensor(1.01, dtype=torch.float64)
    lns2 = xltorch.lnstensor(data, b=base2)
    assert lns2.base == base2

    # Test creation with f parameter
    lns3 = xltorch.lnstensor(data, f=4)
    assert isinstance(lns3.base, torch.Tensor)
    assert lns3.base.dtype == torch.float64

def test_lnstensor_invalid_inputs():
    """Test LNSTensor creation with invalid inputs."""
    # Test invalid base
    with pytest.raises(ValueError):
        xltorch.lnstensor(torch.tensor([1.0, 2.0], dtype=torch.float64), b=0.0)

    with pytest.raises(ValueError):
        xltorch.lnstensor(torch.tensor([1.0, 2.0], dtype=torch.float64), b=1.0)

    # Test conflicting base parameters
    with pytest.raises(ValueError):
        xltorch.lnstensor(torch.tensor([1.0, 2.0], dtype=torch.float64), f=4, b=2.0)

def test_lnstensor_encoding_decoding():
    """Test that encoding and decoding preserves values."""
    data = torch.tensor([1.0, 2.0, 3.0, -1.0, -2.0, -3.0], dtype=torch.float64)
    lns = xltorch.lnstensor(data, f=23)

    # Test that values are preserved
    assert torch.allclose(lns.value, data)

    # Test with different base
    lns2 = xltorch.lnstensor(data, f=10)
    assert torch.allclose(lns2.value, data)

def test_lnstensor_gradients():
    """Test gradient computation and backpropagation."""
    data1 = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64, requires_grad=True)
    data2 = torch.tensor([4.0, 5.0, 6.0], dtype=torch.float64, requires_grad=True)
    lns1 = xltorch.lnstensor(data1, f=23)
    lns2 = xltorch.lnstensor(data2, f=23)

    # Test forward pass
    print(xltorch.get_default_implementation_key(torch.add))
    result = torch.add(lns1, lns2)
    result.backward()

    # Check that gradients are computed
    print(lns1.grad, lns2.grad)
    assert lns1.grad is not None and lns2.grad is not None
    assert torch.allclose(lns1.grad.value, torch.ones_like(data1))
    assert torch.allclose(lns2.grad.value, torch.ones_like(data2))

def test_lnstensor_repr():
    """Test string representation of LNSTensor."""
    data = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
    base = torch.tensor(2.0, dtype=torch.float64)
    lns = xltorch.lnstensor(data, b=base)

    repr_str = repr(lns)
    assert "LNSTensor" in repr_str
    assert "value=" in repr_str
    assert "base=" in repr_str 
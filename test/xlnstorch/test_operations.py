import pytest
import torch
import xlnstorch as xltorch

@pytest.fixture
def sample_tensors():
    """Create sample tensors for testing."""
    data1 = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
    data2 = torch.tensor([4.0, 5.0, 6.0], dtype=torch.float64)
    return xltorch.lnstensor(data1, f=23), xltorch.lnstensor(data2, f=18)

def test_lns_add(sample_tensors):
    """Test LNS addition."""
    lns1, lns2 = sample_tensors

    # Test basic addition
    result = torch.add(lns1, lns2)
    expected = lns1.value + lns2.value
    assert torch.allclose(result.value, expected)

    # Test with alpha
    alpha = 2.0
    result = torch.add(lns1, lns2, alpha=alpha)
    expected = lns1.value + alpha * lns2.value
    assert torch.allclose(result.value, expected)

    # Test with regular tensor
    result = torch.add(lns1, lns2.value)
    expected = lns1.value + lns2.value
    assert torch.allclose(result.value, expected)

def test_lns_sub(sample_tensors):
    """Test LNS subtraction."""
    lns1, lns2 = sample_tensors

    # Test basic subtraction
    result = torch.sub(lns1, lns2)
    expected = lns1.value - lns2.value
    assert torch.allclose(result.value, expected)

    # Test with alpha
    alpha = 2.0
    result = torch.sub(lns1, lns2, alpha=alpha)
    expected = lns1.value - alpha * lns2.value
    assert torch.allclose(result.value, expected)

def test_lns_mul(sample_tensors):
    """Test LNS multiplication."""
    lns1, lns2 = sample_tensors

    # Test basic multiplication
    result = torch.mul(lns1, lns2)
    expected = lns1.value * lns2.value
    assert torch.allclose(result.value, expected)

    # Test with regular tensor
    result = torch.mul(lns1, lns2.value)
    expected = lns1.value * lns2.value
    assert torch.allclose(result.value, expected)

def test_lns_div(sample_tensors):
    """Test LNS division."""
    lns1, lns2 = sample_tensors

    # Test basic division
    result = torch.div(lns1, lns2)
    expected = lns1.value / lns2.value
    assert torch.allclose(result.value, expected)

    # Test with regular tensor
    result = torch.div(lns1, lns2.value)
    expected = lns1.value / lns2.value
    assert torch.allclose(result.value, expected)

def test_lns_unary_ops(sample_tensors):
    """Test unary LNS operations."""
    lns1, _ = sample_tensors

    # Test negation
    result = torch.neg(lns1)
    expected = -lns1.value
    assert torch.allclose(result.value, expected)

    # Test absolute value
    result = torch.abs(lns1)
    expected = torch.abs(lns1.value)
    assert torch.allclose(result.value, expected)

    # Test square root
    result = torch.sqrt(lns1)
    expected = torch.sqrt(lns1.value)
    assert torch.allclose(result.value, expected)

    # Test square
    result = torch.square(lns1)
    expected = lns1.value ** 2
    assert torch.allclose(result.value, expected)

    # Test reciprocal
    result = torch.reciprocal(lns1)
    expected = 1.0 / lns1.value
    assert torch.allclose(result.value, expected)

    # Test sign
    result = torch.sign(lns1)
    expected = torch.sign(lns1.value)
    assert torch.allclose(result.value, expected)

def test_lns_pow(sample_tensors):
    """Test LNS power operation."""
    lns1, _ = sample_tensors

    # Test with integer exponent
    result = torch.pow(lns1, 2)
    expected = lns1.value ** 2
    assert torch.allclose(result.value, expected)

    # Test with float exponent
    result = torch.pow(lns1, 0.5)
    expected = lns1.value ** 0.5
    assert torch.allclose(result.value, expected)
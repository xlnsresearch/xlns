import pytest
import torch
import xlnstorch as xltorch

@pytest.fixture
def sample_tensors():
    """Create sample tensors for testing."""
    data1 = xltorch.lnstensor([1.0, 2.0, 3.0], f=23)
    data2 = xltorch.lnstensor([4.0, 5.0, 6.0], f=18)
    return data1, data2

@pytest.fixture
def edge_case_tensors():
    """Create tensors with edge cases for testing."""
    data1 = xltorch.lnstensor([0.0, -1.0, 1e-10, 1e10], f=23)
    data2 = xltorch.lnstensor([1.0, -2.0, 1e-8, 1e8], f=28)
    return data1, data2

def verify_lns_result(result, expected, msg=None):
    """Helper function to verify LNS tensor results."""
    assert isinstance(result, xltorch.LNSTensor), "Result should be an LNS tensor"
    assert torch.allclose(result.value, expected, rtol=1e-5), msg or "Values don't match expected results"

class TestLNSAddition:
    """Tests for LNS addition operations."""

    def test_basic_addition(self, sample_tensors):
        """Test basic LNS addition."""
        lns1, lns2 = sample_tensors
        result = torch.add(lns1, lns2)
        expected = lns1.value + lns2.value
        verify_lns_result(result, expected, "Basic addition failed")

        # Ensure original base is preserved
        assert result.base == lns1.base, "f-parameter should be preserved from first operand"

    def test_addition_with_alpha(self, sample_tensors):
        """Test LNS addition with alpha parameter."""
        lns1, lns2 = sample_tensors
        alpha = 2.0
        result = torch.add(lns1, lns2, alpha=alpha)
        expected = lns1.value + alpha * lns2.value
        verify_lns_result(result, expected, "Addition with alpha failed")

    def test_addition_with_regular_tensor(self, sample_tensors):
        """Test LNS addition with regular tensor."""
        lns1, lns2 = sample_tensors

        # LNS + regular
        result = torch.add(lns1, lns2.value)
        expected = lns1.value + lns2.value
        verify_lns_result(result, expected, "LNS + regular tensor failed")

        # regular + LNS
        result = torch.add(lns1.value, lns2)
        expected = lns1.value + lns2.value
        verify_lns_result(result, expected, "Regular tensor + LNS failed")

    def test_inplace_addition(self, sample_tensors):
        """Test inplace LNS addition."""
        lns1, lns2 = sample_tensors
        original_value = lns1.value.clone()
        lns1_copy = xltorch.lnstensor(lns1._lns.clone(), from_lns=True, b=lns1.base)
        lns1_copy.add_(lns2)
        expected = original_value + lns2.value
        verify_lns_result(lns1_copy, expected, "Inplace addition failed")

    def test_edge_cases(self, edge_case_tensors):
        """Test LNS addition with edge cases."""
        lns1, lns2 = edge_case_tensors
        result = torch.add(lns1, lns2)
        expected = lns1.value + lns2.value
        verify_lns_result(result, expected, "Edge case addition failed")

class TestLNSSubtraction:
    """Tests for LNS subtraction operations."""

    def test_basic_subtraction(self, sample_tensors):
        """Test basic LNS subtraction."""
        lns1, lns2 = sample_tensors
        result = torch.sub(lns1, lns2)
        expected = lns1.value - lns2.value
        verify_lns_result(result, expected, "Basic subtraction failed")

    def test_subtraction_with_alpha(self, sample_tensors):
        """Test LNS subtraction with alpha parameter."""
        lns1, lns2 = sample_tensors
        alpha = 2.0
        result = torch.sub(lns1, lns2, alpha=alpha)
        expected = lns1.value - alpha * lns2.value
        verify_lns_result(result, expected, "Subtraction with alpha failed")

    def test_subtraction_with_regular_tensor(self, sample_tensors):
        """Test LNS subtraction with regular tensor."""
        lns1, lns2 = sample_tensors

        # LNS - regular
        result = torch.sub(lns1, lns2.value)
        expected = lns1.value - lns2.value
        verify_lns_result(result, expected, "LNS - regular tensor failed")

        # regular - LNS
        result = torch.sub(lns1.value, lns2)
        expected = lns1.value - lns2.value
        verify_lns_result(result, expected, "Regular tensor - LNS failed")

    def test_inplace_subtraction(self, sample_tensors):
        """Test inplace LNS subtraction."""
        lns1, lns2 = sample_tensors
        original_value = lns1.value.clone()
        lns1_copy = xltorch.lnstensor(lns1._lns.clone(), from_lns=True, b=lns1.base)
        lns1_copy.sub_(lns2)
        expected = original_value - lns2.value
        verify_lns_result(lns1_copy, expected, "Inplace subtraction failed")

@pytest.mark.parametrize("op_name,op_func,regular_op", [
    ("multiplication", torch.mul, lambda x, y: x * y),
    ("division", torch.div, lambda x, y: x / y),
])
class TestLNSBinaryOps:
    """Parameterized tests for binary operations."""

    def test_basic_operation(self, sample_tensors, op_name, op_func, regular_op):
        """Test basic operation."""
        lns1, lns2 = sample_tensors
        result = op_func(lns1, lns2)
        expected = regular_op(lns1.value, lns2.value)
        verify_lns_result(result, expected, f"Basic {op_name} failed")

    def test_with_regular_tensor(self, sample_tensors, op_name, op_func, regular_op):
        """Test operation with regular tensor."""
        lns1, lns2 = sample_tensors

        # LNS op regular
        result = op_func(lns1, lns2.value)
        expected = regular_op(lns1.value, lns2.value)
        verify_lns_result(result, expected, f"LNS {op_name} regular tensor failed")

        # regular op LNS
        result = op_func(lns1.value, lns2)
        expected = regular_op(lns1.value, lns2.value) 
        verify_lns_result(result, expected, f"Regular tensor {op_name} LNS failed")

    def test_inplace_operation(self, sample_tensors, op_name, op_func, regular_op):
        """Test inplace operation."""
        lns1, lns2 = sample_tensors
        original_value = lns1.value.clone()
        lns1_copy = xltorch.lnstensor(lns1._lns.clone(), from_lns=True, b=lns1.base)

        # Call the inplace version using getattr
        inplace_op = getattr(lns1_copy, f"{op_func.__name__}_")
        inplace_op(lns2)

        expected = regular_op(original_value, lns2.value)
        verify_lns_result(lns1_copy, expected, f"Inplace {op_name} failed")

@pytest.mark.parametrize("op_name,op_func,regular_op", [
    ("negation", torch.neg, lambda x: -x),
    ("absolute_value", torch.abs, torch.abs),
    ("square_root", torch.sqrt, torch.sqrt),
    ("square", torch.square, lambda x: x ** 2),
    ("reciprocal", torch.reciprocal, lambda x: 1.0 / x),
    ("sign", torch.sign, torch.sign),
])
def test_unary_ops(sample_tensors, op_name, op_func, regular_op):
    """Test unary operations."""
    lns1, _ = sample_tensors
    result = op_func(lns1)
    expected = regular_op(lns1.value)
    verify_lns_result(result, expected, f"{op_name} operation failed")

class TestLNSPower:
    """Tests for LNS power operations."""

    @pytest.mark.parametrize("exponent,description", [
        (2, "integer exponent"),
        (0.5, "float exponent"),
        (0, "zero exponent"),
        (-1, "negative exponent"),
    ])
    def test_power_operation(self, sample_tensors, exponent, description):
        """Test power operation with various exponents."""
        lns1, _ = sample_tensors
        result = torch.pow(lns1, exponent)
        expected = lns1.value ** exponent
        verify_lns_result(result, expected, f"Power operation with {description} failed")

    def test_power_with_tensor_exponent(self, sample_tensors):
        """Test power operation with tensor exponent."""
        lns1, lns2 = sample_tensors
        exponents = torch.tensor([0.5, 1.0, 2.0], dtype=torch.float64)
        result = torch.pow(lns1, exponents)
        expected = lns1.value ** exponents
        verify_lns_result(result, expected, "Power operation with tensor exponent failed")
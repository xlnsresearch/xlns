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

class TestLNSSum:
    """Tests for LNS sum operation."""

    def test_basic_sum(self, sample_tensors):
        """Test basic sum over all elements."""
        lns1, _ = sample_tensors
        result = torch.sum(lns1)
        expected = torch.sum(lns1.value)

        assert isinstance(result, xltorch.LNSTensor), "Sum result should be an LNS tensor"
        assert torch.allclose(result.value, expected), "Basic sum failed"

    def test_dim_sum(self):
        """Test sum along specific dimensions."""
        lns = xltorch.lnstensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], f=23)

        # Sum along dimension 0
        result = torch.sum(lns, dim=0)
        expected = torch.sum(lns.value, dim=0)
        assert isinstance(result, xltorch.LNSTensor), "Sum result should be an LNS tensor"
        assert torch.allclose(result.value, expected), "Sum along dimension 0 failed"
        assert result.shape == expected.shape, "Sum shape mismatch"

        # Sum along dimension 1
        result = torch.sum(lns, dim=1)
        expected = torch.sum(lns.value, dim=1)
        assert isinstance(result, xltorch.LNSTensor), "Sum result should be an LNS tensor"
        assert torch.allclose(result.value, expected), "Sum along dimension 1 failed"
        assert result.shape == expected.shape, "Sum shape mismatch"

    def test_keepdim_sum(self):
        """Test sum with keepdim=True."""
        lns = xltorch.lnstensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], f=23)

        # Sum along dimension 0 with keepdim=True
        result = torch.sum(lns, dim=0, keepdim=True)
        expected = torch.sum(lns.value, dim=0, keepdim=True)
        assert isinstance(result, xltorch.LNSTensor), "Sum result should be an LNS tensor"
        assert torch.allclose(result.value, expected), "Sum with keepdim failed"
        assert result.shape == expected.shape, "Sum with keepdim shape mismatch"

    def test_multi_dim_sum(self):
        """Test sum over multi-dimensional tensors."""
        lns = xltorch.lnstensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], f=23)

        # Sum along multiple dimensions
        result = torch.sum(lns, dim=(0, 2))
        expected = torch.sum(lns.value, dim=(0, 2))
        assert isinstance(result, xltorch.LNSTensor), "Sum result should be an LNS tensor"
        assert torch.allclose(result.value, expected), "Sum along multiple dimensions failed"
        assert result.shape == expected.shape, "Sum along multiple dimensions shape mismatch"

        # Sum along negative dimension
        result = torch.sum(lns, dim=-1)
        expected = torch.sum(lns.value, dim=-1)
        assert isinstance(result, xltorch.LNSTensor), "Sum result should be an LNS tensor"
        assert torch.allclose(result.value, expected), "Sum along negative dimension failed"
        assert result.shape == expected.shape, "Sum along negative dimension shape mismatch"

    def test_edge_cases_sum(self, edge_case_tensors):
        """Test sum with edge case values."""
        lns, _ = edge_case_tensors
        result = torch.sum(lns)
        expected = torch.sum(lns.value)
        assert isinstance(result, xltorch.LNSTensor), "Sum result should be an LNS tensor"
        assert torch.allclose(result.value, expected), "Sum with edge cases failed"

class TestLNSMatmul:
    """Tests for LNS matrix multiplication operation."""

    def test_matrix_multiplication(self):
        """Test basic matrix multiplication."""
        A = xltorch.lnstensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], f=23)
        B = xltorch.lnstensor([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]], f=23)

        result = torch.matmul(A, B)
        expected = torch.matmul(A.value, B.value)

        assert isinstance(result, xltorch.LNSTensor), "Matmul result should be an LNS tensor"
        assert torch.allclose(result.value, expected), "Matrix multiplication failed"
        assert result.shape == (2, 2), "Matrix multiplication shape mismatch"

    def test_batched_matrix_multiplication(self):
        """Test batched matrix multiplication."""
        batch_A = xltorch.lnstensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], f=23)
        batch_B = xltorch.lnstensor([[[9.0, 10.0], [11.0, 12.0]], [[13.0, 14.0], [15.0, 16.0]]], f=23)

        result = torch.matmul(batch_A, batch_B)
        expected = torch.matmul(batch_A.value, batch_B.value)

        assert isinstance(result, xltorch.LNSTensor), "Batched matmul result should be an LNS tensor"
        assert torch.allclose(result.value, expected), "Batched matrix multiplication failed"
        assert result.shape == (2, 2, 2), "Batched matrix multiplication shape mismatch"

    def test_broadcast_matrix_multiplication(self):
        """Test broadcast matrix multiplication."""
        A = xltorch.lnstensor([1.0, 2.0, 3.0], f=23)
        batch_B = xltorch.lnstensor([[4.0, 5.0], [6.0, 7.0], [8.0, 9.0]], f=23)
        result = torch.matmul(A, batch_B)
        expected = torch.matmul(A.value, batch_B.value)

        assert isinstance(result, xltorch.LNSTensor), "Broadcast matmul result should be an LNS tensor"
        assert torch.allclose(result.value, expected), "Broadcast matrix multiplication failed"
        assert result.shape == expected.shape, "Broadcast matrix multiplication shape mismatch"

    def test_different_base_parameters(self):
        """Test matrix multiplication with different base parameters."""
        A = xltorch.lnstensor([[1.0, 2.0], [3.0, 4.0]], f=23)
        B = xltorch.lnstensor([[5.0, 6.0], [7.0, 8.0]], f=28)

        result = torch.matmul(A, B)
        expected = torch.matmul(A.value, B.value)

        assert isinstance(result, xltorch.LNSTensor), "Matmul result should be an LNS tensor"
        assert torch.allclose(result.value, expected, rtol=1e-5), "Matrix multiplication with different bases failed"
        assert result.base == A.base, "Result base should match first tensor"

    def test_mixed_tensor_types(self):
        """Test matrix multiplication with one regular tensor."""
        A = xltorch.lnstensor([[1.0, 2.0], [3.0, 4.0]], f=23)
        B = torch.tensor([[5.0, 6.0], [7.0, 8.0]], dtype=torch.float64)

        # LNS @ regular
        result = torch.matmul(A, B)
        expected = torch.matmul(A.value, B)

        assert isinstance(result, xltorch.LNSTensor), "Mixed matmul result should be an LNS tensor"
        assert torch.allclose(result.value, expected), "LNS @ regular matrix multiplication failed"

        # regular @ LNS
        result = torch.matmul(B, A)
        expected = torch.matmul(B, A.value)

        assert isinstance(result, xltorch.LNSTensor), "Mixed matmul result should be an LNS tensor"
        assert torch.allclose(result.value, expected), "regular @ LNS matrix multiplication failed"
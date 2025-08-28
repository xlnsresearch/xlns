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
def identical_tensors():
    """Create identical tensors for equality testing."""
    data = xltorch.lnstensor([1.0, 2.0, 3.0], f=23)
    return data, xltorch.lnstensor(data._lns.clone(), from_lns=True, b=data.base)

@pytest.fixture
def edge_case_tensors():
    """Create tensors with edge cases for testing."""
    data1 = xltorch.lnstensor([0.0, -1.0, 1e-10, 1e10], f=23)
    data2 = xltorch.lnstensor([0.0, -2.0, 1e-8, 1e8], f=28)
    return data1, data2

@pytest.fixture
def mixed_sign_tensors():
    """Create tensors with mixed signs for comparison testing."""
    data1 = xltorch.lnstensor([1.0, -1.0, 2.0, -2.0], f=23)
    data2 = xltorch.lnstensor([1.0, -1.0, 1.0, -3.0], f=18)
    return data1, data2

def verify_bool_result(result, expected, msg=None):
    """Helper function to verify boolean tensor results."""
    assert isinstance(result, torch.Tensor), "Result should be a tensor"
    assert result.dtype == torch.bool, "Result should be a boolean tensor"
    assert torch.all(result == expected), msg or "Boolean results don't match expected"

class TestLNSEquality:
    """Tests for LNS equality operations."""

    def test_equal(self, identical_tensors, sample_tensors):
        """Test torch.equal operation."""
        # Test with identical tensors
        lns1, lns2 = identical_tensors
        result = torch.equal(lns1, lns2)
        assert result is True, "Equal tensors should return True"

        # Test with different tensors
        lns1, lns2 = sample_tensors
        result = torch.equal(lns1, lns2)
        assert result is False, "Different tensors should return False"

    def test_eq(self, identical_tensors, sample_tensors):
        """Test torch.eq operation."""
        # Test with identical tensors
        lns1, lns2 = identical_tensors
        result = torch.eq(lns1, lns2)
        expected = torch.ones_like(lns1.value, dtype=torch.bool)
        verify_bool_result(result, expected, "Element-wise equality failed for identical tensors")

        # Test with different tensors
        lns1, lns2 = sample_tensors
        result = torch.eq(lns1, lns2)
        expected = torch.eq(lns1.value, lns2.value)
        verify_bool_result(result, expected, "Element-wise equality failed for different tensors")

        # Test with a regular tensor
        result = torch.eq(lns1, lns1.value)
        expected = torch.ones_like(lns1.value, dtype=torch.bool)
        verify_bool_result(result, expected, "Element-wise equality failed for LNS and regular tensor")

    def test_ne(self, identical_tensors, sample_tensors):
        """Test torch.ne operation."""
        # Test with identical tensors
        lns1, lns2 = identical_tensors
        result = torch.ne(lns1, lns2)
        expected = torch.zeros_like(lns1.value, dtype=torch.bool)
        verify_bool_result(result, expected, "Element-wise inequality failed for identical tensors")

        # Test with different tensors
        lns1, lns2 = sample_tensors
        result = torch.ne(lns1, lns2)
        expected = torch.ne(lns1.value, lns2.value)
        verify_bool_result(result, expected, "Element-wise inequality failed for different tensors")

        # Test with a regular tensor
        result = torch.ne(lns1, lns1.value)
        expected = torch.zeros_like(lns1.value, dtype=torch.bool)
        verify_bool_result(result, expected, "Element-wise inequality failed for LNS and regular tensor")

class TestLNSComparisons:
    """Tests for LNS comparison operations."""

    def test_ge(self, sample_tensors, mixed_sign_tensors):
        """Test torch.ge operation."""
        lns1, lns2 = sample_tensors
        result = torch.ge(lns1, lns2)
        expected = torch.ge(lns1.value, lns2.value)
        verify_bool_result(result, expected, "Greater-equal comparison failed for simple case")

        # Test mixed signs
        lns1, lns2 = mixed_sign_tensors
        result = torch.ge(lns1, lns2)
        expected = torch.ge(lns1.value, lns2.value)
        verify_bool_result(result, expected, "Greater-equal comparison failed for mixed signs")

        # Test with a regular tensor
        result = torch.ge(lns1, lns1.value)
        expected = torch.ge(lns1.value, lns1.value)
        verify_bool_result(result, expected, "Greater-equal comparison failed for LNS and regular tensor")

    def test_gt(self, sample_tensors, mixed_sign_tensors):
        """Test torch.gt operation."""
        lns1, lns2 = sample_tensors
        result = torch.gt(lns1, lns2)
        expected = torch.gt(lns1.value, lns2.value)
        verify_bool_result(result, expected, "Greater-than comparison failed for simple case")

        # Test mixed signs
        lns1, lns2 = mixed_sign_tensors
        result = torch.gt(lns1, lns2)
        expected = torch.gt(lns1.value, lns2.value)
        verify_bool_result(result, expected, "Greater-than comparison failed for mixed signs")

        # Test with a regular tensor
        result = torch.gt(lns1, lns1.value)
        expected = torch.gt(lns1.value, lns1.value)
        verify_bool_result(result, expected, "Greater-than comparison failed for LNS and regular tensor")

    def test_le(self, sample_tensors, mixed_sign_tensors):
        """Test torch.le operation."""
        lns1, lns2 = sample_tensors
        result = torch.le(lns1, lns2)
        expected = torch.le(lns1.value, lns2.value)
        verify_bool_result(result, expected, "Less-equal comparison failed for simple case")

        # Test mixed signs
        lns1, lns2 = mixed_sign_tensors
        result = torch.le(lns1, lns2)
        expected = torch.le(lns1.value, lns2.value)
        verify_bool_result(result, expected, "Less-equal comparison failed for mixed signs")

        # Test with a regular tensor
        result = torch.le(lns1, lns1.value)
        expected = torch.le(lns1.value, lns1.value)
        verify_bool_result(result, expected, "Less-equal comparison failed for LNS and regular tensor")

    def test_lt(self, sample_tensors, mixed_sign_tensors):
        """Test torch.lt operation."""
        lns1, lns2 = sample_tensors
        result = torch.lt(lns1, lns2)
        expected = torch.lt(lns1.value, lns2.value)
        verify_bool_result(result, expected, "Less-than comparison failed for simple case")

        # Test mixed signs
        lns1, lns2 = mixed_sign_tensors
        result = torch.lt(lns1, lns2)
        expected = torch.lt(lns1.value, lns2.value)
        verify_bool_result(result, expected, "Less-than comparison failed for mixed signs")

        # Test with a regular tensor
        result = torch.lt(lns1, lns1.value)
        expected = torch.lt(lns1.value, lns1.value)
        verify_bool_result(result, expected, "Less-than comparison failed for LNS and regular tensor")

class TestLNSClose:
    """Tests for LNS isclose and allclose operations."""

    def test_isclose(self, identical_tensors, sample_tensors):
        """Test torch.isclose operation."""
        lns1, lns2 = identical_tensors
        result = torch.isclose(lns1, lns2)
        expected = torch.ones_like(lns1.value, dtype=torch.bool)
        verify_bool_result(result, expected, "isclose failed for identical tensors")

        # Test with different tensors but within tolerance
        lns1, _ = sample_tensors
        lns1_slightly_off = xltorch.lnstensor(lns1.value * (1 + 1e-10), b=lns1.base)
        result = torch.isclose(lns1, lns1_slightly_off)
        expected = torch.ones_like(lns1.value, dtype=torch.bool)
        verify_bool_result(result, expected, "isclose failed for nearly identical tensors")

        # Test with tensors outside of tolerance
        lns1, lns2 = sample_tensors
        result = torch.isclose(lns1, lns2)
        expected = torch.isclose(lns1.value, lns2.value)
        verify_bool_result(result, expected, "isclose failed for different tensors")

        # Test with custom rtol and atol
        result = torch.isclose(lns1, lns2, rtol=1.0, atol=10.0)
        expected = torch.isclose(lns1.value, lns2.value, rtol=1.0, atol=10.0)
        verify_bool_result(result, expected, "isclose failed with custom tolerances")

    def test_allclose(self, identical_tensors, sample_tensors):
        """Test torch.allclose operation."""
        lns1, lns2 = identical_tensors
        result = torch.allclose(lns1, lns2)
        assert result.item() is True, "allclose should return True for identical tensors"

        # Test with different tensors
        lns1, lns2 = sample_tensors
        result = torch.allclose(lns1, lns2)
        expected = torch.allclose(lns1.value, lns2.value)
        assert result == expected, "allclose result doesn't match expected for different tensors"

        # Test with custom rtol and atol
        result = torch.allclose(lns1, lns2, rtol=1.0, atol=10.0)
        expected = torch.allclose(lns1.value, lns2.value, rtol=1.0, atol=10.0)
        assert result == expected, "allclose failed with custom tolerances"

class TestLNSAnyAll:
    """Tests for LNS any and all operations."""

    def test_any(self, sample_tensors, edge_case_tensors):
        """Test torch.any operation."""
        lns1, _ = sample_tensors
        result = torch.any(lns1)
        expected = torch.any(lns1.value != 0)
        assert result == expected, "any failed for non-zero tensor"

        # Test with a tensor that has zeros
        lns1, _ = edge_case_tensors
        result = torch.any(lns1)
        expected = torch.any(lns1.value != 0)
        assert result == expected, "any failed for tensor with zeros"

        # Test with dimension parameter
        result = torch.any(lns1, dim=0)
        expected = torch.any(lns1.value != 0, dim=0)
        assert torch.all(result == expected), "any with dimension failed"

        # Test with keepdim parameter
        result = torch.any(lns1, dim=0, keepdim=True)
        expected = torch.any(lns1.value != 0, dim=0, keepdim=True)
        assert result.shape == expected.shape, "any with keepdim failed to preserve dimensions"
        assert torch.all(result == expected), "any with keepdim failed"

    def test_all(self, sample_tensors, edge_case_tensors):
        """Test torch.all operation."""
        lns1, _ = sample_tensors
        result = torch.all(lns1)
        expected = torch.all(lns1.value != 0)
        assert result == expected, "all failed for non-zero tensor"

        # Test with a tensor that has zeros
        lns1, _ = edge_case_tensors
        result = torch.all(lns1)
        expected = torch.all(lns1.value != 0)
        assert result == expected, "all failed for tensor with zeros"

        # Test with dimension parameter
        result = torch.all(lns1, dim=0)
        expected = torch.all(lns1.value != 0, dim=0)
        assert torch.all(result == expected), "all with dimension failed"

        # Test with keepdim parameter
        result = torch.all(lns1, dim=0, keepdim=True)
        expected = torch.all(lns1.value != 0, dim=0, keepdim=True)
        assert result.shape == expected.shape, "all with keepdim failed to preserve dimensions"
        assert torch.all(result == expected), "all with keepdim failed"

class TestLNSIsin:
    """Tests for LNS isin operation."""

    def test_isin(self, sample_tensors):
        """Test torch.isin operation."""
        lns1, lns2 = sample_tensors
        result = torch.isin(lns1, lns2)
        expected = torch.isin(lns1.value, lns2.value)
        verify_bool_result(result, expected, "isin failed")

        # Test with assume_unique=True
        result = torch.isin(lns1, lns2, assume_unique=True)
        expected = torch.isin(lns1.value, lns2.value, assume_unique=True)
        verify_bool_result(result, expected, "isin with assume_unique failed")

        # Test with invert=True
        result = torch.isin(lns1, lns2, invert=True)
        expected = torch.isin(lns1.value, lns2.value, invert=True)
        verify_bool_result(result, expected, "isin with invert failed")

class TestLNSSorting:
    """Tests for LNS sorting operations."""

    def test_sort(self, sample_tensors, mixed_sign_tensors):
        """Test torch.sort operation."""
        lns1, _ = sample_tensors
        values, indices = torch.sort(lns1)
        expected_values, expected_indices = torch.sort(lns1.value)

        assert isinstance(values, xltorch.LNSTensor), "Sorted values should be an LNS tensor"
        assert torch.allclose(values.value, expected_values), "Sort values failed"
        assert torch.all(indices == expected_indices), "Sort indices failed"

        # Test with descending=True
        values, indices = torch.sort(lns1, descending=True)
        expected_values, expected_indices = torch.sort(lns1.value, descending=True)

        assert isinstance(values, xltorch.LNSTensor), "Sorted values should be an LNS tensor"
        assert torch.allclose(values.value, expected_values), "Sort descending values failed"
        assert torch.all(indices == expected_indices), "Sort descending indices failed"

        # Test with mixed signs
        lns1, _ = mixed_sign_tensors
        values, indices = torch.sort(lns1)
        expected_values, expected_indices = torch.sort(lns1.value)

        assert isinstance(values, xltorch.LNSTensor), "Sorted values should be an LNS tensor"
        assert torch.allclose(values.value, expected_values), "Sort with mixed signs failed"
        assert torch.all(indices == expected_indices), "Sort indices with mixed signs failed"

        # Test with dimension parameter
        lns1_2d = xltorch.lnstensor([[1.0, 3.0, 2.0], [4.0, 5.0, 6.0]], f=23)
        values, indices = torch.sort(lns1_2d, dim=1)
        expected_values, expected_indices = torch.sort(lns1_2d.value, dim=1)

        assert isinstance(values, xltorch.LNSTensor), "Sorted values should be an LNS tensor"
        assert torch.allclose(values.value, expected_values), "Sort with dimension failed"
        assert torch.all(indices == expected_indices), "Sort indices with dimension failed"

    def test_argsort(self, sample_tensors, mixed_sign_tensors):
        """Test torch.argsort operation."""
        lns1, _ = sample_tensors
        indices = torch.argsort(lns1)
        expected_indices = torch.argsort(lns1.value)
        assert torch.all(indices == expected_indices), "Argsort failed"

        # Test with descending=True
        indices = torch.argsort(lns1, descending=True)
        expected_indices = torch.argsort(lns1.value, descending=True)
        assert torch.all(indices == expected_indices), "Argsort descending failed"

        # Test with mixed signs
        lns1, _ = mixed_sign_tensors
        indices = torch.argsort(lns1)
        expected_indices = torch.argsort(lns1.value)
        assert torch.all(indices == expected_indices), "Argsort with mixed signs failed"
        
        # Test with dimension parameter
        lns1_2d = xltorch.lnstensor([[1.0, 3.0, 2.0], [4.0, 5.0, 6.0]], f=23)
        indices = torch.argsort(lns1_2d, dim=1)
        expected_indices = torch.argsort(lns1_2d.value, dim=1)
        assert torch.all(indices == expected_indices), "Argsort with dimension failed"

    def test_kthvalue(self, sample_tensors):
        """Test torch.kthvalue operation."""
        lns1, _ = sample_tensors
        values, indices = torch.kthvalue(lns1, k=2)
        expected_values, expected_indices = torch.kthvalue(lns1.value, k=2)

        assert isinstance(values, xltorch.LNSTensor), "Kthvalue values should be an LNS tensor"
        assert torch.allclose(values.value, expected_values), "Kthvalue values failed"
        assert torch.all(indices == expected_indices), "Kthvalue indices failed"

        # Test with dimension parameter
        lns1_2d = xltorch.lnstensor([[1.0, 3.0, 2.0], [4.0, 5.0, 6.0]], f=23)
        values, indices = torch.kthvalue(lns1_2d, k=2, dim=1)
        expected_values, expected_indices = torch.kthvalue(lns1_2d.value, k=2, dim=1)

        assert isinstance(values, xltorch.LNSTensor), "Kthvalue values should be an LNS tensor"
        assert torch.allclose(values.value, expected_values), "Kthvalue with dimension failed"
        assert torch.all(indices == expected_indices), "Kthvalue indices with dimension failed"

        # Test with keepdim parameter
        values, indices = torch.kthvalue(lns1_2d, k=2, dim=1, keepdim=True)
        expected_values, expected_indices = torch.kthvalue(lns1_2d.value, k=2, dim=1, keepdim=True)

        assert isinstance(values, xltorch.LNSTensor), "Kthvalue values should be an LNS tensor"
        assert torch.allclose(values.value, expected_values), "Kthvalue with keepdim failed"
        assert torch.all(indices == expected_indices), "Kthvalue indices with keepdim failed"
        assert values.shape == expected_values.shape, "Kthvalue shape with keepdim failed"

class TestLNSMinMax:
    """Tests for LNS minimum and maximum operations."""

    def test_maximum(self, sample_tensors, mixed_sign_tensors):
        """Test torch.maximum operation."""
        lns1, lns2 = sample_tensors
        result = torch.maximum(lns1, lns2)
        expected = torch.maximum(lns1.value, lns2.value)

        assert isinstance(result, xltorch.LNSTensor), "Maximum result should be an LNS tensor"
        assert torch.allclose(result.value, expected), "Maximum values failed"

        # Test with mixed signs
        lns1, lns2 = mixed_sign_tensors
        result = torch.maximum(lns1, lns2)
        expected = torch.maximum(lns1.value, lns2.value)

        assert isinstance(result, xltorch.LNSTensor), "Maximum with mixed signs should be an LNS tensor"
        assert torch.allclose(result.value, expected), "Maximum with mixed signs failed"

        # Test with a regular tensor
        result = torch.maximum(lns1, lns1.value)
        expected = torch.maximum(lns1.value, lns1.value)

        assert isinstance(result, xltorch.LNSTensor), "Maximum with regular tensor should be an LNS tensor"
        assert torch.allclose(result.value, expected), "Maximum with regular tensor failed"

    def test_minimum(self, sample_tensors, mixed_sign_tensors):
        """Test torch.minimum operation."""
        lns1, lns2 = sample_tensors
        result = torch.minimum(lns1, lns2)
        expected = torch.minimum(lns1.value, lns2.value)

        assert isinstance(result, xltorch.LNSTensor), "Minimum result should be an LNS tensor"
        assert torch.allclose(result.value, expected), "Minimum values failed"

        # Test with mixed signs
        lns1, lns2 = mixed_sign_tensors
        result = torch.minimum(lns1, lns2)
        expected = torch.minimum(lns1.value, lns2.value)

        assert isinstance(result, xltorch.LNSTensor), "Minimum with mixed signs should be an LNS tensor"
        assert torch.allclose(result.value, expected), "Minimum with mixed signs failed"

        # Test with a regular tensor
        result = torch.minimum(lns1, lns1.value)
        expected = torch.minimum(lns1.value, lns1.value)

        assert isinstance(result, xltorch.LNSTensor), "Minimum with regular tensor should be an LNS tensor"
        assert torch.allclose(result.value, expected), "Minimum with regular tensor failed"
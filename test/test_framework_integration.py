# test/test_framework_integration.py

import pytest
import numpy as np
import xlns as xl
from xlnsconf.framework.common import LNSFrameworkBase

def test_basic_framework_conversion():
    """Test basic data conversion functionality."""
    # Create test data
    test_data = np.array([[1.0, 2.0], [3.0, 4.0]])
    
    # Convert to LNS format
    lns_tensor = LNSFrameworkBase(test_data)
    
    # Verify data is preserved
    np.testing.assert_array_almost_equal(
        lns_tensor.lns_data.nd,
        xl.xlnsnp(test_data).nd
    )

def test_precision_handling():
    """Test precision specification."""
    test_data = np.array([1.0, 2.0])
    
    # Test with custom precision
    custom_precision = 10
    lns_tensor = LNSFrameworkBase(test_data, precision=custom_precision)
    
    # Verify precision is maintained
    assert lns_tensor._precision == custom_precision

def test_operations():
    """Test basic LNS operations."""
    x = LNSFrameworkBase([1.0, 2.0])
    y = LNSFrameworkBase([3.0, 4.0])
    
    # Test multiplication (should be simple in LNS)
    result = x.lns_data * y.lns_data
    expected = xl.xlnsnp([3.0, 8.0])
    
    np.testing.assert_array_almost_equal(
        result.nd,
        expected.nd
    )
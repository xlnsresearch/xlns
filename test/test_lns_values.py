import numpy as np
import xlns as xl
from xlnsconf.framework.common import LNSFrameworkBase

def test_lns_values():
    print("Testing LNS Framework:")
    print("-" * 50)
    
    # Test 1: Basic array conversion
    print("\nTest 1: Basic Array Conversion")
    test_data = np.array([[1.0, 2.0], [3.0, 4.0]])
    lns_tensor = LNSFrameworkBase(test_data)
    print(f"Original data:\n{test_data}")
    print(f"LNS data:\n{lns_tensor.lns_data.nd}")
    
    # Test 2: Custom precision
    print("\nTest 2: Custom Precision")
    test_data_2 = np.array([1.0, 2.0])
    custom_precision = 10
    lns_tensor_2 = LNSFrameworkBase(test_data_2, precision=custom_precision)
    print(f"Original data: {test_data_2}")
    print(f"LNS precision: {lns_tensor_2._precision}")
    
    # Test 3: Basic operations
    print("\nTest 3: Basic Operations")
    x = LNSFrameworkBase(np.array([1.0, 2.0]))
    y = LNSFrameworkBase(np.array([3.0, 4.0]))
    
    # Test multiplication
    result = x.lns_data * y.lns_data
    expected = xl.xlnsnp([3.0, 8.0])
    print(f"Multiplication test:")
    print(f"x: {x.lns_data.nd}")
    print(f"y: {y.lns_data.nd}")
    print(f"x * y: {result.nd}")
    print(f"Expected: {expected.nd}")
    
    # Test additional values
    print("\nTest 4: Additional Value Tests")
    test_values = [0.1, 1.0, 10.0, 100.0]
    for val in test_values:
        try:
            lns_val = LNSFrameworkBase(np.array([val]))
            print(f"\nTesting value: {val}")
            print(f"LNS representation: {lns_val.lns_data.nd}")
        except Exception as e:
            print(f"Error processing value {val}: {str(e)}")

if __name__ == "__main__":
    test_lns_values()
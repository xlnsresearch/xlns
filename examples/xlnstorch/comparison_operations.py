import xlnstorch as xltorch

a = xltorch.lnstensor([1.0, 2.0, 3.0])
b = xltorch.lnstensor([3.0, 2.0, 1.0])

# Comparison operations
eq_result = a == b  # element-wise equality
ne_result = a != b  # element-wise inequality
gt_result = a > b   # element-wise greater than
ge_result = a >= b  # element-wise greater than or equal
lt_result = a < b   # element-wise less than
le_result = a <= b  # element-wise less than or equal

print(f"a == b: {eq_result}")
print(f"a != b: {ne_result}")
print(f"a > b: {gt_result}")
print(f"a >= b: {ge_result}")
print(f"a < b: {lt_result}")
print(f"a <= b: {le_result}")
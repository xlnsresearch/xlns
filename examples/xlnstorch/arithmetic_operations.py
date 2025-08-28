import xlnstorch as xltorch

a = xltorch.lnstensor([1.0, 2.0, 3.0])
b = xltorch.lnstensor([4.0, 5.0, 6.0])

# Basic operations
add_result = a + b
sub_result = a - b
mul_result = a * b
div_result = a / b

# In-place operations
c = a.clone()
c.add_(b)  # c += b

# Other math operations
sqrt_result = a.sqrt()
abs_result = (-a).abs()
recip_result = a.reciprocal()
square_result = a.square()

# Sum all elements
d = xltorch.lnstensor([[1.0, 2.0], [3.0, 4.0]])
sum_all = d.sum()

# Sum along dimensions
sum_dim0 = d.sum(dim=0)  # sum rows
sum_dim1 = d.sum(dim=1)  # sum columns

print(f"a + b = {add_result}")
print(f"a - b = {sub_result}")
print(f"a * b = {mul_result}")
print(f"a / b = {div_result}")
print(f"sqrt(a) = {sqrt_result}")
print(f"abs(a) = {abs_result}")
print(f"1/a = {recip_result}")
print(f"a^2 = {square_result}")
print(f"Sum of all elements of d: {sum_all}")
print(f"Sum along dim 0 of d: {sum_dim0}")
print(f"Sum along dim 1 of d: {sum_dim1}")
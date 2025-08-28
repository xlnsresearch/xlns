import torch
import xlnstorch as xltorch
import numpy as np

# Create LNSTensor from different data sources
tensor1 = xltorch.lnstensor([1.0, 2.0, 3.0])  # from Python list
tensor2 = xltorch.lnstensor(torch.tensor([4.0, 5.0, 6.0]))  # from torch tensor
tensor3 = xltorch.lnstensor(np.array([7.0, 8.0, 9.0]))  # from numpy array

print(tensor1)
print(tensor2)
print(tensor3)

# Specify base using 'b' parameter
tensor_base2 = xltorch.lnstensor([1.0, 2.0, 4.0], b=2.0)

# Specify base using 'f' parameter (f bits of fraction precision)
tensor_f8 = xltorch.lnstensor([1.0, 2.0, 4.0], f=8)

# Create zero tensors
zeros = xltorch.zeros(3, 4)  # 3x4 tensor of zeros
zeros_like = xltorch.zeros_like(tensor1)  # same shape as tensor1

# Create tensors of ones
ones = xltorch.ones(2, 3)
ones_like = xltorch.ones_like(tensor1)

# Create tensors with specific value
full = xltorch.full((2, 2), 5.0)
full_like = xltorch.full_like(tensor2, 7.0)

# Create tensors with random values
rand = xltorch.rand(3, 3)  # uniform distribution [0, 1]
rand_like = xltorch.rand_like(tensor3)

randn = xltorch.randn(3, 3)  # normal distribution
randn_like = xltorch.randn_like(tensor3)

print(f"Zeros: {zeros}")
print(f"Ones: {ones}")
print(f"Random (uniform): {rand}")
print(f"Random (normal): {randn}")
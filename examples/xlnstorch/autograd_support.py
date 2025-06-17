import xlnstorch as xltorch

# Create tensors with gradient tracking
a = xltorch.lnstensor([1.0, 2.0, 3.0], requires_grad=True)
b = xltorch.lnstensor([4.0, 5.0, 6.0], requires_grad=True)

# Perform operations
c = a * b
d = c.sum()

# Backward pass to compute gradients
d.backward()

# Access gradients
print(f"a: {a}")
print(f"b: {b}")
print(f"c = a * b: {c}")
print(f"d = sum(c): {d}")
print(f"a.grad: {a.grad}")
print(f"b.grad: {b.grad}")

# Detach from computation graph
detached = c.detach()
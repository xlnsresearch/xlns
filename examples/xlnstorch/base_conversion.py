import xlnstorch as xltorch

# Create tensors with different bases
a = xltorch.lnstensor([1.0, 2.0, 3.0], f=23)
b = xltorch.lnstensor([1.0, 2.0, 3.0], f=18)

# Convert between bases
c = xltorch.lnstensor(a, f=13) # Convert from f=23 to f=13

print(f"Original tensor (f=23): {a}")
print(f"Same values (f=18): {b}")
print(f"Converted from a to f=13: {c}")
print(f"LNS representation of a: {a.lns}")
print(f"LNS representation of b: {b.lns}")
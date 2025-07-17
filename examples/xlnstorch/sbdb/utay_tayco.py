import torch
import xlnstorch as xltorch
from xlnstorch.benchmark import BenchmarkRunner, BinaryBench
import xlnsconf.utah_tayco_ufunc
import matplotlib.pyplot as plt

# see https://github.com/xlnsresearch/xlns/blob/main/src/xlnsconf/utah_tayco_ufunc.py
xltorch.operators.register_xlnsconf_implementation(
    xlnsconf.utah_tayco_ufunc.sbdb_ufunc_utah_tayco,
    "utah_tayco_conf"
)

# vary the following parameters as needed
f = 8
shape = (1000, 1000)
warmup = 5
iters = 20
make_heatmap = lambda: xltorch.viz.make_error_grid(
    torch.add,
    x_range=(-10.0, 10.0),
    y_range=(-10.0, 10.0),
    steps=201,
    f=f,
    absolute=True
)

bench = BinaryBench(torch.add, shape, lns=True, device="cpu")
runner = BenchmarkRunner(bench, warmup=warmup, iters=iters)


with xltorch.operators.override_sbdb_implementation("ideal"):
    ideal_result = runner.run()
    ideal_xs, ideal_ys, ideal_err = make_heatmap()

with xltorch.operators.override_sbdb_implementation("utah_tayco"):
    tayco_torch_result = runner.run()
    tayco_torch_xs, tayco_torch_ys, tayco_torch_err = make_heatmap()

with xltorch.operators.override_sbdb_implementation("utah_tayco_conf"):
    tayco_conf_result = runner.run()
    tayco_conf_xs, tayco_conf_ys, tayco_conf_err = make_heatmap()


print(f"Ideal xlnstorch wall_ms: {ideal_result.wall_ms:.2f} ms")
print(f"Utah Tayco xlnstorch wall_ms: {tayco_torch_result.wall_ms:.2f} ms")
print(f"Utah Tayco xlnsconf wall_ms: {tayco_conf_result.wall_ms:.2f} ms")

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 7))

# Hide the unused fourth subplot
ax4.axis('off')

# Plot the heatmaps and capture the returned objects to control colorbars
im1 = xltorch.viz.plot_error_heatmap(ideal_err, ideal_xs, ideal_ys, ax=ax1, cmap="viridis")
im2 = xltorch.viz.plot_error_heatmap(tayco_torch_err, tayco_torch_xs, tayco_torch_ys, ax=ax2, cmap="viridis")
im3 = xltorch.viz.plot_error_heatmap(tayco_conf_err, tayco_conf_xs, tayco_conf_ys, ax=ax3, cmap="viridis")

# Make plots square by setting equal aspect ratio
ax1.set_aspect('equal')
ax2.set_aspect('equal') 
ax3.set_aspect('equal')

ax1.set_title("ideal addition")
ax2.set_title("utah_tayco (xlnstorch) addition")
ax3.set_title("utah_tayco (xlnsconf) addition")

# Use tight_layout to automatically adjust spacing and colorbar sizes
plt.tight_layout()

plt.show()
import torch
import xlnstorch as xltorch
from xlnstorch.benchmark import BenchmarkRunner, BinaryBench
import xlnsconf.tab_ufunc
import matplotlib.pyplot as plt
import xlns

# see https://github.com/xlnsresearch/xlns/blob/main/src/xlnsconf/tab_ufunc.py
xltorch.register_xlnsconf_implementation(
    xlnsconf.tab_ufunc.sbdb_ufunc_tab,
    "tab_conf"
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

# load the table for the xlnstorch and xlnsconf implementations
xlns.xlnssetF(f)
xlnsconf.tab_ufunc.get_table("tmp")
xltorch.operators.implementations.tab.get_table("tmp", f=f)

bench = BinaryBench(torch.add, shape, lns=True, device="cpu")
runner = BenchmarkRunner(bench, warmup=warmup, iters=iters)

with xltorch.override_sbdb_implementation("ideal"):
    ideal_result = runner.run()
    ideal_xs, ideal_ys, ideal_err = make_heatmap()

with xltorch.override_sbdb_implementation("tab"):
    table_torch_result = runner.run()
    table_torch_xs, table_torch_ys, table_torch_err = make_heatmap()

with xltorch.override_sbdb_implementation("tab_conf"):
    table_conf_result = runner.run()
    table_conf_xs, table_conf_ys, table_conf_err = make_heatmap()


print(f"Ideal xlnstorch wall_ms: {ideal_result.wall_ms:.2f} ms")
print(f"Table xlnstorch wall_ms: {table_torch_result.wall_ms:.2f} ms")
print(f"Table xlnsconf wall_ms: {table_conf_result.wall_ms:.2f} ms")

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 7))

# Hide the unused fourth subplot
ax4.axis('off')

# Plot the heatmaps and capture the returned objects to control colorbars
im1 = xltorch.viz.plot_error_heatmap(ideal_err, ideal_xs, ideal_ys, ax=ax1, cmap="viridis")
im2 = xltorch.viz.plot_error_heatmap(table_torch_err, table_torch_xs, table_torch_ys, ax=ax2, cmap="viridis")
im3 = xltorch.viz.plot_error_heatmap(table_conf_err, table_conf_xs, table_conf_ys, ax=ax3, cmap="viridis")

# Make plots square by setting equal aspect ratio
ax1.set_aspect('equal')
ax2.set_aspect('equal') 
ax3.set_aspect('equal')

ax1.set_title("ideal addition")
ax2.set_title("table (xlnstorch) addition")
ax3.set_title("table (xlnsconf) addition")

# Use tight_layout to automatically adjust spacing and colorbar sizes
plt.tight_layout()

plt.show()
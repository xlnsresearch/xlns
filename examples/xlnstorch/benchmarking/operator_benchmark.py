import torch
import xlnstorch as xltorch
from xlnstorch.benchmark import Benchmark, BenchmarkRunner, UnaryBench, BinaryBench

class PrecisionBench(Benchmark):

    def __init__(self, precision, shape, device="cpu"):
        self.precision = precision
        self.shape = shape
        self.device = device

    def make_inputs(self):
        a = xltorch.randn(*self.shape, f=self.precision, device=self.device)
        b = xltorch.randn_like(a)
        return a, b

    def forward(self, x, y):
        return torch.mul(x, y)

def demo_unary_operations():
    """
    Benchmark unary operations comparing LNS vs regular PyTorch.
    """
    print("=== Unary Operations Benchmark ===")

    operations = [
        ("sqrt", torch.sqrt),
        ("abs", torch.abs),
        ("sign", torch.sign),
        ("tanh", torch.tanh),
        ("sigmoid", torch.sigmoid),
    ]

    shape = (1000, 1000)
    warmup = 5
    iters = 20

    print(f"Benchmarking shape {shape} with {warmup} warmup, {iters} iterations")
    print()

    for op_name, op_func in operations:
        print(f"--- {op_name.upper()} ---")

        # Benchmark LNS version
        lns_bench = UnaryBench(op_func, shape, lns=True, device="cpu")
        lns_runner = BenchmarkRunner(lns_bench, warmup=warmup, iters=iters)
        lns_result = lns_runner.run()

        # Benchmark regular PyTorch version
        torch_bench = UnaryBench(op_func, shape, lns=False, device="cpu")
        torch_runner = BenchmarkRunner(torch_bench, warmup=warmup, iters=iters)
        torch_result = torch_runner.run()

        print(f"LNS wall_ms: {lns_result.wall_ms:.2f} ms")
        print(f"PyTorch wall_ms: {torch_result.wall_ms:.2f} ms")

        # Calculate overhead
        overhead = (lns_result.wall_ms / torch_result.wall_ms - 1) * 100
        print(f"LNS overhead: {overhead:.1f}%")
        print()

def demo_binary_operations():
    """
    Benchmark binary operations comparing LNS vs regular PyTorch.
    """
    print("=== Binary Operations Benchmark ===")

    operations = [
        ("add", torch.add),
        ("sub", torch.sub),
        ("mul", torch.mul),
        ("div", torch.div),
    ]

    shape = (1000, 1000)
    warmup = 5
    iters = 20

    print(f"Benchmarking shape {shape} with {warmup} warmup, {iters} iterations")
    print()

    for op_name, op_func in operations:
        print(f"--- {op_name.upper()} ---")

        # Benchmark LNS version
        lns_bench = BinaryBench(op_func, shape, lns=True, device="cpu")
        lns_runner = BenchmarkRunner(lns_bench, warmup=warmup, iters=iters)
        lns_result = lns_runner.run()

        # Benchmark regular PyTorch version
        torch_bench = BinaryBench(op_func, shape, lns=False, device="cpu")
        torch_runner = BenchmarkRunner(torch_bench, warmup=warmup, iters=iters)
        torch_result = torch_runner.run()

        print(f"LNS wall_ms: {lns_result.wall_ms:.2f} ms")
        print(f"PyTorch wall_ms: {torch_result.wall_ms:.2f} ms")

        # Calculate overhead
        overhead = (lns_result.wall_ms / torch_result.wall_ms - 1) * 100
        print(f"LNS overhead: {overhead:.1f}%")
        print()

def demo_precision_performance():
    """
    Benchmark how different LNS precisions affect performance.
    """
    print("=== Precision vs Performance ===")

    precisions = [4, 8, 12, 16]
    shape = (1000, 1000)
    warmup = 5
    iters = 20

    print(f"Benchmarking multiplication across different precisions")
    print(f"Shape: {shape}, warmup: {warmup}, iterations: {iters}")
    print()

    results = {}
    for f in precisions:
        print(f"--- Precision f={f} bits ---")

        bench = PrecisionBench(f, shape)
        runner = BenchmarkRunner(bench, warmup=warmup, iters=iters)
        result = runner.run()

        results[f] = result
        print(f"wall_ms: {result.wall_ms:.2f} ms")
        print()

    # Compare relative performance
    print("Performance comparison (these should all be close to 1.0x):")
    base_time = results[precisions[0]].wall_ms
    for f in precisions:
        relative = results[f].wall_ms / base_time
        print(f"  f={f:2d}: {relative:.2f}x relative to f={precisions[0]}")

def main():
    """
    Main demo function that runs basic benchmark examples.
    """
    print("xlnstorch Basic Benchmark Demo")
    print("=" * 40)
    print()

    # Run basic demos
    demo_unary_operations()
    print("\n" + "=" * 40 + "\n")

    demo_binary_operations()
    print("\n" + "=" * 40 + "\n")

    demo_precision_performance()

if __name__ == "__main__":
    main()
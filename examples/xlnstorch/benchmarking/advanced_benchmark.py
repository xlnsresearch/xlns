import torch
import xlnstorch as xltorch
from xlnstorch.benchmark import Benchmark, BenchmarkRunner, UnaryBench, BinaryBench

class CustomNeuralNetBench(Benchmark):
    """
    Custom benchmark for a simple neural network forward pass.

    This demonstrates how to create custom benchmarks for more complex operations.
    """

    def __init__(self, input_size, hidden_size, output_size, lns=True, device="cpu"):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.lns = lns
        self.device = device

        # Create weights for the network
        if lns:
            self.w1 = xltorch.randn(input_size, hidden_size, device=device, requires_grad=True)
            self.w2 = xltorch.randn(hidden_size, output_size, device=device, requires_grad=True)
        else:
            self.w1 = torch.randn(input_size, hidden_size, device=device, requires_grad=True, dtype=torch.float64)
            self.w2 = torch.randn(hidden_size, output_size, device=device, requires_grad=True, dtype=torch.float64)

    def make_inputs(self):
        if self.lns:
            x = xltorch.randn(32, self.input_size, device=self.device, requires_grad=True)
        else:
            x = torch.randn(32, self.input_size, device=self.device, requires_grad=True, dtype=torch.float64)
        return (x,)

    def forward(self, x):
        # Simple 2-layer neural network
        h = torch.matmul(x, self.w1)
        h_activated = torch.tanh(h)
        output = torch.matmul(h_activated, self.w2)
        return output

def demo_custom_benchmark():
    """
    Demonstrate custom benchmark for neural network operations.
    """
    print("=== Custom Neural Network Benchmark ===")

    input_size = 128
    hidden_size = 256
    output_size = 64
    warmup = 3
    iters = 10

    print(f"Neural network: {input_size} -> {hidden_size} -> {output_size}")
    print(f"Batch size: 32, warmup: {warmup}, iterations: {iters}")
    print()

    # Benchmark LNS neural network
    lns_net_bench = CustomNeuralNetBench(input_size, hidden_size, output_size, lns=True)
    lns_net_runner = BenchmarkRunner(lns_net_bench, warmup=warmup, iters=iters)
    lns_net_result = lns_net_runner.run()

    # Benchmark regular PyTorch neural network
    torch_net_bench = CustomNeuralNetBench(input_size, hidden_size, output_size, lns=False)
    torch_net_runner = BenchmarkRunner(torch_net_bench, warmup=warmup, iters=iters)
    torch_net_result = torch_net_runner.run()

    print(f"LNS wall_ms: {lns_net_result.wall_ms:.2f} ms")
    print(f"PyTorch wall_ms: {torch_net_result.wall_ms:.2f} ms")

    # Calculate overhead
    overhead = (lns_net_result.wall_ms / torch_net_result.wall_ms - 1) * 100
    print(f"LNS overhead: {overhead:.1f}%")
    print()

def demo_profiling():
    """
    Demonstrate profiling functionality to see detailed performance breakdown.
    """
    print("=== Profiling Demo ===")

    # Create a benchmark for matrix multiplication
    shape = (64, 64)
    print(f"Profiling matrix multiplication with shape {shape}")
    print()

    # Profile LNS matmul
    print("--- LNS Matrix Multiplication Profile ---")
    lns_bench = BinaryBench(torch.matmul, shape, lns=True, device="cpu")
    lns_runner = BenchmarkRunner(lns_bench, warmup=2, iters=5, profile=True)
    lns_result = lns_runner.run()

    print("Summary:")
    lns_result.print()
    print()

    # Save detailed profile to file
    try:
        lns_result.save_full_profile("lns_matmul_profile.txt", row_limit=20)
        print("Detailed LNS profile saved to 'lns_matmul_profile.txt'")
    except Exception as e:
        print(f"Could not save profile: {e}")

    print()

    # Profile regular PyTorch matmul for comparison
    print("--- PyTorch Matrix Multiplication Profile ---")
    torch_bench = BinaryBench(torch.matmul, shape, lns=False, device="cpu")
    torch_runner = BenchmarkRunner(torch_bench, warmup=2, iters=5, profile=True)
    torch_result = torch_runner.run()

    print("Summary:")
    torch_result.print()
    print()

    try:
        torch_result.save_full_profile("torch_matmul_profile.txt", row_limit=20)
        print("Detailed PyTorch profile saved to 'torch_matmul_profile.txt'")
    except Exception as e:
        print(f"Could not save profile: {e}")

    print()

def demo_gradient_benchmark():
    """
    Benchmark operations with gradient computation (backward pass).
    """
    print("=== Gradient Computation Benchmark ===")

    shape = (64, 64)
    warmup = 3
    iters = 10

    print(f"Benchmarking with gradients, shape {shape}")
    print("Forward + Backward pass timing")
    print()

    operations = [
        ("tanh", torch.tanh),
        ("relu", torch.nn.functional.relu)
    ]

    for op_name, op_func in operations:
        print(f"--- {op_name.upper()} (with gradients) ---")

        lns_bench = UnaryBench(op_func, shape, lns=True, backward=True, device="cpu")
        torch_bench = UnaryBench(op_func, shape, lns=False, backward=True, device="cpu")

        # Run with backward pass enabled
        lns_runner = BenchmarkRunner(lns_bench, warmup=warmup, iters=iters, backward=True)
        torch_runner = BenchmarkRunner(torch_bench, warmup=warmup, iters=iters, backward=True)

        lns_result = lns_runner.run()
        torch_result = torch_runner.run()

        print(f"LNS (forward + backward) wall_ms: {lns_result.wall_ms:.2f} ms")
        print(f"PyTorch (forward + backward) wall_ms: {torch_result.wall_ms:.2f} ms")

        overhead = (lns_result.wall_ms / torch_result.wall_ms - 1) * 100
        print(f"LNS overhead: {overhead:.1f}%")
        print()

class MemoryIntensiveBench(Benchmark):
    """
    Custom benchmark to test memory-intensive operations.
    """

    def __init__(self, shape, lns=True, device="cpu"):
        self.shape = shape
        self.lns = lns
        self.device = device

    def make_inputs(self):
        if self.lns:
            # Create large tensors
            a = xltorch.randn(*self.shape, device=self.device)
            b = xltorch.randn(*self.shape, device=self.device)
            c = xltorch.randn(*self.shape, device=self.device)
        else:
            a = torch.randn(*self.shape, device=self.device, dtype=torch.float64)
            b = torch.randn(*self.shape, device=self.device, dtype=torch.float64)
            c = torch.randn(*self.shape, device=self.device, dtype=torch.float64)
        return a, b, c

    def forward(self, a, b, c):
        # Memory-intensive chain of operations
        result = a + b
        result = result * c
        result = torch.sum(result)
        return result

def demo_memory_benchmark():
    """
    Demonstrate benchmarking memory-intensive operations.
    """
    print("=== Memory-Intensive Operations Benchmark ===")

    shape = (16, 16)
    warmup = 2
    iters = 5

    print(f"Memory-intensive operations with shape {shape}")
    print(f"Operations: add + multiply + sum, warmup: {warmup}, iterations: {iters}")
    print()

    # Benchmark LNS version with profiling to see memory usage
    print("--- LNS Memory-Intensive Operations ---")
    lns_bench = MemoryIntensiveBench(shape, lns=True)
    lns_runner = BenchmarkRunner(lns_bench, warmup=warmup, iters=iters, profile=True)
    lns_result = lns_runner.run()
    lns_result.print()
    print()

    # Benchmark PyTorch version
    print("--- PyTorch Memory-Intensive Operations ---")
    torch_bench = MemoryIntensiveBench(shape, lns=False)
    torch_runner = BenchmarkRunner(torch_bench, warmup=warmup, iters=iters, profile=True)
    torch_result = torch_runner.run()
    torch_result.print()

    # Compare memory usage
    print(f"Memory comparison:")
    print(f"  LNS CPU memory: {lns_result.cpu_mem_mb:.2f} MB")
    print(f"  PyTorch CPU memory: {torch_result.cpu_mem_mb:.2f} MB")

    if lns_result.cpu_mem_mb > 0 and torch_result.cpu_mem_mb > 0:
        mem_overhead = (lns_result.cpu_mem_mb / torch_result.cpu_mem_mb - 1) * 100
        print(f"  LNS memory overhead: {mem_overhead:.1f}%")

    print()

def main():
    """
    Main demo function that runs advanced benchmark examples.
    """
    print("xlnstorch Advanced Benchmark Demo")
    print("=" * 50)
    print()

    # Run advanced demos
    demo_custom_benchmark()
    print("\n" + "=" * 50 + "\n")

    demo_profiling()
    print("\n" + "=" * 50 + "\n")

    demo_gradient_benchmark()
    print("\n" + "=" * 50 + "\n")

    demo_memory_benchmark()

    print("\n" + "=" * 50)
    print("Advanced Benchmark Demo completed!")
    print("\nGenerated files:")
    print("  - lns_matmul_profile.txt (detailed LNS profiling)")
    print("  - torch_matmul_profile.txt (detailed PyTorch profiling)")

if __name__ == "__main__":
    main()
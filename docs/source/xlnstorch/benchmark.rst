.. currentmodule:: xlnstorch

.. _benchmark-doc:

Benchmarking
============

The ``xlnstorch.benchmark`` submodule provides tools for performance measurement
and profiling of LNS operations. This module includes infrastructure for 
timing operations, memory usage tracking, and comparing performance between
standard PyTorch and LNS implementations.

Quick Start
-----------

Here's a simple example of benchmarking a unary operation:

.. code-block:: python

    import torch
    import xlnstorch as xltorch
    from xlnstorch.benchmark import UnaryBench, BenchmarkRunner

    # Benchmark LNS ReLU operation
    bench = UnaryBench(
        func=torch.relu,
        shape=(1000, 1000),
        device="cpu"
    )

    runner = BenchmarkRunner(bench, warmup=10, iters=100)
    result = runner.run()
    result.print()

And for benchmarking a binary operation e.g. `torch.mul`, we replace the
unary benchmark with a binary one:

.. code-block:: python

    # Benchmark LNS multiplication
    bench = BinaryBench(
        func=torch.mul,
        shape=(1000, 1000),
        device="cpu"
    )

Benchmark Classes
-----------------

:class:`Benchmark`
    Base class for creating custom benchmarks

:class:`BenchmarkRunner`
    Orchestrates benchmark execution with timing and profiling

:class:`BenchResult`
    Container for benchmark results and metrics

.. autosummary::
    :toctree: generated
    :nosignatures:

    benchmark.Benchmark
    benchmark.BenchmarkRunner
    benchmark.BenchResult

Pre-built Benchmarks
--------------------

The module includes ready-to-use benchmark classes for common operations:

.. autosummary::
    :toctree: generated
    :nosignatures:

    benchmark.UnaryBench
    benchmark.BinaryBench

Creating Custom Benchmarks
---------------------------

To create a custom benchmark, inherit from :class:`benchmark.Benchmark` and 
implement the required methods:

.. code-block:: python

    from xlnstorch.benchmark import Benchmark, BenchmarkRunner
    import xlnstorch as xltorch

    class MyCustomBench(Benchmark):
        device = "cpu"

        def make_inputs(self):
            # Return inputs as tuple or dict
            x = xltorch.randn(100, 100, device=self.device)
            y = xltorch.randn(100, 100, device=self.device)
            return (x, y)

        def forward(self, x, y):
            # The operation to benchmark
            return x @ y  # Matrix multiplication

    # Run the benchmark
    runner = BenchmarkRunner(MyCustomBench())
    result = runner.run()

Profiling Support
-----------------

The benchmark module integrates with PyTorch's profiler to provide detailed
performance analysis:

.. code-block:: python

    # Enable profiling
    runner = BenchmarkRunner(bench, profile=True)
    result = runner.run()

    # Save detailed profiler output
    result.save_full_profile("profile_output.txt")

    # Access profiler data directly
    if result.prof is not None:
        print(result.prof.key_averages().table())

Metrics and Results
-------------------

:class:`BenchResult` provides comprehensive timing and memory metrics:

- **Wall-clock timing**: Mean latency and percentiles (P50, P90, P99)
- **CPU/CUDA time**: Self-time reported by the profiler
- **Memory usage**: Peak CPU and CUDA memory consumption
- **Profiler integration**: Full profiler output for detailed analysis

.. code-block:: python

    result = runner.run()

    print(f"Mean latency: {result.wall_ms:.2f} ms")
    print(f"P99 latency: {result.p99:.2f} ms")
    print(f"CPU time: {result.cpu_us:.2f} μs")
    print(f"Memory usage: {result.cpu_mem_mb:.2f} MB")

Comparing Implementations
-------------------------

A common use case is comparing LNS vs standard PyTorch performance:

.. code-block:: python

    import torch
    import xlnstorch as xltorch
    from xlnstorch.benchmark import UnaryBench, BenchmarkRunner

    shape = (1000, 1000)

    # Benchmark standard PyTorch
    torch_bench = UnaryBench(torch.relu, shape, lns=False)
    torch_result = BenchmarkRunner(torch_bench).run()

    # Benchmark LNS implementation
    lns_bench = UnaryBench(xltorch.relu, shape, lns=True)
    lns_result = BenchmarkRunner(lns_bench).run()

    print("PyTorch ReLU:")
    torch_result.print()
    print("LNS ReLU:")
    lns_result.print()

Backward Pass Benchmarking
---------------------------

You can benchmark operations including their backward passes:

.. code-block:: python

    # Enable backward pass benchmarking
    runner = BenchmarkRunner(bench, backward=True)
    result = runner.run()

This is useful for understanding the full computational cost of operations
during training.

API Reference
-------------

For detailed API documentation, see the individual class and method documentation:

.. autosummary::
    :toctree: generated

    benchmark.Benchmark.make_inputs
    benchmark.Benchmark.forward
    benchmark.Benchmark.post_forward
    benchmark.Benchmark.before_epoch
    benchmark.Benchmark.after_epoch
    benchmark.BenchmarkRunner.run
    benchmark.BenchResult.print
    benchmark.BenchResult.save_full_profile
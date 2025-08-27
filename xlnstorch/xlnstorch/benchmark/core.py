
from __future__ import annotations
from typing import Any, Dict, Tuple, Callable, Iterable, Union, Optional
from dataclasses import dataclass, fields
import time
import numpy as np
import torch
from torch.profiler import profile, ProfilerActivity
from xlnstorch import LNSTensor

def _sync(dev: torch.device) -> None:
    """
    Explicitly synchronize the device.

    For CUDA devices, this guarantees all kernels launched so far finish
    before the function returns. For CPU devices, this is a no-op.
    """
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)

@dataclass
class BenchResult:
    """
    A container holding the aggregate numbers of a single benchmark run.

    Attributes
    ----------
    prof : Optional[torch.profiler.profile]
        The profiler output, if profiling was enabled. Otherwise ``None``.
    wall_ms: float
        Mean wall-clock latency in milliseconds.
    p50 / p90 / p99 : float
        Percentiles (ms) of the wall-clock distribution.
    cpu_us / cuda_us : float
        Self CPU / CUDA time reported by `torch.profiler` in micro-seconds.
        Values are zero when ``profile=False`` or the respective device is
        unavailable.
    cpu_mem_mb / cuda_mem_mb : float
        Peak memory (mb) reported by `torch.profiler`. Same fallback rules
        as above apply.
    """

    prof : Optional[torch.profiler.profile]
    wall_ms: float
    p50: float
    p90: float
    p99: float
    cpu_us: float = 0.0
    cpu_mem_mb: float = 0.0
    cuda_us: float = 0.0
    cuda_mem_mb: float = 0.0

    def print(self) -> None:
        """Nicely format the dataclass to stdout (monospaced columns)."""
        headers = [f.name for f in fields(self) if f.name != "prof"]
        values = [getattr(self, name) for name in headers]

        # format floats with two decimals, leave ints / others unchanged
        fmt_vals = [
            f"{v:.2f}" if isinstance(v, float) else str(v)
            for v in values
        ]
        # Column widths required for each field
        widths = [max(len(h), len(v)) for h, v in zip(headers, fmt_vals)]

        def line(parts: Iterable[str]) -> str:
            return "  ".join(p.ljust(w) for p, w in zip(parts, widths))

        print(line(headers))
        print("  ".join("-" * w for w in widths))
        print(line(fmt_vals))

    def save_full_profile(self, path: str, group_by_stack_n=0, sort_by="cpu_time_total", row_limit=10) -> None:
        if self.prof is None:
            raise ValueError("No profiler data available; run with profile=True")
        with open(path, "w") as f:
            f.write(self.prof.key_averages(group_by_stack_n=group_by_stack_n).table(
                sort_by=sort_by,
                row_limit=row_limit,
            ))

class Benchmark:
    """
    Base-class that users should inherit from in order to create benchmarks.

    Only 3 hooks are strictly required to be implemented:

    1. ``device`` class attribute (defaults to "cpu")
    2. :py:meth:`make_inputs`
    3. :py:meth:`forward`

    Optional hooks allow for custom book-keeping around each iteration /
    epoch as well as post-processing of the forward result.
    """

    # The device the benchmark will run on.  Can be overwritten in subclasses.
    device: Union[torch.device, str] = "cpu"

    def make_inputs(self):
        """
        Produce input tensors for one iteration of the benchmark.

        Returns
        -------
        Union[tuple, dict]
            *Tuple* treated as ``(*args,)``.
            *Dict* treated as ``(**kwargs)``.

        Raises
        ------
        ValueError
            If the return value is neither tuple nor dict.
        """
        raise NotImplementedError

    def forward(self, *args, **kwargs):
        """
        The workload under test (e.g. ``model(*args, **kwargs)``).
        Must be overriden by subclasses.
        """
        raise NotImplementedError

    def post_forward(self, output: Any) -> None:
        """
        Optional light-weight processing of the forward result. Called
        outside the timing region so it has no effect on latencies.
        """
        pass

    def before_epoch(self, idx: int) -> None:
        """
        Called before the *idx*-th epoch.  Currently a stub kept for API
        completeness; timing is performed per-iteration internally.
        """
        pass

    def after_epoch(self, idx: int) -> None:
        """Symmetric counterpart of :py:meth:`before_epoch`."""
        pass

class BenchmarkRunner:
    """
    Orchestrates warm-up, steady-state timing and optional profiling.

    Note: The runner clones all tensors returned by :py:meth:`Benchmark.make_inputs`
    before each iteration to avoid in-place side-effects leaking across runs.

    Parameters
    ----------
    bench_obj : Benchmark
        An instance of a :class:`Benchmark` subclass.
    warmup : int
        Number of warm-up iterations executed before timing starts.
    iters : int
        Number of timed iterations.
    profile : bool
        Whether to profile the benchmark using `torch.profiler`.
    backward : bool
        Whether to run a backward pass after each forward pass.
    """

    def __init__(
            self,
            bench_obj: Benchmark,
            warmup: int = 10,
            iters: int = 100,
            profile: bool = False,
            backward: bool = False,
        ):
        self.bench_obj = bench_obj
        self.warmup = warmup
        self.iters = iters
        self.profile = profile
        self.backward = backward

        if not isinstance(bench_obj, Benchmark):
            raise TypeError("bench_obj must be an instance of Benchmark")

    def run(self) -> BenchResult:
        """
        Execute the benchmark.

        Returns
        -------
        BenchResult
            The aggregated results as defined in :class:`BenchResult`.
        """

        device = torch.device(self.bench_obj.device)
        self.bench_obj.device = device # # make sure subclass sees resolved dev

        base_args, base_kwargs = self._construct_inputs()
        prof_ctx = None

        if self.profile:
            prof_ctx = self._profile_loop(base_args, base_kwargs, device)

        times = self._time_loop(base_args, base_kwargs, device)
        result = self._aggregate(times, prof_ctx)

        return result

    def _time_loop(self, base_args, base_kwargs, device) -> list[float]:
        """Return a list of millisecond latencies of length ``self.iters``."""
        # warm-up (not timed)
        for i in range(self.warmup):

            args, kwargs = self._clone((base_args, base_kwargs))
            self.bench_obj.before_epoch(i)

            out = self.bench_obj.forward(*args, **kwargs)

            if self.backward and isinstance(out, (LNSTensor, torch.Tensor)):
                out.sum().backward()

            self.bench_obj.post_forward(out)
            self.bench_obj.after_epoch(i)

        _sync(device)

        times: list[float] = []
        for j in range(self.iters):

            args, kwargs = self._clone((base_args, base_kwargs))
            self.bench_obj.before_epoch(i + j)

            t0 = time.perf_counter()
            out = self.bench_obj.forward(*args, **kwargs)

            if self.backward and isinstance(out, (LNSTensor, torch.Tensor)):
                out.sum().backward()

            _sync(device)

            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000) # seconds -> milliseconds

            self.bench_obj.post_forward(out)
            self.bench_obj.after_epoch(i)

        return times

    def _profile_loop(self, base_args, base_kwargs, device: torch.device) -> torch.profiler.profile:
        """Profile a single iteration and return the populated Profile object."""
        activities = [ProfilerActivity.CPU]
        if device.type == "cuda":
            activities.append(ProfilerActivity.CUDA)

        args, kwargs = self._clone((base_args, base_kwargs))
        self.bench_obj.before_epoch(0)

        with profile(
            activities=activities,
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
            experimental_config=torch._C._profiler._ExperimentalConfig(verbose=True),
        ) as prof:
            out = self.bench_obj.forward(*args, **kwargs)

            if self.backward and isinstance(out, (LNSTensor, torch.Tensor)):
                out.sum().backward()

            _sync(device)

        self.bench_obj.post_forward(out)
        self.bench_obj.after_epoch(0)

        return prof

    def _construct_inputs(self) -> Tuple[Tuple, Dict]:
        """Resolve ``Benchmark.make_inputs`` return value to ``(args, kwargs)``."""
        maybe_inputs = self.bench_obj.make_inputs()
        if isinstance(maybe_inputs, tuple):
            return maybe_inputs, {}
        elif isinstance(maybe_inputs, dict):
            return maybe_inputs
        else:
            raise ValueError("make_inputs must return tuple or dict")

    def _clone(self, data: Any) -> Any:
        """Recursively clone tensors contained in *data*. Scalars and non-tensor objects are returned as-is."""
        if isinstance(data, (torch.Tensor, LNSTensor)):
            return data.detach().clone().requires_grad_(data.requires_grad)

        if isinstance(data, list):
            return [self._clone(x) for x in data]

        if isinstance(data, tuple):
            return tuple(self._clone(x) for x in data)

        if isinstance(data, dict):
            return {k: self._clone(v) for k, v in data.items()}

        return data

    def _aggregate(self, times: list[float], prof: torch.profiler.profile) -> BenchResult:
        """
        Convert raw timing list (and optional profiler output) into BenchResult.
        """
        wall_ms = float(np.mean(times))
        p50, p90, p99 = np.percentile(times, [50, 90, 99])

        cpu_us = cpu_mem_mb = cuda_us = cuda_mem_mb = 0
        if prof is not None:
            # torch.profiler metrics are in micro-seconds
            key_agg = prof.key_averages(group_by_stack_n=5)
            cpu_us  = sum(e.self_cpu_time_total for e in key_agg)
            cpu_mem_mb = sum(e.cpu_memory_usage for e in key_agg) / (1024 * 1024) # convert to MB
            if torch.cuda.is_available():
                cuda_us  = sum(e.self_cuda_time_total for e in key_agg)
                cuda_mem_mb = sum(e.cuda_memory_usage for e in key_agg) / (1024 * 1024) # convert to MB

        return BenchResult(prof, wall_ms, p50, p90, p99, cpu_us, cpu_mem_mb, cuda_us, cuda_mem_mb)
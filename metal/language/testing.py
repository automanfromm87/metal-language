"""metal.language.testing - Kernel benchmarking utilities

Provides do_bench() similar to triton.testing.do_bench for measuring
kernel execution time using Metal GPU timestamps.
"""

import statistics


# Thread-local bench run count, checked by _KernelLauncher
_bench_runs = None


def do_bench(fn, warmup=5, rep=20, total_bytes=None):
    """Benchmark a kernel launch, return median GPU time in milliseconds.

    Args:
        fn: callable that launches a kernel, e.g.
            lambda: my_kernel[grid](x, y, out, n, BLOCK_SIZE=256)
        warmup: number of warmup iterations (discarded)
        rep: number of measured iterations
        total_bytes: if given, also computes and returns bandwidth in GB/s

    Returns:
        float: median time in ms
        or (float, float): (median_ms, gbps) if total_bytes is given
    """
    global _bench_runs
    n_total = warmup + rep

    # Set the global so _KernelLauncher picks it up
    _bench_runs = n_total
    try:
        result = fn()
    finally:
        _bench_runs = None

    # result is (result_arrays, gpu_times) when bench mode is active
    if not isinstance(result, tuple) or len(result) != 2:
        raise RuntimeError("do_bench: kernel did not return timing data")

    _result_arrays, gpu_times = result

    # Discard warmup, keep rep measurements
    measured = gpu_times[warmup:]
    median_s = statistics.median(measured)
    median_ms = median_s * 1000.0

    if total_bytes is not None:
        gbps = total_bytes / (median_s * 1e9) if median_s > 0 else float('inf')
        return median_ms, gbps

    return median_ms

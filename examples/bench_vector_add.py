"""Benchmark vector addition kernel"""

import numpy as np
import metal.language as ml


@ml.kernel
def add_kernel(
    x_ptr, y_ptr, out_ptr,
    n_elements,
    BLOCK_SIZE: ml.constexpr,
):
    pid = ml.program_id(0)
    offsets = pid * BLOCK_SIZE + ml.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = ml.load(x_ptr + offsets, mask=mask)
    y = ml.load(y_ptr + offsets, mask=mask)
    ml.store(out_ptr + offsets, x + y, mask=mask)


n = 1_000_000
x = np.random.randn(n).astype(np.float32)
y = np.random.randn(n).astype(np.float32)
out = np.zeros(n, dtype=np.float32)

grid = (ml.cdiv(n, 256),)
total_bytes = 3 * n * 4  # read x, y + write out

ms, gbps = ml.testing.do_bench(
    lambda: add_kernel[grid](x, y, out, n, BLOCK_SIZE=256),
    warmup=5,
    rep=20,
    total_bytes=total_bytes,
)

print(f"add_kernel (n={n}): {ms:.3f} ms, {gbps:.2f} GB/s")

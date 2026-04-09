"""Triton-style vector addition - Metal Language"""

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


print("=== Metal Shader ===")
add_kernel.show_metal(BLOCK_SIZE=256)

print("=== GPU Compute ===")
n = 1000
x = np.random.randn(n).astype(np.float32)
y = np.random.randn(n).astype(np.float32)
out = np.zeros(n, dtype=np.float32)

grid = (ml.cdiv(n, 256),)
result_arrays = add_kernel[grid](x, y, out, n, BLOCK_SIZE=256)
out = result_arrays[2]

print(f"n = {n}")
print(f"first 5 results: {out[:5]}")
print(f"expected first 5: {(x + y)[:5]}")
print(f"correct = {np.allclose(out, x + y)}")

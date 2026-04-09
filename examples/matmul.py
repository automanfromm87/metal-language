"""Tiled matrix multiplication - 2D grid + loop accumulation"""

import numpy as np
import metal.language as ml


@ml.kernel
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    BLOCK_SIZE: ml.constexpr,
):
    pid_m = ml.program_id(0)
    pid_n = ml.program_id(1)
    tid_val = ml.arange(0, BLOCK_SIZE)

    row = pid_m * BLOCK_SIZE + tid_val
    col = pid_n
    mask = row < M

    acc = 0.0
    for k in range(K):
        a_val = ml.load(a_ptr + row * K + k, mask=mask)
        b_val = ml.load(b_ptr + k * N + col)
        acc = acc + a_val * b_val

    ml.store(c_ptr + row * N + col, acc, mask=mask)


M, N, K = 64, 48, 32
BLOCK = 64

print("=== GPU Matrix Multiplication ===")
a = np.random.randn(M, K).astype(np.float32)
b = np.random.randn(K, N).astype(np.float32)
c = np.zeros((M, N), dtype=np.float32)

grid = (ml.cdiv(M, BLOCK), N)
result = matmul_kernel[grid](a.flatten(), b.flatten(), c.flatten(), M, N, K, BLOCK_SIZE=BLOCK)

gpu_c = result[2].reshape(M, N)
expected = a @ b

print(f"A: {M}x{K}, B: {K}x{N}, C: {M}x{N}")
print(f"max error = {np.max(np.abs(gpu_c - expected)):.6f}")
print(f"correct = {np.allclose(gpu_c, expected, atol=1e-3)}")

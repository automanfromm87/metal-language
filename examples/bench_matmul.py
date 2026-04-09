"""Benchmark: naive vs optimized Metal matmul vs CPU"""

import numpy as np
import time
import metal.language as ml


# --- Naive GPU matmul: 1 column per threadgroup ---
@ml.kernel
def matmul_naive(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    BLOCK_M: ml.constexpr,
):
    pid_m = ml.program_id(0)
    pid_n = ml.program_id(1)
    tid = ml.arange(0, BLOCK_M)

    row = pid_m * BLOCK_M + tid
    col = pid_n
    mask = row < M

    acc = 0.0
    for k in range(K):
        a_val = ml.load(a_ptr + row * K + k, mask=mask)
        b_val = ml.load(b_ptr + k * N + col)
        acc = acc + a_val * b_val

    ml.store(c_ptr + row * N + col, acc, mask=mask)


# --- Optimized GPU matmul: register blocking on N (4 columns) ---
@ml.kernel
def matmul_opt4(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    BLOCK_M: ml.constexpr,
):
    pid_m = ml.program_id(0)
    pid_n = ml.program_id(1)
    tid = ml.arange(0, BLOCK_M)

    row = pid_m * BLOCK_M + tid
    col = pid_n * 4
    mask = row < M

    acc0 = 0.0
    acc1 = 0.0
    acc2 = 0.0
    acc3 = 0.0
    for k in range(K):
        a_val = ml.load(a_ptr + row * K + k, mask=mask)
        b0 = ml.load(b_ptr + k * N + col)
        b1 = ml.load(b_ptr + k * N + col + 1)
        b2 = ml.load(b_ptr + k * N + col + 2)
        b3 = ml.load(b_ptr + k * N + col + 3)
        acc0 = acc0 + a_val * b0
        acc1 = acc1 + a_val * b1
        acc2 = acc2 + a_val * b2
        acc3 = acc3 + a_val * b3

    ml.store(c_ptr + row * N + col, acc0, mask=mask)
    ml.store(c_ptr + row * N + col + 1, acc1, mask=mask)
    ml.store(c_ptr + row * N + col + 2, acc2, mask=mask)
    ml.store(c_ptr + row * N + col + 3, acc3, mask=mask)


# --- Optimized GPU matmul: register blocking on N (8 columns) ---
@ml.kernel
def matmul_opt8(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    BLOCK_M: ml.constexpr,
):
    pid_m = ml.program_id(0)
    pid_n = ml.program_id(1)
    tid = ml.arange(0, BLOCK_M)

    row = pid_m * BLOCK_M + tid
    col = pid_n * 8
    mask = row < M

    acc0 = 0.0
    acc1 = 0.0
    acc2 = 0.0
    acc3 = 0.0
    acc4 = 0.0
    acc5 = 0.0
    acc6 = 0.0
    acc7 = 0.0
    for k in range(K):
        a_val = ml.load(a_ptr + row * K + k, mask=mask)
        b0 = ml.load(b_ptr + k * N + col)
        b1 = ml.load(b_ptr + k * N + col + 1)
        b2 = ml.load(b_ptr + k * N + col + 2)
        b3 = ml.load(b_ptr + k * N + col + 3)
        b4 = ml.load(b_ptr + k * N + col + 4)
        b5 = ml.load(b_ptr + k * N + col + 5)
        b6 = ml.load(b_ptr + k * N + col + 6)
        b7 = ml.load(b_ptr + k * N + col + 7)
        acc0 = acc0 + a_val * b0
        acc1 = acc1 + a_val * b1
        acc2 = acc2 + a_val * b2
        acc3 = acc3 + a_val * b3
        acc4 = acc4 + a_val * b4
        acc5 = acc5 + a_val * b5
        acc6 = acc6 + a_val * b6
        acc7 = acc7 + a_val * b7

    ml.store(c_ptr + row * N + col, acc0, mask=mask)
    ml.store(c_ptr + row * N + col + 1, acc1, mask=mask)
    ml.store(c_ptr + row * N + col + 2, acc2, mask=mask)
    ml.store(c_ptr + row * N + col + 3, acc3, mask=mask)
    ml.store(c_ptr + row * N + col + 4, acc4, mask=mask)
    ml.store(c_ptr + row * N + col + 5, acc5, mask=mask)
    ml.store(c_ptr + row * N + col + 6, acc6, mask=mask)
    ml.store(c_ptr + row * N + col + 7, acc7, mask=mask)


# --- Benchmarking helpers ---

def bench_gpu(kernel_func, a, b, grid, block_m, warmup=3, rep=10):
    M, K = a.shape
    N = b.shape[1]
    c = np.zeros((M, N), dtype=np.float32)

    for _ in range(warmup):
        kernel_func[grid](a.flatten(), b.flatten(), c.flatten(), M, N, K, BLOCK_M=block_m)

    ms = ml.testing.do_bench(
        lambda: kernel_func[grid](a.flatten(), b.flatten(), c.flatten(), M, N, K, BLOCK_M=block_m),
        warmup=warmup, rep=rep,
    )

    result = kernel_func[grid](a.flatten(), b.flatten(), c.flatten(), M, N, K, BLOCK_M=block_m)
    gpu_c = result[2].reshape(M, N)
    return ms, gpu_c


def bench_numpy(a, b, warmup=3, rep=10):
    for _ in range(warmup):
        np.dot(a, b)
    times = []
    for _ in range(rep):
        t0 = time.perf_counter()
        c = np.dot(a, b)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
    return np.median(times), c


def bench_naive_cpu(a, b):
    """Pure Python naive matmul for fair comparison (no BLAS)"""
    M, K = a.shape
    N = b.shape[1]
    c = np.zeros((M, N), dtype=np.float32)
    t0 = time.perf_counter()
    for i in range(M):
        for j in range(N):
            s = 0.0
            for kk in range(K):
                s += float(a[i, kk]) * float(b[kk, j])
            c[i, j] = s
    t1 = time.perf_counter()
    return (t1 - t0) * 1000, c


def gflops(M, N, K, ms):
    return 2 * M * N * K / (ms * 1e-3) / 1e9


# --- Main ---

print("=" * 70)
print("Matmul Benchmark: naive GPU vs optimized GPU vs CPU (BLAS) vs CPU (naive)")
print("=" * 70)

BLOCK_M = 64

for M, N, K in [(128, 128, 128), (256, 256, 256), (512, 512, 512)]:
    a = np.random.randn(M, K).astype(np.float32)
    b = np.random.randn(K, N).astype(np.float32)
    expected = a @ b

    print(f"\n--- [{M}x{K}] x [{K}x{N}] ---")

    # Naive GPU
    grid_naive = (ml.cdiv(M, BLOCK_M), N)
    ms_naive, c_naive = bench_gpu(matmul_naive, a, b, grid_naive, BLOCK_M)
    ok_naive = np.allclose(c_naive, expected, atol=1e-2)
    print(f"  GPU naive:     {ms_naive:8.3f} ms  {gflops(M,N,K,ms_naive):8.2f} GFLOPS  correct={ok_naive}")

    # Optimized GPU (block N=4)
    grid_opt4 = (ml.cdiv(M, BLOCK_M), ml.cdiv(N, 4))
    ms_opt4, c_opt4 = bench_gpu(matmul_opt4, a, b, grid_opt4, BLOCK_M)
    ok_opt4 = np.allclose(c_opt4, expected, atol=1e-2)
    speedup4 = ms_naive / ms_opt4
    print(f"  GPU opt(N=4):  {ms_opt4:8.3f} ms  {gflops(M,N,K,ms_opt4):8.2f} GFLOPS  correct={ok_opt4}  {speedup4:.1f}x vs naive")

    # Optimized GPU (block N=8)
    grid_opt8 = (ml.cdiv(M, BLOCK_M), ml.cdiv(N, 8))
    ms_opt8, c_opt8 = bench_gpu(matmul_opt8, a, b, grid_opt8, BLOCK_M)
    ok_opt8 = np.allclose(c_opt8, expected, atol=1e-2)
    speedup8 = ms_naive / ms_opt8
    print(f"  GPU opt(N=8):  {ms_opt8:8.3f} ms  {gflops(M,N,K,ms_opt8):8.2f} GFLOPS  correct={ok_opt8}  {speedup8:.1f}x vs naive")

    # NumPy BLAS
    ms_blas, _ = bench_numpy(a, b)
    print(f"  CPU (BLAS):    {ms_blas:8.3f} ms  {gflops(M,N,K,ms_blas):8.2f} GFLOPS  (Apple Accelerate/AMX)")

    # Naive CPU (only for small sizes)
    if M <= 128:
        ms_cpu, c_cpu = bench_naive_cpu(a, b)
        ok_cpu = np.allclose(c_cpu, expected, atol=1e-2)
        gpu_vs_cpu = ms_cpu / ms_opt8
        print(f"  CPU (naive):   {ms_cpu:8.3f} ms  {gflops(M,N,K,ms_cpu):8.2f} GFLOPS  correct={ok_cpu}  GPU {gpu_vs_cpu:.1f}x faster")

print()
print("Notes:")
print("  - GPU opt reuses A loads across N columns (register blocking)")
print("  - CPU BLAS uses Apple AMX hardware accelerator (unfair comparison)")
print("  - CPU naive is pure Python triple loop (fair comparison)")

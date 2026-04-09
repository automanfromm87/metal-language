"""Softmax example - load/store + mask + reduction"""

import numpy as np
import metal.language as ml


@ml.kernel
def softmax_kernel(
    input_ptr, output_ptr,
    n_cols,
    BLOCK_SIZE: ml.constexpr,
):
    row_idx = ml.program_id(0)
    col_offsets = ml.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    row = ml.load(input_ptr + row_idx * n_cols + col_offsets, mask=mask, other=-10000.0)

    row_max = ml.max(row, axis=0)
    numerator = ml.exp(row - row_max)
    denom = ml.sum(numerator, axis=0)
    result = numerator / denom

    ml.store(output_ptr + row_idx * n_cols + col_offsets, result, mask=mask)


print("=== GPU Softmax ===")
n_rows, n_cols = 4, 10
x = np.random.randn(n_rows, n_cols).astype(np.float32)
out = np.zeros_like(x)

result_arrays = softmax_kernel[(n_rows,)](
    x.flatten(), out.flatten(), n_cols, BLOCK_SIZE=256
)
gpu_out = result_arrays[1].reshape(n_rows, n_cols)

def numpy_softmax(x):
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)

expected = numpy_softmax(x)
print(f"GPU softmax[0]: {gpu_out[0]}")
print(f"numpy[0]:       {expected[0]}")
print(f"correct = {np.allclose(gpu_out, expected, atol=1e-5)}")

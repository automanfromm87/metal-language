# Metal Language

> **This is a learning project.** It is built for educational purposes to explore GPU programming concepts, compiler IR design, and the Apple Metal framework. It is not intended for production use.

A Triton-style Python DSL that compiles to Apple Metal compute shaders. Write GPU kernels in Python with block-level programming abstractions, and run them natively on macOS Metal GPUs.

## How it works

```
@ml.kernel Python function
        |
        v
   AST parsing (ast_to_graph.py)
        |
        v
   Graph IR (graph_ir.py)        -- High-level SSA, block-level semantics
        |
        v
   Tile Lowering (tile_lower.py)  -- Expand blocks to per-thread ops
        |
        v
   Tile IR (tile_ir.py)           -- Low-level, explicit GPU concepts
        |
        v
   Metal Emit (metal_emit.py)     -- Tile IR -> Metal Shading Language
        |
        v
   Host Emit (host_emit.py)       -- Generate Objective-C runtime code
        |
        v
   clang compile + GPU execute
```

## Quick start

Requirements: macOS with Metal support, Python 3.10+, numpy.

```bash
pip install numpy
```

### Vector addition

```python
import numpy as np
import metal.language as ml

@ml.kernel
def add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: ml.constexpr):
    pid = ml.program_id(0)
    offsets = pid * BLOCK_SIZE + ml.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = ml.load(x_ptr + offsets, mask=mask)
    y = ml.load(y_ptr + offsets, mask=mask)
    ml.store(out_ptr + offsets, x + y, mask=mask)

n = 1000
x = np.random.randn(n).astype(np.float32)
y = np.random.randn(n).astype(np.float32)
out = np.zeros(n, dtype=np.float32)

grid = (ml.cdiv(n, 256),)
result = add_kernel[grid](x, y, out, n, BLOCK_SIZE=256)
out = result[2]

print(np.allclose(out, x + y))  # True
```

### Run examples

```bash
PYTHONPATH=. python3 examples/vector_add.py
PYTHONPATH=. python3 examples/softmax.py
PYTHONPATH=. python3 examples/matmul.py
PYTHONPATH=. python3 examples/bench_vector_add.py
```

## API reference

### Kernel definition

```python
@ml.kernel
def my_kernel(ptr_arg, scalar_arg, CONST: ml.constexpr):
    ...
```

- **Pointer parameters**: numpy arrays, passed as Metal device buffers
- **Scalar parameters**: `int` or `float`, passed via `setBytes`
- **constexpr parameters**: compile-time constants, baked into the shader

### Launch syntax

```python
kernel[grid](arg1, arg2, ..., BLOCK_SIZE=256)
```

`grid` is a tuple of 1-3 integers specifying threadgroup grid dimensions.

### Built-in functions

| Function | Description |
|---|---|
| `ml.program_id(axis)` | Threadgroup index in grid |
| `ml.arange(start, end)` | Per-thread tile indices |
| `ml.load(ptr + offset, mask=, other=)` | Masked memory read |
| `ml.store(ptr + offset, value, mask=)` | Masked memory write |
| `ml.atomic_add(ptr + offset, value)` | Atomic addition |
| `ml.zeros(shape)` | Zero-initialized tile |
| `ml.where(cond, x, y)` | Conditional select |
| `ml.exp`, `ml.log`, `ml.sqrt`, `ml.abs` | Element-wise math |
| `ml.sum(x, axis=0)` | Sum reduction |
| `ml.max(x, axis=0)` | Max reduction |
| `ml.cdiv(a, b)` | Ceiling division |

### Debugging

```python
kernel.show_metal(BLOCK_SIZE=256)     # Print generated Metal shader
kernel.show_graph_ir(BLOCK_SIZE=256)  # Print high-level Graph IR
kernel.show_tile_ir(BLOCK_SIZE=256)   # Print low-level Tile IR
```

### Benchmarking

```python
ms = ml.testing.do_bench(
    lambda: kernel[grid](x, y, out, n, BLOCK_SIZE=256),
    warmup=5, rep=20,
)
print(f"{ms:.3f} ms")

# With bandwidth calculation:
ms, gbps = ml.testing.do_bench(
    lambda: kernel[grid](x, y, out, n, BLOCK_SIZE=256),
    total_bytes=3 * n * 4,
)
print(f"{ms:.3f} ms, {gbps:.2f} GB/s")
```

Uses Metal GPU timestamps (`GPUStartTime` / `GPUEndTime`) for precise GPU-only timing.

## Project structure

```
metal/
  __init__.py
  language/
    __init__.py          # Public API exports
    lang.py              # ml.* sentinel functions
    kernel.py            # @ml.kernel decorator and launcher
    types.py             # Type system
    graph_ir.py          # Graph IR data structures (SSA)
    ast_to_graph.py      # Python AST -> Graph IR
    tile_ir.py           # Tile IR data structures
    tile_lower.py        # Graph IR -> Tile IR lowering
    metal_emit.py        # Tile IR -> Metal Shading Language
    host_emit.py         # Generate Objective-C host code
    compiler.py          # Compile and execute via clang + Metal
    testing.py           # do_bench() benchmarking utility

examples/
    vector_add.py        # Basic vector addition
    softmax.py           # Softmax with reductions
    matmul.py            # Tiled matrix multiplication
    bench_vector_add.py  # Performance benchmarking
```

## Limitations

- Float32 only (no half/int data types in kernels yet)
- No shared memory tiling in user kernels (shared memory is used internally for reductions)
- No `if` statements in kernel code (use `ml.where()` instead)
- Single-device only
- No autotuning

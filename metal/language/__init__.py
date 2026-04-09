"""metal.language - Triton-style Metal GPU compute DSL

Usage:
    import metal.language as ml

    @ml.kernel
    def add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: ml.constexpr):
        pid = ml.program_id(0)
        offsets = pid * BLOCK_SIZE + ml.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = ml.load(x_ptr + offsets, mask=mask)
        y = ml.load(y_ptr + offsets, mask=mask)
        ml.store(out_ptr + offsets, x + y, mask=mask)

    add_kernel[(ml.cdiv(n, 256),)](x, y, out, n, BLOCK_SIZE=256)
"""

from .types import dtype
from .lang import (
    constexpr,
    program_id,
    arange,
    load,
    store,
    atomic_add,
    zeros,
    where,
    exp, log, sqrt, abs,
    sum, max,
    cdiv,
)
from .kernel import kernel
from .kernel import kernel as jit
from . import testing

__all__ = [
    "kernel", "jit", "dtype", "constexpr", "cdiv",
    "program_id", "arange", "load", "store",
    "atomic_add", "zeros", "where",
    "exp", "log", "sqrt", "abs",
    "sum", "max",
]

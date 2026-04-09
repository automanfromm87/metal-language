"""ml namespace - Triton-style Metal programming API

These objects are sentinels/markers on the Python side and do not execute real logic.
The AST parser recognizes them and generates corresponding Metal code.
"""


class _ConstExprType:
    """Compile-time constant type marker for kernel parameter annotations"""
    def __repr__(self):
        return "ml.constexpr"


constexpr = _ConstExprType()


def program_id(axis=0):
    """Get the current program (threadgroup) ID in the grid

    Metal mapping: threadgroup_position_in_grid.x/y/z
    """
    raise RuntimeError("ml.program_id() can only be used inside @ml.kernel functions")


def arange(start, end):
    """Generate tile indices [start, start+1, ..., end-1]

    Metal mapping: thread_index_in_threadgroup (one element per thread)
    """
    raise RuntimeError("ml.arange() can only be used inside @ml.kernel functions")


def load(ptr, mask=None, other=0.0):
    """Load data from GPU memory with optional boundary mask

    Metal mapping: mask ? buffer[offset] : other
    """
    raise RuntimeError("ml.load() can only be used inside @ml.kernel functions")


def store(ptr, value, mask=None):
    """Store data to GPU memory with optional boundary mask

    Metal mapping: if (mask) buffer[offset] = value;
    """
    raise RuntimeError("ml.store() can only be used inside @ml.kernel functions")


def atomic_add(ptr, value):
    """Atomic addition

    Metal mapping: atomic_fetch_add_explicit
    """
    raise RuntimeError("ml.atomic_add() can only be used inside @ml.kernel functions")


def zeros(shape, dtype=None):
    """Create zero-initialized tile"""
    raise RuntimeError("ml.zeros() can only be used inside @ml.kernel functions")


def where(condition, x, y):
    """Conditional select: condition ? x : y

    Metal mapping: select(y, x, condition)
    """
    raise RuntimeError("ml.where() can only be used inside @ml.kernel functions")


# Math functions (mapped to Metal built-in functions when used inside kernels)
def exp(x):
    raise RuntimeError("ml.exp() can only be used inside @ml.kernel functions")

def log(x):
    raise RuntimeError("ml.log() can only be used inside @ml.kernel functions")

def sqrt(x):
    raise RuntimeError("ml.sqrt() can only be used inside @ml.kernel functions")

def abs(x):
    raise RuntimeError("ml.abs() can only be used inside @ml.kernel functions")


# Reduction operations
def sum(x, axis=0):
    """Tile sum reduction

    Metal mapping: threadgroup shared memory reduction
    """
    raise RuntimeError("ml.sum() can only be used inside @ml.kernel functions")


def max(x, axis=0):
    """Tile max reduction

    Metal mapping: threadgroup shared memory reduction
    """
    raise RuntimeError("ml.max() can only be used inside @ml.kernel functions")


# Utility functions (these work normally on the Python side)
def cdiv(a, b):
    """Ceiling division: ceil(a / b)"""
    return (a + b - 1) // b

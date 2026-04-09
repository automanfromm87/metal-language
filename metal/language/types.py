"""Metal DSL type system - Python to Metal type mapping"""

import numpy as np


class MetalType:
    def __init__(self, name, metal_name, numpy_dtype, size):
        self.name = name
        self.metal_name = metal_name
        self.numpy_dtype = numpy_dtype
        self.size = size

    def __repr__(self):
        return f"dtype.{self.name}"


class _DTypeNamespace:
    float32 = MetalType("float32", "float", np.float32, 4)
    float16 = MetalType("float16", "half", np.float16, 2)
    int32 = MetalType("int32", "int", np.int32, 4)
    uint32 = MetalType("uint32", "uint", np.uint32, 4)
    int16 = MetalType("int16", "short", np.int16, 2)
    uint16 = MetalType("uint16", "ushort", np.uint16, 2)
    int8 = MetalType("int8", "char", np.int8, 1)
    uint8 = MetalType("uint8", "uchar", np.uint8, 1)

    _by_name = {}
    _by_numpy = {}

    @classmethod
    def _init_lookups(cls):
        for attr in dir(cls):
            val = getattr(cls, attr)
            if isinstance(val, MetalType):
                cls._by_name[attr] = val
                cls._by_numpy[val.numpy_dtype] = val

    @classmethod
    def from_numpy(cls, np_dtype):
        np_dtype = np.dtype(np_dtype)
        for metal_type in cls._by_name.values():
            if np.dtype(metal_type.numpy_dtype) == np_dtype:
                return metal_type
        raise ValueError(f"Unsupported numpy dtype: {np_dtype}")


dtype = _DTypeNamespace()
dtype._init_lookups()

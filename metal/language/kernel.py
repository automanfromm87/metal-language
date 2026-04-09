"""Metal DSL kernel decorator - Triton-style block-level kernel definition"""

import ast
import hashlib
import inspect
import os
import textwrap
import numpy as np

from .types import dtype
from .lang import _ConstExprType


class KernelParam:
    """Kernel parameter definition"""
    POINTER = "pointer"    # numpy array -> device buffer
    SCALAR = "scalar"      # int/float scalar -> setBytes
    CONSTEXPR = "constexpr"  # compile-time constant -> #define

    def __init__(self, name, kind, buffer_index=None):
        self.name = name
        self.kind = kind
        self.buffer_index = buffer_index  # Only for pointer type


class KernelDef:
    """Parsed kernel definition with grid launch support"""

    def __init__(self, name, params, body_ast, source, constants=None):
        self.name = name
        self.params = params          # List[KernelParam]
        self.body_ast = body_ast
        self.source = source
        self.constants = constants or {}

    def __getitem__(self, grid):
        """kernel[grid] syntax returns a launcher"""
        if isinstance(grid, int):
            grid = (grid,)
        return _KernelLauncher(self, grid)

    def show_metal(self, **constexpr_vals):
        """Print the generated Metal shader code"""
        from .ast_to_graph import ast_to_graph
        from .tile_lower import lower
        from .metal_emit import emit as metal_emit

        graph_ir = ast_to_graph(self, constexpr_vals)
        tile_kernel = lower(graph_ir, (1,), _get_block_size(constexpr_vals))
        code = metal_emit(tile_kernel)
        print(code)
        return code

    def show_graph_ir(self, **constexpr_vals):
        """Print the Graph IR"""
        from .ast_to_graph import ast_to_graph
        graph_ir = ast_to_graph(self, constexpr_vals)
        dump = graph_ir.dump()
        print(dump)
        return dump

    def show_tile_ir(self, grid=(1,), **constexpr_vals):
        """Print the Tile IR"""
        from .ast_to_graph import ast_to_graph
        from .tile_lower import lower

        graph_ir = ast_to_graph(self, constexpr_vals)
        tile_kernel = lower(graph_ir, grid, _get_block_size(constexpr_vals))
        dump = tile_kernel.dump()
        print(dump)
        return dump


class _KernelLauncher:
    """Intermediate object for kernel[grid](...)"""

    def __init__(self, kernel_def, grid):
        self.kernel_def = kernel_def
        self.grid = grid

    def __call__(self, *args, **kwargs):
        from .compiler import compile_and_run
        from .testing import _bench_runs

        kd = self.kernel_def
        params = kd.params

        # Separate constexpr kwargs
        constexpr_vals = {}
        for p in params:
            if p.kind == KernelParam.CONSTEXPR:
                if p.name in kwargs:
                    constexpr_vals[p.name] = kwargs.pop(p.name)
                elif p.name in kd.constants:
                    constexpr_vals[p.name] = kd.constants[p.name]
                else:
                    raise ValueError(f"Missing constexpr parameter: {p.name}")

        # Copy parameter list to avoid mutating shared KernelDef.params
        import copy
        params = [copy.copy(p) for p in params]
        kd = copy.copy(kd)
        kd.params = params

        non_constexpr = [p for p in params if p.kind != KernelParam.CONSTEXPR]
        if len(args) != len(non_constexpr):
            raise ValueError(
                f"Kernel '{kd.name}' expects {len(non_constexpr)} arguments, "
                f"but got {len(args)}"
            )

        pointer_arrays = []
        scalar_values = []

        buf_idx = 0
        for p, val in zip(non_constexpr, args):
            if isinstance(val, np.ndarray):
                p.kind = KernelParam.POINTER
                p.buffer_index = buf_idx
                buf_idx += 1
                pointer_arrays.append(val)
            elif isinstance(val, (int, float)):
                p.kind = KernelParam.SCALAR
                scalar_values.append(val)
            else:
                raise ValueError(f"Unsupported type for parameter '{p.name}': {type(val)}")

        # IR pipeline: AST -> Graph IR -> Tile IR -> Metal + Host
        from .ast_to_graph import ast_to_graph
        from .tile_lower import lower
        from .metal_emit import emit as metal_emit
        from .host_emit import emit_host

        n_runs = _bench_runs if _bench_runs else 1
        graph_ir = ast_to_graph(kd, constexpr_vals)
        tile_kernel = lower(graph_ir, self.grid, _get_block_size(constexpr_vals))
        metal_code = metal_emit(tile_kernel)
        host_code = emit_host(tile_kernel, metal_code, n_runs=n_runs)

        # Compile and run
        return compile_and_run(
            metal_code, host_code,
            pointer_arrays, scalar_values,
            constexpr_vals, kd.name,
            n_runs=n_runs,
        )


def _get_block_size(constexpr_vals, default=256):
    for name, val in constexpr_vals.items():
        if "BLOCK" in name.upper():
            return val
    return default


def kernel(func):
    """@ml.kernel decorator: convert a Python function to a Metal compute kernel

    Parameter types are distinguished by annotations:
    - No annotation: pointer parameter (numpy array)
    - int/float annotation: scalar parameter
    - ml.constexpr annotation: compile-time constant
    """
    source = inspect.getsource(func)
    source = textwrap.dedent(source)

    tree = ast.parse(source)
    func_def = tree.body[0]

    if not isinstance(func_def, ast.FunctionDef):
        raise ValueError("@ml.kernel can only be used on function definitions")

    # Parse parameters
    params = []

    for arg in func_def.args.args:
        annotation = arg.annotation

        if annotation is not None and _is_constexpr_annotation(annotation):
            params.append(KernelParam(arg.arg, KernelParam.CONSTEXPR))
        else:
            # pointer vs scalar is determined at call time based on value type
            params.append(KernelParam(arg.arg, KernelParam.POINTER))

    # Capture external constants
    constants = {}
    frame = inspect.currentframe().f_back
    caller_locals = frame.f_locals
    caller_globals = frame.f_globals

    param_names = {p.name for p in params}
    free_names = _find_free_names(func_def, param_names)
    for name in free_names:
        if name in caller_locals:
            val = caller_locals[name]
        elif name in caller_globals:
            val = caller_globals[name]
        else:
            continue
        if isinstance(val, (int, float)):
            constants[name] = val

    return KernelDef(
        name=func_def.name,
        params=params,
        body_ast=func_def.body,
        source=source,
        constants=constants,
    )


def _is_constexpr_annotation(node):
    """Check if an annotation is ml.constexpr"""
    if isinstance(node, ast.Attribute):
        if isinstance(node.value, ast.Name) and node.attr == "constexpr":
            return True
    if isinstance(node, ast.Name) and node.id == "constexpr":
        return True
    return False


def _find_free_names(func_def, param_names):
    """Find variable names referenced but not defined within the function"""
    defined = set(param_names)
    defined.update(["ml", "range", "math", "True", "False", "None"])
    used = set()

    class NameCollector(ast.NodeVisitor):
        def visit_Name(self, node):
            if node.id not in defined:
                used.add(node.id)
            self.generic_visit(node)

        def visit_Assign(self, node):
            self.visit(node.value)
            for target in node.targets:
                if isinstance(target, ast.Name):
                    defined.add(target.id)

        def visit_For(self, node):
            if isinstance(node.target, ast.Name):
                defined.add(node.target.id)
            self.generic_visit(node)

    NameCollector().visit(func_def)
    return used

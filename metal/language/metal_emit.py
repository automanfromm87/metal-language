"""Metal Emit - Tile IR -> Metal Shading Language text

Thin translation layer; each Tile IR node maps directly to Metal syntax.
"""

from . import tile_ir as T


def emit(tile_kernel):
    """Translate a TileKernel to a Metal shader source string"""
    emitter = _MetalEmitter(tile_kernel)
    return emitter.emit()


class _MetalEmitter:
    def __init__(self, tk):
        self.tk = tk

    def emit(self):
        lines = []
        lines.append("#include <metal_stdlib>")
        lines.append("using namespace metal;")
        lines.append("")

        params = self._gen_params()
        lines.append(f"kernel void {self.tk.name}(")
        lines.append("    " + ",\n    ".join(params))
        lines.append(") {")

        for sa in self.tk.shared:
            lines.append(f"    threadgroup float {sa.name}[{sa.size}];")

        for stmt in self.tk.body:
            lines.extend(self._emit_stmt(stmt, indent=1))

        lines.append("}")
        return "\n".join(lines)

    def _gen_params(self):
        """Generate Metal kernel parameter list"""
        parts = []
        buf_idx = 0
        for name, is_const in self.tk.buf_params:
            qualifier = "const device" if is_const else "device"
            parts.append(f"{qualifier} float* {name} [[buffer({buf_idx})]]")
            buf_idx += 1

        for name in self.tk.scalar_params:
            parts.append(f"constant uint& {name} [[buffer({buf_idx})]]")
            buf_idx += 1

        # Built-in parameters
        parts.append("uint3 tgid [[threadgroup_position_in_grid]]")
        parts.append("uint tid [[thread_index_in_threadgroup]]")
        return parts

    def _emit_stmt(self, stmt, indent):
        """Emit a single statement, return list of lines"""
        prefix = "    " * indent

        if isinstance(stmt, (T.LetFloat, T.LetUint, T.LetAuto)):
            type_kw = {T.LetFloat: "float", T.LetUint: "uint", T.LetAuto: "auto"}[type(stmt)]
            return [f"{prefix}{type_kw} {stmt.name} = {self._emit_expr(stmt.expr)};"]

        elif isinstance(stmt, T.BufStore):
            val = self._emit_expr(stmt.value)
            idx = self._emit_expr(stmt.index)
            if stmt.guard:
                guard = self._emit_expr(stmt.guard)
                return [f"{prefix}if ({guard}) {stmt.buf_name}[{idx}] = {val};"]
            return [f"{prefix}{stmt.buf_name}[{idx}] = {val};"]

        elif isinstance(stmt, T.SharedStore):
            val = self._emit_expr(stmt.value)
            idx = self._emit_expr(stmt.index)
            return [f"{prefix}{stmt.name}[{idx}] = {val};"]

        elif isinstance(stmt, T.Barrier):
            return [f"{prefix}threadgroup_barrier(mem_flags::mem_threadgroup);"]

        elif isinstance(stmt, T.ForStmt):
            return self._emit_for(stmt, indent)

        elif isinstance(stmt, T.IfStmt):
            lines = [f"{prefix}if ({self._emit_expr(stmt.cond)}) {{"]
            for s in stmt.body:
                lines.extend(self._emit_stmt(s, indent + 1))
            lines.append(f"{prefix}}}")
            return lines

        elif isinstance(stmt, T.Accum):
            return [f"{prefix}{stmt.name} += {self._emit_expr(stmt.expr)};"]

        elif isinstance(stmt, T.Assign):
            return [f"{prefix}{stmt.name} = {self._emit_expr(stmt.expr)};"]

        return [f"{prefix}/* unknown: {type(stmt).__name__} */"]

    def _emit_for(self, stmt, indent):
        """Emit a for loop"""
        prefix = "    " * indent
        var = stmt.var
        start = self._emit_expr(stmt.start)
        end = self._emit_expr(stmt.end)

        if stmt.step == -1:
            # Special: reduction s >>= 1 pattern
            lines = [f"{prefix}for (uint {var} = {start}; {var} > {end}; {var} >>= 1) {{"]
        elif stmt.step < 0:
            lines = [f"{prefix}for (int {var} = {start}; {var} > {end}; {var} += {stmt.step}) {{"]
        elif stmt.step == 1:
            lines = [f"{prefix}for (uint {var} = {start}; {var} < {end}; {var}++) {{"]
        else:
            lines = [f"{prefix}for (uint {var} = {start}; {var} < {end}; {var} += {stmt.step}) {{"]

        for s in stmt.body:
            lines.extend(self._emit_stmt(s, indent + 1))
        lines.append(f"{prefix}}}")
        return lines

    def _emit_expr(self, expr):
        """Emit an expression"""
        if expr is None:
            return "0"

        if isinstance(expr, T.ThreadIdx):
            return "tid"

        elif isinstance(expr, T.GroupIdx):
            return f"tgid.{expr.dim}"

        elif isinstance(expr, T.BufLoad):
            idx = self._emit_expr(expr.index)
            return f"{expr.buf_name}[{idx}]"

        elif isinstance(expr, T.SharedLoad):
            idx = self._emit_expr(expr.index)
            return f"{expr.name}[{idx}]"

        elif isinstance(expr, T.BinOp):
            left = self._emit_expr(expr.left)
            right = self._emit_expr(expr.right)
            op = expr.op
            if op == "max":
                return f"metal::max({left}, {right})"
            return f"({left} {op} {right})"

        elif isinstance(expr, T.UnaryOp):
            operand = self._emit_expr(expr.operand)
            op = expr.op
            if op in ("exp", "log", "sqrt", "abs"):
                return f"metal::{op}({operand})"
            if op == "!":
                return f"(!{operand})"
            return f"{op}({operand})"

        elif isinstance(expr, T.Cond):
            c = self._emit_expr(expr.cond)
            t = self._emit_expr(expr.then_val)
            f = self._emit_expr(expr.else_val)
            return f"({c} ? {t} : {f})"

        elif isinstance(expr, T.Var):
            return expr.name

        elif isinstance(expr, T.IntLit):
            return str(expr.value)

        elif isinstance(expr, T.FloatLit):
            v = expr.value
            if v == float('inf'):
                return "INFINITY"
            if v == float('-inf'):
                return "(-INFINITY)"
            return f"{v}f"

        return f"/* unknown: {type(expr).__name__} */"

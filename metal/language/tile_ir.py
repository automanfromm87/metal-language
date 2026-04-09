"""Tile IR - Low-level intermediate representation with explicit GPU concepts

All block-level abstractions are expanded to thread-level operations.
Thread index, threadgroup index, shared memory, and barriers are all explicit.

This is the final IR layer before Metal code emission; each node maps directly to Metal syntax.
"""

from dataclasses import dataclass, field


# === Expressions (per-thread scalars) ===

@dataclass
class TileExpr:
    pass

@dataclass
class ThreadIdx(TileExpr):
    """thread_index_in_threadgroup"""
    dim: str = "x"

@dataclass
class GroupIdx(TileExpr):
    """threadgroup_position_in_grid"""
    dim: str = "x"

@dataclass
class BufLoad(TileExpr):
    """Device buffer read: buf[index]"""
    buf_name: str
    index: TileExpr = None

@dataclass
class SharedLoad(TileExpr):
    """Threadgroup shared memory read: shared[index]"""
    name: str
    index: TileExpr = None

@dataclass
class BinOp(TileExpr):
    """Binary operation: left op right"""
    op: str = ""
    left: TileExpr = None
    right: TileExpr = None

@dataclass
class UnaryOp(TileExpr):
    """Unary operation: op(operand)"""
    op: str = ""
    operand: TileExpr = None

@dataclass
class Cond(TileExpr):
    """Conditional expression: cond ? then_val : else_val"""
    cond: TileExpr = None
    then_val: TileExpr = None
    else_val: TileExpr = None

@dataclass
class Var(TileExpr):
    """Local variable reference"""
    name: str = ""

@dataclass
class IntLit(TileExpr):
    """Integer literal"""
    value: int = 0

@dataclass
class FloatLit(TileExpr):
    """Float literal"""
    value: float = 0.0


# === Statements ===

@dataclass
class TileStmt:
    pass

@dataclass
class LetFloat(TileStmt):
    """Float variable declaration: float name = expr;"""
    name: str = ""
    expr: TileExpr = None

@dataclass
class LetUint(TileStmt):
    """Uint variable declaration: uint name = expr;"""
    name: str = ""
    expr: TileExpr = None

@dataclass
class LetAuto(TileStmt):
    """Auto variable declaration: auto name = expr;"""
    name: str = ""
    expr: TileExpr = None

@dataclass
class BufStore(TileStmt):
    """Buffer write: buf[index] = value; (optional guard)"""
    buf_name: str = ""
    index: TileExpr = None
    value: TileExpr = None
    guard: TileExpr = None

@dataclass
class SharedStore(TileStmt):
    """Shared memory write: shared[index] = value;"""
    name: str = ""
    index: TileExpr = None
    value: TileExpr = None

@dataclass
class Barrier(TileStmt):
    """threadgroup_barrier(mem_flags::mem_threadgroup);"""
    pass

@dataclass
class ForStmt(TileStmt):
    """For loop"""
    var: str = ""
    start: TileExpr = None
    end: TileExpr = None
    step: int = 1
    body: list = field(default_factory=list)

@dataclass
class IfStmt(TileStmt):
    """Conditional statement: if (cond) { body }"""
    cond: TileExpr = None
    body: list = field(default_factory=list)

@dataclass
class Accum(TileStmt):
    """Accumulate: name += expr;"""
    name: str = ""
    expr: TileExpr = None

@dataclass
class Assign(TileStmt):
    """Assignment: name = expr;"""
    name: str = ""
    expr: TileExpr = None


# === Kernel structure ===

@dataclass
class SharedAlloc:
    """Threadgroup shared memory allocation"""
    name: str
    size: int       # Number of elements

@dataclass
class TileKernel:
    """A complete compute kernel"""
    name: str
    buf_params: list = field(default_factory=list)     # [(name, is_const)]
    scalar_params: list = field(default_factory=list)  # [name]
    grid: tuple = (1, 1, 1)
    block: tuple = (256, 1, 1)
    shared: list = field(default_factory=list)          # [SharedAlloc]
    body: list = field(default_factory=list)            # [TileStmt]

    def dump(self):
        lines = [f"TileKernel({self.name}):"]
        lines.append(f"  grid={self.grid}, block={self.block}")
        lines.append(f"  bufs: {self.buf_params}")
        lines.append(f"  scalars: {self.scalar_params}")
        if self.shared:
            lines.append(f"  shared: {[(s.name, s.size) for s in self.shared]}")
        lines.append("  body:")
        for stmt in self.body:
            lines.extend(_dump_stmt(stmt, indent=4))
        return "\n".join(lines)


def _dump_stmt(stmt, indent=4):
    prefix = " " * indent
    if isinstance(stmt, LetFloat):
        return [f"{prefix}float {stmt.name} = {_dump_expr(stmt.expr)};"]
    elif isinstance(stmt, LetUint):
        return [f"{prefix}uint {stmt.name} = {_dump_expr(stmt.expr)};"]
    elif isinstance(stmt, LetAuto):
        return [f"{prefix}auto {stmt.name} = {_dump_expr(stmt.expr)};"]
    elif isinstance(stmt, BufStore):
        guard = f" [if {_dump_expr(stmt.guard)}]" if stmt.guard else ""
        return [f"{prefix}{stmt.buf_name}[{_dump_expr(stmt.index)}] = {_dump_expr(stmt.value)};{guard}"]
    elif isinstance(stmt, SharedStore):
        return [f"{prefix}shared.{stmt.name}[{_dump_expr(stmt.index)}] = {_dump_expr(stmt.value)};"]
    elif isinstance(stmt, Barrier):
        return [f"{prefix}barrier;"]
    elif isinstance(stmt, ForStmt):
        lines = [f"{prefix}for {stmt.var} in [{_dump_expr(stmt.start)}, {_dump_expr(stmt.end)}) step {stmt.step}:"]
        for s in stmt.body:
            lines.extend(_dump_stmt(s, indent + 4))
        return lines
    elif isinstance(stmt, Accum):
        return [f"{prefix}{stmt.name} += {_dump_expr(stmt.expr)};"]
    elif isinstance(stmt, Assign):
        return [f"{prefix}{stmt.name} = {_dump_expr(stmt.expr)};"]
    return [f"{prefix}??? {type(stmt).__name__}"]


def _dump_expr(expr):
    if expr is None:
        return "?"
    if isinstance(expr, ThreadIdx):
        return f"tid.{expr.dim}"
    elif isinstance(expr, GroupIdx):
        return f"gid.{expr.dim}"
    elif isinstance(expr, BufLoad):
        return f"{expr.buf_name}[{_dump_expr(expr.index)}]"
    elif isinstance(expr, SharedLoad):
        return f"shared.{expr.name}[{_dump_expr(expr.index)}]"
    elif isinstance(expr, BinOp):
        return f"({_dump_expr(expr.left)} {expr.op} {_dump_expr(expr.right)})"
    elif isinstance(expr, UnaryOp):
        return f"{expr.op}({_dump_expr(expr.operand)})"
    elif isinstance(expr, Cond):
        return f"({_dump_expr(expr.cond)} ? {_dump_expr(expr.then_val)} : {_dump_expr(expr.else_val)})"
    elif isinstance(expr, Var):
        return expr.name
    elif isinstance(expr, IntLit):
        return str(expr.value)
    elif isinstance(expr, FloatLit):
        return f"{expr.value}f"
    return f"???{type(expr).__name__}"

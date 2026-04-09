"""Graph IR - High-level SSA intermediate representation with block-level semantics

Each value is identified by a unique register number.
Types distinguish scalar (single thread) and tile (one element per thread).

Example (vector_add):
    %0 = program_id(axis=0)           : scalar_uint
    %1 = const_int(256)               : scalar_uint
    %2 = mul(%0, %1)                  : scalar_uint
    %3 = arange(0, 256)               : tile_uint
    %4 = add(%2, %3)                  : tile_uint
    %5 = lt(%4, @n_elements)          : tile_bool
    %6 = load(@x_ptr, %4, mask=%5)    : tile_float
    %7 = load(@y_ptr, %4, mask=%5)    : tile_float
    %8 = add(%6, %7)                  : tile_float
         store(@out_ptr, %4, %8, mask=%5)
"""

from dataclasses import dataclass, field


class VType:
    """Value types"""
    SCALAR_UINT = "scalar_uint"
    SCALAR_FLOAT = "scalar_float"
    SCALAR_BOOL = "scalar_bool"
    TILE_UINT = "tile_uint"
    TILE_FLOAT = "tile_float"
    TILE_BOOL = "tile_bool"

    @staticmethod
    def is_tile(vtype):
        return vtype.startswith("tile_")

    @staticmethod
    def is_scalar(vtype):
        return vtype.startswith("scalar_")

    @staticmethod
    def is_bool(vtype):
        return vtype.endswith("_bool")

    @staticmethod
    def is_float(vtype):
        return vtype.endswith("_float")

    @staticmethod
    def promote(a, b):
        """Type promotion: scalar + tile -> tile, uint + float -> float"""
        a_tile = VType.is_tile(a)
        b_tile = VType.is_tile(b)
        a_float = VType.is_float(a)
        b_float = VType.is_float(b)

        is_tile = a_tile or b_tile
        is_float = a_float or b_float
        prefix = "tile_" if is_tile else "scalar_"
        suffix = "float" if is_float else "uint"
        return prefix + suffix


class Op:
    """Graph IR opcodes"""
    # Data sources
    PROGRAM_ID = "program_id"
    ARANGE = "arange"
    CONST_INT = "const_int"
    CONST_FLOAT = "const_float"
    PARAM_REF = "param_ref"       # Kernel parameter reference (pointer or scalar)

    # Arithmetic
    ADD = "add"
    SUB = "sub"
    MUL = "mul"
    DIV = "div"
    MOD = "mod"

    # Comparison
    LT = "lt"
    LE = "le"
    GT = "gt"
    GE = "ge"
    EQ = "eq"
    NE = "ne"

    # Logic
    AND = "and"
    OR = "or"
    NOT = "not"

    # Memory
    LOAD = "load"
    STORE = "store"
    ATOMIC_ADD = "atomic_add"

    # Reduction
    REDUCE_SUM = "reduce_sum"
    REDUCE_MAX = "reduce_max"

    # Math
    EXP = "exp"
    LOG = "log"
    SQRT = "sqrt"
    ABS = "abs"

    # Control flow
    WHERE = "where"


@dataclass
class IRNode:
    """An IR instruction: %dst = op(args...) : vtype"""
    dst: int                      # Destination register (-1 for void ops like store)
    op: str                       # Op enum value
    args: list                    # Operand register numbers
    vtype: str                    # VType
    attrs: dict = field(default_factory=dict)
    # Common attrs keys:
    #   axis: int             (program_id)
    #   value: int|float      (const_int, const_float)
    #   start, end: int       (arange)
    #   param_name: str       (param_ref)
    #   param_kind: str       (param_ref: "pointer"/"scalar")
    #   mask_reg: int         (load, store)
    #   other_val: float      (load)


@dataclass
class ForLoop:
    """For loop"""
    var_reg: int                  # Loop variable register
    start: int                    # Start value (register number or immediate)
    end: int                      # End value
    step: int = 1
    body: list = field(default_factory=list)  # List[IRNode | ForLoop]
    is_start_imm: bool = True     # Whether start is an immediate
    is_end_imm: bool = False      # Whether end is an immediate
    carried_vars: dict = field(default_factory=dict)  # {name: (init_reg, loop_reg)}


@dataclass
class KernelIR:
    """Complete Graph IR for a kernel"""
    name: str
    params: list                  # [(name, kind)] - "pointer"/"scalar"/"constexpr"
    body: list                    # List[IRNode | ForLoop]
    constexpr_vals: dict
    next_reg: int = 0             # Next available register number
    reg_types: dict = field(default_factory=dict)  # reg -> VType

    def alloc_reg(self, vtype):
        """Allocate a new register"""
        reg = self.next_reg
        self.next_reg += 1
        self.reg_types[reg] = vtype
        return reg

    def emit(self, op, args, vtype, attrs=None):
        """Append an instruction and return the destination register"""
        reg = self.alloc_reg(vtype)
        self.body.append(IRNode(reg, op, args, vtype, attrs or {}))
        return reg

    def emit_void(self, op, args, attrs=None):
        """Append a void instruction (e.g. store)"""
        self.body.append(IRNode(-1, op, args, "void", attrs or {}))

    def dump(self):
        """Print IR"""
        lines = [f"kernel {self.name}:"]
        lines.append(f"  params: {self.params}")
        lines.append(f"  constexpr: {self.constexpr_vals}")
        lines.append(f"  body:")
        for stmt in self.body:
            lines.extend(self._dump_stmt(stmt, indent=4))
        return "\n".join(lines)

    def _dump_stmt(self, stmt, indent=4):
        prefix = " " * indent
        if isinstance(stmt, IRNode):
            args_str = ", ".join(
                f"%{a}" if isinstance(a, int) and a >= 0 else str(a)
                for a in stmt.args
            )
            attrs_str = ""
            if stmt.attrs:
                attrs_str = " " + str(stmt.attrs)
            if stmt.dst >= 0:
                return [f"{prefix}%{stmt.dst} = {stmt.op}({args_str}) : {stmt.vtype}{attrs_str}"]
            else:
                return [f"{prefix}{stmt.op}({args_str}){attrs_str}"]
        elif isinstance(stmt, ForLoop):
            lines = [f"{prefix}for %{stmt.var_reg} in [{stmt.start}, {stmt.end}) step {stmt.step}:"]
            for s in stmt.body:
                lines.extend(self._dump_stmt(s, indent + 4))
            return lines
        return [f"{prefix}???"]

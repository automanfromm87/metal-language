"""Tile Lowering - Graph IR -> Tile IR

Expands block-level Graph IR to thread-level Tile IR:
- program_id -> GroupIdx
- arange -> ThreadIdx
- load(mask) -> Cond(mask, BufLoad, other)
- store(mask) -> BufStore(guard=mask)
- reduce_sum/max -> SharedAlloc + Barrier + reduction loop
"""

from .graph_ir import KernelIR, IRNode, ForLoop, VType, Op
from . import tile_ir as T

_BINOP_MAP = {
    Op.ADD: "+", Op.SUB: "-", Op.MUL: "*", Op.DIV: "/", Op.MOD: "%",
    Op.LT: "<", Op.LE: "<=", Op.GT: ">", Op.GE: ">=", Op.EQ: "==", Op.NE: "!=",
}

_UNARYOP_MAP = {Op.EXP: "exp", Op.LOG: "log", Op.SQRT: "sqrt", Op.ABS: "abs"}


def lower(kernel_ir, grid, block_size):
    """Lower Graph IR to Tile IR

    Args:
        kernel_ir: KernelIR
        grid: tuple, threadgroup grid dimensions
        block_size: int, threads_per_threadgroup
    Returns:
        TileKernel
    """
    lowerer = _TileLowerer(kernel_ir, grid, block_size)
    return lowerer.lower()


class _TileLowerer:
    def __init__(self, kernel_ir, grid, block_size):
        self.kir = kernel_ir
        self.grid = grid
        self.block_size = block_size

        # Register -> Tile IR expression mapping
        self.reg_to_expr = {}

        self.buf_params = []      # [(name, is_const)]
        self.scalar_params = []   # [name]

        # Shared memory allocations
        self.shared = []
        self.reduction_counter = 0

        # Output statements
        self.body = []

    def lower(self):
        # Classify parameters
        for name, kind in self.kir.params:
            if kind == "pointer":
                self.buf_params.append((name, True))
            elif kind == "scalar":
                self.scalar_params.append(name)

        # Lower each IR statement
        self._lower_stmts(self.kir.body, self.body)

        # Detect output buffers (written by store)
        written_bufs = set()
        self._find_written_bufs(self.body, written_bufs)
        final_buf_params = []
        for name, _ in self.buf_params:
            is_const = name not in written_bufs
            final_buf_params.append((name, is_const))

        # Normalize grid
        g = self.grid
        if len(g) == 1:
            g = (g[0], 1, 1)
        elif len(g) == 2:
            g = (g[0], g[1], 1)

        return T.TileKernel(
            name=self.kir.name,
            buf_params=final_buf_params,
            scalar_params=self.scalar_params,
            grid=g,
            block=(self.block_size, 1, 1),
            shared=self.shared,
            body=self.body,
        )

    def _lower_stmts(self, stmts, output):
        for stmt in stmts:
            if isinstance(stmt, IRNode):
                self._lower_node(stmt, output)
            elif isinstance(stmt, ForLoop):
                self._lower_for(stmt, output)

    def _lower_node(self, node, output):
        op = node.op
        dst = node.dst

        if op == Op.PARAM_REF:
            name = node.attrs["param_name"]
            self.reg_to_expr[dst] = T.Var(name)
            return

        elif op == Op.PROGRAM_ID:
            axis = node.attrs.get("axis", 0)
            dim_map = {0: "x", 1: "y", 2: "z"}
            var_name = f"pid_{axis}"
            expr = T.GroupIdx(dim_map[axis])
            output.append(T.LetUint(var_name, expr))
            self.reg_to_expr[dst] = T.Var(var_name)
            return

        elif op == Op.ARANGE:
            var_name = "tid_val"
            output.append(T.LetUint(var_name, T.ThreadIdx("x")))
            self.reg_to_expr[dst] = T.Var(var_name)
            return

        elif op == Op.CONST_INT:
            self.reg_to_expr[dst] = T.IntLit(node.attrs["value"])
            return

        elif op == Op.CONST_FLOAT:
            self.reg_to_expr[dst] = T.FloatLit(node.attrs["value"])
            return

        elif op in _BINOP_MAP:
            left = self._get_expr(node.args[0])
            right = self._get_expr(node.args[1])
            self._emit_var(dst, T.BinOp(_BINOP_MAP[op], left, right), node.vtype, output)
            return

        elif op in (Op.AND, Op.OR):
            left = self._get_expr(node.args[0])
            right = self._get_expr(node.args[1])
            op_str = "&&" if op == Op.AND else "||"
            self._emit_var(dst, T.BinOp(op_str, left, right), node.vtype, output)
            return

        elif op == Op.NOT:
            self._emit_var(dst, T.UnaryOp("!", self._get_expr(node.args[0])), node.vtype, output)
            return

        elif op in _UNARYOP_MAP:
            self._emit_var(dst, T.UnaryOp(_UNARYOP_MAP[op], self._get_expr(node.args[0])), node.vtype, output)
            return

        elif op == Op.LOAD:
            self._lower_load(node, output)
            return

        elif op == Op.STORE:
            self._lower_store(node, output)
            return

        elif op == Op.ATOMIC_ADD:
            self._lower_atomic_add(node, output)
            return

        elif op in (Op.REDUCE_SUM, Op.REDUCE_MAX):
            self._lower_reduction(node, op, output)
            return

        elif op == Op.WHERE:
            cond = self._get_expr(node.args[0])
            then_val = self._get_expr(node.args[1])
            else_val = self._get_expr(node.args[2])
            self._emit_var(dst, T.Cond(cond, then_val, else_val), node.vtype, output)
            return

        raise ValueError(f"Unsupported Graph IR op: {op}")

    def _lower_load(self, node, output):
        """load(ptr, offset, mask?, other?) -> Cond(mask, BufLoad, FloatLit)"""
        buf_name = node.attrs["param_name"]
        offset_expr = self._get_expr(node.args[1])
        mask_reg = node.attrs.get("mask_reg")
        other_val = node.attrs.get("other_val", 0.0)

        load_expr = T.BufLoad(buf_name, offset_expr)

        if mask_reg is not None:
            mask_expr = self._get_expr(mask_reg)
            result = T.Cond(mask_expr, load_expr, T.FloatLit(other_val))
        else:
            result = load_expr

        var_name = f"r{node.dst}"
        output.append(T.LetFloat(var_name, result))
        self.reg_to_expr[node.dst] = T.Var(var_name)

    def _lower_store(self, node, output):
        """store(ptr, offset, value, mask?) -> BufStore(guard=mask)"""
        buf_name = node.attrs["param_name"]
        offset_expr = self._get_expr(node.args[1])
        value_expr = self._get_expr(node.args[2])
        mask_reg = node.attrs.get("mask_reg")

        guard = self._get_expr(mask_reg) if mask_reg is not None else None
        output.append(T.BufStore(buf_name, offset_expr, value_expr, guard))

    def _lower_atomic_add(self, node, output):
        """atomic_add -> BufStore with atomic (simplified)"""
        # TODO: proper atomic support in Tile IR
        buf_name = node.attrs["param_name"]
        offset_expr = self._get_expr(node.args[1])
        value_expr = self._get_expr(node.args[2])
        output.append(T.BufStore(buf_name, offset_expr, value_expr))

    def _lower_reduction(self, node, op, output):
        """reduce_sum/max -> shared memory + barrier + reduction loop"""
        self.reduction_counter += 1
        rid = self.reduction_counter
        shared_name = f"shared_{rid}"
        input_expr = self._get_expr(node.args[0])
        var_name = f"r{node.dst}"

        # Allocate shared memory
        self.shared.append(T.SharedAlloc(shared_name, self.block_size))

        # shared[tid] = input
        output.append(T.SharedStore(shared_name, T.Var("tid_val"), input_expr))
        output.append(T.Barrier())

        # Reduction loop
        reduce_op = "+" if op == Op.REDUCE_SUM else "max"
        loop_var = f"s_{rid}"

        left = T.SharedLoad(shared_name, T.Var("tid_val"))
        right = T.SharedLoad(shared_name, T.BinOp("+", T.Var("tid_val"), T.Var(loop_var)))
        update_expr = T.BinOp("+" if reduce_op == "+" else "max", left, right)

        loop_body = [
            T.IfStmt(
                cond=T.BinOp("<", T.Var("tid_val"), T.Var(loop_var)),
                body=[
                    T.SharedStore(shared_name, T.Var("tid_val"), update_expr),
                ]
            ),
            T.Barrier(),
        ]

        output.append(T.ForStmt(
            var=loop_var,
            start=T.BinOp("/", T.IntLit(self.block_size), T.IntLit(2)),
            end=T.IntLit(0),
            step=-1,  # Special: step -1 means s >>= 1 pattern
            body=loop_body,
        ))

        # Read result
        output.append(T.LetFloat(var_name, T.SharedLoad(shared_name, T.IntLit(0))))
        self.reg_to_expr[node.dst] = T.Var(var_name)

    def _lower_for(self, loop, output):
        """ForLoop -> ForStmt"""
        # Start value
        if loop.is_start_imm:
            start = T.IntLit(loop.start)
        else:
            start = self._get_expr(loop.start)

        # End value
        end = self._get_expr(loop.end)

        var_name = f"k_{loop.var_reg}"
        self.reg_to_expr[loop.var_reg] = T.Var(var_name)

        # Handle loop-carried variables:
        # Declare variables before the loop, update with Assign inside
        carried_var_names = {}  # loop_reg -> var_name
        for name, (init_reg, loop_reg) in loop.carried_vars.items():
            carried_name = f"acc_{name}"
            init_expr = self._get_expr(init_reg)
            # Declare accumulator variable before the loop
            vtype = self.kir.reg_types.get(init_reg, "scalar_float")
            if VType.is_float(vtype):
                output.append(T.LetFloat(carried_name, init_expr))
            else:
                output.append(T.LetAuto(carried_name, init_expr))
            # Map init_reg to this variable (loop body references init_reg via acc_name)
            self.reg_to_expr[init_reg] = T.Var(carried_name)
            carried_var_names[loop_reg] = carried_name

        # Lower loop body
        loop_body = []
        self._lower_stmts(loop.body, loop_body)

        # Replace LetFloat/LetAuto with Assign for carried variables in loop body
        reg_to_carried = {f"r{reg}": cname for reg, cname in carried_var_names.items()}
        for i, stmt in enumerate(loop_body):
            if isinstance(stmt, (T.LetFloat, T.LetAuto)) and stmt.name in reg_to_carried:
                carried_name = reg_to_carried[stmt.name]
                loop_body[i] = T.Assign(carried_name, stmt.expr)

        # Map loop_reg to carried variable name (for post-loop references)
        for loop_reg, carried_name in carried_var_names.items():
            self.reg_to_expr[loop_reg] = T.Var(carried_name)

        output.append(T.ForStmt(
            var=var_name,
            start=start,
            end=end,
            step=loop.step,
            body=loop_body,
        ))

    def _emit_var(self, dst, expr, vtype, output):
        """Generate variable declaration and register in reg_to_expr"""
        var_name = f"r{dst}"
        if VType.is_float(vtype):
            output.append(T.LetFloat(var_name, expr))
        else:
            output.append(T.LetAuto(var_name, expr))
        self.reg_to_expr[dst] = T.Var(var_name)

    def _get_expr(self, reg_or_val):
        """Get the Tile IR expression for a register"""
        if isinstance(reg_or_val, int) and reg_or_val in self.reg_to_expr:
            return self.reg_to_expr[reg_or_val]
        if isinstance(reg_or_val, int):
            return T.IntLit(reg_or_val)
        return T.FloatLit(float(reg_or_val))

    def _find_written_bufs(self, stmts, written):
        """Find buffers that are written to"""
        for stmt in stmts:
            if isinstance(stmt, T.BufStore):
                written.add(stmt.buf_name)
            elif isinstance(stmt, T.ForStmt):
                self._find_written_bufs(stmt.body, written)
            elif isinstance(stmt, T.IfStmt):
                self._find_written_bufs(stmt.body, written)

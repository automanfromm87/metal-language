"""Python AST -> Graph IR converter

Converts the AST of an @ml.kernel function into a Graph IR KernelIR.
"""

import ast
from .graph_ir import KernelIR, IRNode, ForLoop, VType, Op


# Python binary operators -> IR opcodes
_BINOP_MAP = {
    ast.Add: Op.ADD, ast.Sub: Op.SUB, ast.Mult: Op.MUL,
    ast.Div: Op.DIV, ast.Mod: Op.MOD, ast.FloorDiv: Op.DIV,
}

_CMPOP_MAP = {
    ast.Lt: Op.LT, ast.LtE: Op.LE, ast.Gt: Op.GT,
    ast.GtE: Op.GE, ast.Eq: Op.EQ, ast.NotEq: Op.NE,
}

_ML_MATH = {"exp": Op.EXP, "log": Op.LOG, "sqrt": Op.SQRT, "abs": Op.ABS}


def ast_to_graph(kernel_def, constexpr_vals):
    """Convert a KernelDef (with AST) to KernelIR"""
    builder = _GraphBuilder(kernel_def, constexpr_vals)
    builder.build()
    return builder.ir


class _GraphBuilder:
    def __init__(self, kernel_def, constexpr_vals):
        params = [(p.name, p.kind) for p in kernel_def.params]
        self.ir = KernelIR(
            name=kernel_def.name,
            params=params,
            body=[],
            constexpr_vals=constexpr_vals,
        )
        self.kernel_def = kernel_def
        self.all_constants = {**kernel_def.constants, **constexpr_vals}

        # Name -> register number mapping
        self.name_to_reg = {}

        # Pre-allocate registers for each parameter
        self.pointer_params = set()
        self.scalar_params = set()
        for p in kernel_def.params:
            if p.kind == "constexpr":
                continue
            if p.kind == "pointer":
                reg = self.ir.alloc_reg(VType.TILE_FLOAT)
                self.ir.body.append(IRNode(
                    reg, Op.PARAM_REF, [], VType.TILE_FLOAT,
                    {"param_name": p.name, "param_kind": "pointer"}
                ))
                self.pointer_params.add(p.name)
            else:
                reg = self.ir.alloc_reg(VType.SCALAR_UINT)
                self.ir.body.append(IRNode(
                    reg, Op.PARAM_REF, [], VType.SCALAR_UINT,
                    {"param_name": p.name, "param_kind": "scalar"}
                ))
                self.scalar_params.add(p.name)
            self.name_to_reg[p.name] = reg

        # Current write target (for ForLoop)
        self._current_body = self.ir.body

    def build(self):
        for stmt in self.kernel_def.body_ast:
            self._gen_stmt(stmt)

    # === Statements ===

    def _gen_stmt(self, node):
        if isinstance(node, ast.Assign):
            self._gen_assign(node)
        elif isinstance(node, ast.AugAssign):
            self._gen_aug_assign(node)
        elif isinstance(node, ast.Expr):
            self._gen_expr_stmt(node)
        elif isinstance(node, ast.For):
            self._gen_for(node)
        elif isinstance(node, ast.If):
            raise ValueError("if statements not supported, use ml.where() instead")
        else:
            raise ValueError(f"Unsupported statement: {type(node).__name__}")

    def _gen_assign(self, node):
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            raise ValueError("Assignment target must be a simple variable name")

        value_reg = self._gen_expr(node.value)
        self.name_to_reg[target.id] = value_reg

    def _gen_aug_assign(self, node):
        if not isinstance(node.target, ast.Name):
            raise ValueError("Augmented assignment target must be a simple variable name")

        name = node.target.id
        old_reg = self.name_to_reg[name]
        right_reg = self._gen_expr(node.value)
        op = _BINOP_MAP.get(type(node.op))
        if op is None:
            raise ValueError(f"Unsupported operator: {type(node.op).__name__}")

        old_type = self.ir.reg_types[old_reg]
        right_type = self.ir.reg_types[right_reg]
        result_type = VType.promote(old_type, right_type)
        new_reg = self.ir.emit(op, [old_reg, right_reg], result_type)
        self.name_to_reg[name] = new_reg

    def _gen_expr_stmt(self, node):
        """Expression statement (mainly ml.store)"""
        if isinstance(node.value, ast.Call) and self._is_ml_call(node.value):
            func_name = node.value.func.attr
            if func_name == "store":
                self._gen_ml_store(node.value)
                return
            if func_name == "atomic_add":
                self._gen_atomic_add(node.value)
                return
        self._gen_expr(node.value)

    def _gen_for(self, node):
        if not isinstance(node.iter, ast.Call):
            raise ValueError("for loops only support range()")

        func = node.iter.func
        if not (isinstance(func, ast.Name) and func.id == "range"):
            raise ValueError("for loops only support range()")

        var_name = node.target.id
        args = node.iter.args

        # Parse range arguments
        if len(args) == 1:
            start, end_val, step = 0, args[0], 1
        elif len(args) == 2:
            start, end_val, step = args[0], args[1], 1
        elif len(args) == 3:
            start, end_val, step = args[0], args[1], args[2]
        else:
            raise ValueError("range() accepts at most 3 arguments")

        # Parse start
        if isinstance(start, int):
            start_val = start
            is_start_imm = True
        elif isinstance(start, ast.Constant) and isinstance(start.value, int):
            start_val = start.value
            is_start_imm = True
        else:
            start_val = self._gen_expr(start)
            is_start_imm = False

        # Parse end
        end_reg = self._gen_expr(end_val)
        is_end_imm = False
        if self._current_body:
            end_node = self._current_body[-1]
            if isinstance(end_node, IRNode) and end_node.op == Op.CONST_INT:
                is_end_imm = True

        # Parse step
        if isinstance(step, int):
            step_val = step
        elif isinstance(step, ast.Constant):
            step_val = step.value
        elif isinstance(step, ast.Name) and step.id in self.all_constants:
            step_val = int(self.all_constants[step.id])
        else:
            step_val = 1

        # Create loop variable register
        var_reg = self.ir.alloc_reg(VType.SCALAR_UINT)
        self.name_to_reg[var_name] = var_reg

        # Build ForLoop
        loop = ForLoop(
            var_reg=var_reg,
            start=start_val,
            end=end_reg,
            step=step_val,
            body=[],
            is_start_imm=is_start_imm,
            is_end_imm=is_end_imm,
        )

        # Snapshot variable mapping before the loop
        pre_loop_regs = dict(self.name_to_reg)

        # Generate loop body
        old_body = self._current_body
        self._current_body = loop.body

        for stmt in node.body:
            self._gen_stmt(stmt)

        # Detect loop-carried dependencies
        carried_vars = {}
        for name, reg in self.name_to_reg.items():
            if name in pre_loop_regs and pre_loop_regs[name] != reg and name != var_name:
                carried_vars[name] = (pre_loop_regs[name], reg)

        loop.carried_vars = carried_vars

        self._current_body = old_body
        self._current_body.append(loop)

    # === Expressions ===

    def _gen_expr(self, node):
        """Generate an expression, return the result register number"""
        if isinstance(node, ast.BinOp):
            return self._gen_binop(node)
        elif isinstance(node, ast.UnaryOp):
            return self._gen_unaryop(node)
        elif isinstance(node, ast.Compare):
            return self._gen_compare(node)
        elif isinstance(node, ast.BoolOp):
            return self._gen_boolop(node)
        elif isinstance(node, ast.Name):
            return self._gen_name(node)
        elif isinstance(node, ast.Constant):
            return self._gen_constant(node)
        elif isinstance(node, ast.Call):
            return self._gen_call(node)
        elif isinstance(node, ast.IfExp):
            return self._gen_ifexp(node)
        elif isinstance(node, ast.Subscript):
            raise ValueError("Use ml.load() instead of direct subscript access")
        raise ValueError(f"Unsupported expression: {type(node).__name__}")

    def _gen_binop(self, node):
        left = self._gen_expr(node.left)
        right = self._gen_expr(node.right)
        op = _BINOP_MAP.get(type(node.op))
        if op is None:
            raise ValueError(f"Unsupported operator: {type(node.op).__name__}")
        result_type = VType.promote(
            self.ir.reg_types[left], self.ir.reg_types[right]
        )
        return self._emit_to_current(op, [left, right], result_type)

    def _gen_unaryop(self, node):
        operand = self._gen_expr(node.operand)
        if isinstance(node.op, ast.USub):
            # -x -> 0 - x
            zero = self._emit_to_current(
                Op.CONST_FLOAT, [], VType.SCALAR_FLOAT, {"value": 0.0}
            )
            return self._emit_to_current(
                Op.SUB, [zero, operand], self.ir.reg_types[operand]
            )
        elif isinstance(node.op, ast.Not):
            return self._emit_to_current(Op.NOT, [operand], self.ir.reg_types[operand])
        raise ValueError(f"Unsupported unary op: {type(node.op).__name__}")

    def _gen_compare(self, node):
        left = self._gen_expr(node.left)
        if len(node.ops) != 1:
            raise ValueError("Chained comparisons not supported, use 'and' instead")
        op = _CMPOP_MAP.get(type(node.ops[0]))
        if op is None:
            raise ValueError(f"Unsupported comparison: {type(node.ops[0]).__name__}")
        right = self._gen_expr(node.comparators[0])

        left_type = self.ir.reg_types[left]
        right_type = self.ir.reg_types[right]
        is_tile = VType.is_tile(left_type) or VType.is_tile(right_type)
        result_type = VType.TILE_BOOL if is_tile else VType.SCALAR_BOOL
        return self._emit_to_current(op, [left, right], result_type)

    def _gen_boolop(self, node):
        op = Op.AND if isinstance(node.op, ast.And) else Op.OR
        result = self._gen_expr(node.values[0])
        for val in node.values[1:]:
            right = self._gen_expr(val)
            promoted = VType.promote(self.ir.reg_types[result], self.ir.reg_types[right])
            result_type = VType.TILE_BOOL if VType.is_tile(promoted) else VType.SCALAR_BOOL
            result = self._emit_to_current(op, [result, right], result_type)
        return result

    def _gen_name(self, node):
        name = node.id
        if name in self.name_to_reg:
            return self.name_to_reg[name]
        if name in self.all_constants:
            val = self.all_constants[name]
            if isinstance(val, int):
                return self._emit_to_current(
                    Op.CONST_INT, [], VType.SCALAR_UINT, {"value": val}
                )
            return self._emit_to_current(
                Op.CONST_FLOAT, [], VType.SCALAR_FLOAT, {"value": val}
            )
        raise ValueError(f"Undefined variable: {name}")

    def _gen_constant(self, node):
        if isinstance(node.value, int):
            return self._emit_to_current(
                Op.CONST_INT, [], VType.SCALAR_UINT, {"value": node.value}
            )
        elif isinstance(node.value, float):
            return self._emit_to_current(
                Op.CONST_FLOAT, [], VType.SCALAR_FLOAT, {"value": node.value}
            )
        raise ValueError(f"Unsupported constant type: {type(node.value)}")

    def _gen_call(self, node):
        if self._is_ml_call(node):
            return self._gen_ml_call(node)

        func_name = None
        if isinstance(node.func, ast.Attribute):
            if isinstance(node.func.value, ast.Name) and node.func.value.id == "math":
                func_name = node.func.attr
        elif isinstance(node.func, ast.Name):
            func_name = node.func.id

        if func_name and func_name in _ML_MATH:
            return self._emit_math_unary(func_name, node.args[0])

        if func_name == "range":
            raise ValueError("range() can only be used in for loops")

        raise ValueError(f"Unsupported function call: {ast.dump(node.func)}")

    def _emit_math_unary(self, func_name, arg_node):
        operand = self._gen_expr(arg_node)
        op_type = self.ir.reg_types[operand]
        result_type = VType.TILE_FLOAT if VType.is_tile(op_type) else VType.SCALAR_FLOAT
        return self._emit_to_current(_ML_MATH[func_name], [operand], result_type)

    def _gen_ifexp(self, node):
        """Conditional expression: x if cond else y -> where(cond, x, y)"""
        cond = self._gen_expr(node.test)
        then_val = self._gen_expr(node.body)
        else_val = self._gen_expr(node.orelse)
        result_type = VType.promote(
            self.ir.reg_types[then_val], self.ir.reg_types[else_val]
        )
        return self._emit_to_current(
            Op.WHERE, [cond, then_val, else_val], result_type
        )

    # === ml.xxx calls ===

    def _gen_ml_call(self, node):
        func_name = node.func.attr

        if func_name == "program_id":
            axis = 0
            if node.args:
                axis = self._get_const_int(node.args[0])
            return self._emit_to_current(
                Op.PROGRAM_ID, [], VType.SCALAR_UINT, {"axis": axis}
            )

        elif func_name == "arange":
            start = self._get_const_int(node.args[0])
            end = self._get_const_int(node.args[1])
            return self._emit_to_current(
                Op.ARANGE, [], VType.TILE_UINT, {"start": start, "end": end}
            )

        elif func_name == "load":
            return self._gen_ml_load(node)

        elif func_name == "zeros":
            return self._emit_to_current(
                Op.CONST_FLOAT, [], VType.SCALAR_FLOAT, {"value": 0.0}
            )

        elif func_name in ("sum", "max"):
            input_reg = self._gen_expr(node.args[0])
            op = Op.REDUCE_SUM if func_name == "sum" else Op.REDUCE_MAX
            return self._emit_to_current(op, [input_reg], VType.SCALAR_FLOAT)

        elif func_name in _ML_MATH:
            return self._emit_math_unary(func_name, node.args[0])

        elif func_name == "where":
            cond = self._gen_expr(node.args[0])
            x = self._gen_expr(node.args[1])
            y = self._gen_expr(node.args[2])
            result_type = VType.promote(
                self.ir.reg_types[x], self.ir.reg_types[y]
            )
            return self._emit_to_current(Op.WHERE, [cond, x, y], result_type)

        elif func_name == "cdiv":
            a = self._gen_expr(node.args[0])
            b = self._gen_expr(node.args[1])
            # (a + b - 1) / b
            one = self._emit_to_current(
                Op.CONST_INT, [], VType.SCALAR_UINT, {"value": 1}
            )
            ab = self._emit_to_current(Op.ADD, [a, b], VType.SCALAR_UINT)
            ab1 = self._emit_to_current(Op.SUB, [ab, one], VType.SCALAR_UINT)
            return self._emit_to_current(Op.DIV, [ab1, b], VType.SCALAR_UINT)

        raise ValueError(f"Unsupported ml function: {func_name}")

    def _gen_ml_load(self, node):
        """ml.load(ptr_expr, mask=..., other=...)"""
        ptr_expr = node.args[0]

        mask_reg = None
        other_val = 0.0
        for kw in node.keywords:
            if kw.arg == "mask":
                mask_reg = self._gen_expr(kw.value)
            elif kw.arg == "other":
                if isinstance(kw.value, ast.Constant):
                    other_val = float(kw.value.value)
                elif isinstance(kw.value, ast.UnaryOp) and isinstance(kw.value.op, ast.USub):
                    other_val = -float(kw.value.operand.value)
        if len(node.args) > 1 and mask_reg is None:
            mask_reg = self._gen_expr(node.args[1])
        if len(node.args) > 2:
            other_val = float(node.args[2].value) if isinstance(node.args[2], ast.Constant) else other_val

        ptr_name, offset_reg = self._resolve_pointer_expr(ptr_expr)

        attrs = {"param_name": ptr_name, "other_val": other_val}
        if mask_reg is not None:
            attrs["mask_reg"] = mask_reg

        ptr_reg = self.name_to_reg[ptr_name]
        return self._emit_to_current(
            Op.LOAD, [ptr_reg, offset_reg], VType.TILE_FLOAT, attrs
        )

    def _gen_ml_store(self, node):
        """ml.store(ptr_expr, value, mask=...)"""
        ptr_expr = node.args[0]
        value_reg = self._gen_expr(node.args[1])

        mask_reg = None
        for kw in node.keywords:
            if kw.arg == "mask":
                mask_reg = self._gen_expr(kw.value)
        if len(node.args) > 2 and mask_reg is None:
            mask_reg = self._gen_expr(node.args[2])

        ptr_name, offset_reg = self._resolve_pointer_expr(ptr_expr)

        attrs = {"param_name": ptr_name}
        if mask_reg is not None:
            attrs["mask_reg"] = mask_reg

        ptr_reg = self.name_to_reg[ptr_name]
        store_node = IRNode(-1, Op.STORE, [ptr_reg, offset_reg, value_reg], "void", attrs)
        self._current_body.append(store_node)

    def _gen_atomic_add(self, node):
        ptr_expr = node.args[0]
        value_reg = self._gen_expr(node.args[1])
        ptr_name, offset_reg = self._resolve_pointer_expr(ptr_expr)
        ptr_reg = self.name_to_reg[ptr_name]
        atomic_node = IRNode(-1, Op.ATOMIC_ADD, [ptr_reg, offset_reg, value_reg], "void",
                              {"param_name": ptr_name})
        self._current_body.append(atomic_node)

    def _resolve_pointer_expr(self, node):
        """Resolve ptr + offset -> (ptr_name, offset_reg)"""
        if isinstance(node, ast.Name):
            if node.id in self.pointer_params:
                zero = self._emit_to_current(
                    Op.CONST_INT, [], VType.SCALAR_UINT, {"value": 0}
                )
                return node.id, zero

        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            if isinstance(node.left, ast.Name) and node.left.id in self.pointer_params:
                offset = self._gen_expr(node.right)
                return node.left.id, offset
            if isinstance(node.right, ast.Name) and node.right.id in self.pointer_params:
                offset = self._gen_expr(node.left)
                return node.right.id, offset
            # Recursive: (ptr + a) + b
            try:
                name, inner = self._resolve_pointer_expr(node.left)
                right = self._gen_expr(node.right)
                combined = self._emit_to_current(
                    Op.ADD, [inner, right],
                    VType.promote(self.ir.reg_types[inner], self.ir.reg_types[right])
                )
                return name, combined
            except ValueError:
                pass
            try:
                name, inner = self._resolve_pointer_expr(node.right)
                left = self._gen_expr(node.left)
                combined = self._emit_to_current(
                    Op.ADD, [left, inner],
                    VType.promote(self.ir.reg_types[left], self.ir.reg_types[inner])
                )
                return name, combined
            except ValueError:
                pass

        raise ValueError(f"Cannot resolve pointer expression: {ast.dump(node)}")

    # === Utilities ===

    def _emit_to_current(self, op, args, vtype, attrs=None):
        """Append an instruction to the current body (may be a loop body)"""
        reg = self.ir.alloc_reg(vtype)
        self._current_body.append(IRNode(reg, op, args, vtype, attrs or {}))
        return reg

    def _is_ml_call(self, node):
        return (isinstance(node, ast.Call) and
                isinstance(node.func, ast.Attribute) and
                isinstance(node.func.value, ast.Name) and
                node.func.value.id == "ml")

    def _get_const_int(self, node):
        if isinstance(node, ast.Constant) and isinstance(node.value, int):
            return node.value
        if isinstance(node, ast.Name) and node.id in self.all_constants:
            return int(self.all_constants[node.id])
        raise ValueError(f"Compile-time constant integer required: {ast.dump(node)}")

"""Graph IR optimization passes

Runs on KernelIR before lowering to Tile IR:
1. Constant Folding  — evaluate const ops at compile time
2. CSE              — eliminate duplicate computations
3. DCE              — remove unused instructions
"""

import operator
from .graph_ir import IRNode, ForLoop, Op


def optimize(kernel_ir):
    """Run all optimization passes."""
    constant_fold(kernel_ir)
    cse(kernel_ir)
    dce(kernel_ir)
    return kernel_ir


# ---------------------------------------------------------------------------
# Pass 1: Constant Folding
# ---------------------------------------------------------------------------

_INT_OPS = {
    Op.ADD: operator.add, Op.SUB: operator.sub,
    Op.MUL: operator.mul, Op.MOD: operator.mod,
}

_FLOAT_OPS = {
    Op.ADD: operator.add, Op.SUB: operator.sub,
    Op.MUL: operator.mul, Op.DIV: operator.truediv,
}


def constant_fold(kernel_ir):
    """Fold binary ops on constants into a single constant."""
    # reg -> (op, value) for constants
    const_vals = {}
    for stmt in kernel_ir.body:
        if isinstance(stmt, IRNode):
            if stmt.op == Op.CONST_INT:
                const_vals[stmt.dst] = ("int", stmt.attrs["value"])
            elif stmt.op == Op.CONST_FLOAT:
                const_vals[stmt.dst] = ("float", stmt.attrs["value"])

    _fold_body(kernel_ir.body, const_vals, kernel_ir)


def _fold_body(body, const_vals, kernel_ir):
    for i, stmt in enumerate(body):
        if isinstance(stmt, ForLoop):
            _fold_body(stmt.body, const_vals, kernel_ir)
            continue
        if not isinstance(stmt, IRNode):
            continue

        if stmt.op == Op.CONST_INT:
            const_vals[stmt.dst] = ("int", stmt.attrs["value"])
        elif stmt.op == Op.CONST_FLOAT:
            const_vals[stmt.dst] = ("float", stmt.attrs["value"])
        elif stmt.op in _INT_OPS and len(stmt.args) == 2:
            a, b = stmt.args
            if a in const_vals and b in const_vals:
                at, av = const_vals[a]
                bt, bv = const_vals[b]
                if at == "int" and bt == "int" and stmt.op in _INT_OPS:
                    result = _INT_OPS[stmt.op](av, bv)
                    stmt.op = Op.CONST_INT
                    stmt.args = []
                    stmt.attrs = {"value": int(result)}
                    stmt.vtype = "scalar_uint"
                    const_vals[stmt.dst] = ("int", int(result))
                elif stmt.op in _FLOAT_OPS:
                    result = _FLOAT_OPS[stmt.op](float(av), float(bv))
                    stmt.op = Op.CONST_FLOAT
                    stmt.args = []
                    stmt.attrs = {"value": result}
                    stmt.vtype = "scalar_float"
                    const_vals[stmt.dst] = ("float", result)
        elif stmt.op in _FLOAT_OPS and stmt.op not in _INT_OPS:
            # DIV is float-only
            if len(stmt.args) == 2:
                a, b = stmt.args
                if a in const_vals and b in const_vals:
                    av = float(const_vals[a][1])
                    bv = float(const_vals[b][1])
                    result = _FLOAT_OPS[stmt.op](av, bv)
                    stmt.op = Op.CONST_FLOAT
                    stmt.args = []
                    stmt.attrs = {"value": result}
                    stmt.vtype = "scalar_float"
                    const_vals[stmt.dst] = ("float", result)


# ---------------------------------------------------------------------------
# Pass 2: Common Subexpression Elimination
# ---------------------------------------------------------------------------

# Ops that are pure (no side effects, safe to CSE)
_PURE_OPS = {
    Op.PROGRAM_ID, Op.ARANGE, Op.CONST_INT, Op.CONST_FLOAT, Op.PARAM_REF,
    Op.ADD, Op.SUB, Op.MUL, Op.DIV, Op.MOD,
    Op.LT, Op.LE, Op.GT, Op.GE, Op.EQ, Op.NE,
    Op.AND, Op.OR, Op.NOT,
    Op.EXP, Op.LOG, Op.SQRT, Op.ABS,
    Op.WHERE, Op.LOAD,
}


def _node_key(node):
    """Build a hashable key for CSE comparison."""
    # For attrs, only include semantically relevant keys
    relevant = {}
    for k, v in node.attrs.items():
        if isinstance(v, (int, float, str, bool)):
            relevant[k] = v
    return (node.op, tuple(node.args), tuple(sorted(relevant.items())))


def cse(kernel_ir):
    """Eliminate common subexpressions."""
    remap = {}
    _cse_body(kernel_ir.body, remap)
    # Rewrite all register references using remap
    if remap:
        _rewrite_refs(kernel_ir.body, remap)


def _cse_body(body, remap):
    seen = {}  # key -> dst_reg
    to_remove = set()

    for i, stmt in enumerate(body):
        if isinstance(stmt, ForLoop):
            # Rewrite ForLoop's own references first
            _cse_body(stmt.body, remap)
            continue
        if not isinstance(stmt, IRNode):
            continue

        # Apply existing remaps to this node's args before computing key
        stmt.args = [remap.get(a, a) if isinstance(a, int) else a for a in stmt.args]
        if "mask_reg" in stmt.attrs and isinstance(stmt.attrs["mask_reg"], int):
            stmt.attrs["mask_reg"] = remap.get(stmt.attrs["mask_reg"], stmt.attrs["mask_reg"])

        if stmt.op not in _PURE_OPS or stmt.dst < 0:
            continue

        key = _node_key(stmt)
        if key in seen:
            # Duplicate found — remap this register to the original
            remap[stmt.dst] = seen[key]
            to_remove.add(i)
        else:
            seen[key] = stmt.dst

    # Remove duplicates in reverse order
    for i in sorted(to_remove, reverse=True):
        body.pop(i)


def _rewrite_refs(body, remap):
    """Rewrite all register references using the remap dict."""
    for stmt in body:
        if isinstance(stmt, IRNode):
            stmt.args = [remap.get(a, a) if isinstance(a, int) else a for a in stmt.args]
            if "mask_reg" in stmt.attrs and isinstance(stmt.attrs["mask_reg"], int):
                stmt.attrs["mask_reg"] = remap.get(stmt.attrs["mask_reg"], stmt.attrs["mask_reg"])
        elif isinstance(stmt, ForLoop):
            if isinstance(stmt.end, int):
                stmt.end = remap.get(stmt.end, stmt.end)
            if not stmt.is_start_imm and isinstance(stmt.start, int):
                stmt.start = remap.get(stmt.start, stmt.start)
            # Rewrite carried_vars
            new_carried = {}
            for name, (init_reg, loop_reg) in stmt.carried_vars.items():
                new_carried[name] = (remap.get(init_reg, init_reg), remap.get(loop_reg, loop_reg))
            stmt.carried_vars = new_carried
            _rewrite_refs(stmt.body, remap)


# ---------------------------------------------------------------------------
# Pass 3: Dead Code Elimination
# ---------------------------------------------------------------------------

def dce(kernel_ir):
    """Remove instructions whose results are never used."""
    changed = True
    while changed:
        used = _collect_used(kernel_ir.body)
        changed = _eliminate_dead(kernel_ir.body, used)


def _collect_used(body):
    """Collect all register numbers that are referenced."""
    used = set()
    for stmt in body:
        if isinstance(stmt, IRNode):
            for a in stmt.args:
                if isinstance(a, int) and a >= 0:
                    used.add(a)
            if "mask_reg" in stmt.attrs and isinstance(stmt.attrs["mask_reg"], int):
                used.add(stmt.attrs["mask_reg"])
        elif isinstance(stmt, ForLoop):
            if isinstance(stmt.end, int):
                used.add(stmt.end)
            if not stmt.is_start_imm and isinstance(stmt.start, int):
                used.add(stmt.start)
            for _name, (init_reg, loop_reg) in stmt.carried_vars.items():
                if isinstance(init_reg, int):
                    used.add(init_reg)
                if isinstance(loop_reg, int):
                    used.add(loop_reg)
            used |= _collect_used(stmt.body)
    return used


# Side-effect ops that should never be eliminated
_SIDE_EFFECT_OPS = {Op.STORE, Op.ATOMIC_ADD}


def _eliminate_dead(body, used):
    """Remove dead nodes. Returns True if anything changed."""
    to_remove = []
    for i, stmt in enumerate(body):
        if isinstance(stmt, IRNode):
            if stmt.dst >= 0 and stmt.dst not in used and stmt.op not in _SIDE_EFFECT_OPS:
                to_remove.append(i)
        elif isinstance(stmt, ForLoop):
            _eliminate_dead(stmt.body, used)

    for i in reversed(to_remove):
        body.pop(i)

    return len(to_remove) > 0

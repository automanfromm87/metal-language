"""Graph IR optimization passes

Runs on KernelIR before lowering to Tile IR:
1. Constant Folding          — evaluate const ops at compile time
2. Algebraic Simplification  — identity/annihilation/cancellation rules
3. Strength Reduction        — replace expensive ops with cheaper equivalents
4. CSE                       — eliminate duplicate computations
5. LICM                      — hoist loop-invariant code out of loops
6. Loop Unrolling            — unroll small known-trip-count loops
7. DCE                       — remove unused instructions
"""

import copy
import operator
from .graph_ir import IRNode, ForLoop, Op


def optimize(kernel_ir):
    """Run all optimization passes."""
    constant_fold(kernel_ir)
    algebraic_simplify(kernel_ir)
    strength_reduce(kernel_ir)
    cse(kernel_ir)
    licm(kernel_ir)
    loop_unroll(kernel_ir)
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
            # Carried var init_regs are NOT constant inside the loop
            loop_const_vals = dict(const_vals)
            for _name, (init_reg, _loop_reg) in stmt.carried_vars.items():
                loop_const_vals.pop(init_reg, None)
            _fold_body(stmt.body, loop_const_vals, kernel_ir)
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


# ---------------------------------------------------------------------------
# Pass 4: Algebraic Simplification
# ---------------------------------------------------------------------------

def algebraic_simplify(kernel_ir):
    """Simplify identity, annihilation, and cancellation patterns."""
    const_vals = {}
    for stmt in kernel_ir.body:
        if isinstance(stmt, IRNode):
            if stmt.op == Op.CONST_INT:
                const_vals[stmt.dst] = ("int", stmt.attrs["value"])
            elif stmt.op == Op.CONST_FLOAT:
                const_vals[stmt.dst] = ("float", stmt.attrs["value"])

    remap = {}
    _simplify_body(kernel_ir.body, const_vals, remap)
    if remap:
        _rewrite_refs(kernel_ir.body, remap)


def _is_zero(const_vals, reg):
    if reg in const_vals:
        return const_vals[reg][1] == 0 or const_vals[reg][1] == 0.0
    return False


def _is_one(const_vals, reg):
    if reg in const_vals:
        return const_vals[reg][1] == 1 or const_vals[reg][1] == 1.0
    return False


def _simplify_body(body, const_vals, remap):
    for stmt in body:
        if isinstance(stmt, ForLoop):
            # Carried var init_regs are NOT constant inside the loop
            loop_const_vals = dict(const_vals)
            for _name, (init_reg, _loop_reg) in stmt.carried_vars.items():
                loop_const_vals.pop(init_reg, None)
            _simplify_body(stmt.body, loop_const_vals, remap)
            continue
        if not isinstance(stmt, IRNode):
            continue

        if stmt.op == Op.CONST_INT:
            const_vals[stmt.dst] = ("int", stmt.attrs["value"])
        elif stmt.op == Op.CONST_FLOAT:
            const_vals[stmt.dst] = ("float", stmt.attrs["value"])
        elif stmt.op in (Op.ADD, Op.SUB, Op.MUL, Op.DIV) and len(stmt.args) == 2:
            a, b = stmt.args
            # Apply existing remaps
            a = remap.get(a, a) if isinstance(a, int) else a
            b = remap.get(b, b) if isinstance(b, int) else b
            stmt.args = [a, b]

            if stmt.op == Op.ADD:
                # add(x, 0) -> x, add(0, x) -> x
                if _is_zero(const_vals, b):
                    remap[stmt.dst] = a
                elif _is_zero(const_vals, a):
                    remap[stmt.dst] = b
            elif stmt.op == Op.SUB:
                # sub(x, 0) -> x
                if _is_zero(const_vals, b):
                    remap[stmt.dst] = a
                # sub(x, x) -> 0
                elif a == b:
                    stmt.op = Op.CONST_INT
                    stmt.args = []
                    stmt.attrs = {"value": 0}
                    stmt.vtype = "scalar_uint"
                    const_vals[stmt.dst] = ("int", 0)
            elif stmt.op == Op.MUL:
                # mul(x, 1) -> x, mul(1, x) -> x
                if _is_one(const_vals, b):
                    remap[stmt.dst] = a
                elif _is_one(const_vals, a):
                    remap[stmt.dst] = b
                # mul(x, 0) -> 0, mul(0, x) -> 0
                elif _is_zero(const_vals, b):
                    remap[stmt.dst] = b
                elif _is_zero(const_vals, a):
                    remap[stmt.dst] = a
            elif stmt.op == Op.DIV:
                # div(x, 1) -> x
                if _is_one(const_vals, b):
                    remap[stmt.dst] = a


# ---------------------------------------------------------------------------
# Pass 5: Strength Reduction
# ---------------------------------------------------------------------------

def strength_reduce(kernel_ir):
    """Replace expensive ops with cheaper equivalents."""
    const_vals = {}
    for stmt in kernel_ir.body:
        if isinstance(stmt, IRNode):
            if stmt.op == Op.CONST_INT:
                const_vals[stmt.dst] = ("int", stmt.attrs["value"])
            elif stmt.op == Op.CONST_FLOAT:
                const_vals[stmt.dst] = ("float", stmt.attrs["value"])

    _strength_reduce_body(kernel_ir.body, const_vals, kernel_ir)


def _strength_reduce_body(body, const_vals, kernel_ir):
    for i, stmt in enumerate(body):
        if isinstance(stmt, ForLoop):
            loop_const_vals = dict(const_vals)
            for _name, (init_reg, _loop_reg) in stmt.carried_vars.items():
                loop_const_vals.pop(init_reg, None)
            _strength_reduce_body(stmt.body, loop_const_vals, kernel_ir)
            continue
        if not isinstance(stmt, IRNode):
            continue

        if stmt.op == Op.CONST_INT:
            const_vals[stmt.dst] = ("int", stmt.attrs["value"])
        elif stmt.op == Op.CONST_FLOAT:
            const_vals[stmt.dst] = ("float", stmt.attrs["value"])

        if len(stmt.args) != 2:
            continue
        a, b = stmt.args

        # div(x, const_float) -> mul(x, 1/const_float)
        if stmt.op == Op.DIV and b in const_vals:
            bt, bv = const_vals[b]
            if bt == "float" and bv != 0.0:
                recip_reg = kernel_ir.alloc_reg("scalar_float")
                recip_node = IRNode(recip_reg, Op.CONST_FLOAT, [], "scalar_float",
                                    {"value": 1.0 / bv})
                body.insert(i, recip_node)
                const_vals[recip_reg] = ("float", 1.0 / bv)
                stmt.op = Op.MUL
                stmt.args = [a, recip_reg]

        # mul(x, 2) -> add(x, x)
        elif stmt.op == Op.MUL:
            if b in const_vals and const_vals[b][1] == 2:
                stmt.op = Op.ADD
                stmt.args = [a, a]
            elif a in const_vals and const_vals[a][1] == 2:
                stmt.op = Op.ADD
                stmt.args = [b, b]


# ---------------------------------------------------------------------------
# Pass 6: Loop-Invariant Code Motion (LICM)
# ---------------------------------------------------------------------------

def licm(kernel_ir):
    """Move loop-invariant computations out of ForLoop bodies."""
    _licm_body(kernel_ir.body)


def _licm_body(body):
    """Process all ForLoops in a body list (recursive)."""
    for i, stmt in enumerate(body):
        if isinstance(stmt, ForLoop):
            # First recurse into nested loops
            _licm_body(stmt.body)
            # Then hoist invariant code from this loop
            _hoist_from_loop(body, i, stmt)


def _hoist_from_loop(parent_body, loop_idx, loop):
    """Hoist invariant nodes from a ForLoop to before it in parent_body."""
    # Registers that are "carried" (updated each iteration) - never hoist these
    carried_loop_regs = set()
    for _name, (init_reg, loop_reg) in loop.carried_vars.items():
        carried_loop_regs.add(loop_reg)

    changed = True
    while changed:
        changed = False

        # Collect all registers defined inside the loop
        inner_defined = {loop.var_reg}
        for stmt in loop.body:
            if isinstance(stmt, IRNode) and stmt.dst >= 0:
                inner_defined.add(stmt.dst)
            elif isinstance(stmt, ForLoop):
                inner_defined.add(stmt.var_reg)
                _collect_defined(stmt.body, inner_defined)

        to_hoist = []
        for idx, stmt in enumerate(loop.body):
            if not isinstance(stmt, IRNode):
                continue
            if stmt.dst < 0:  # side-effect ops
                continue
            if stmt.op in _SIDE_EFFECT_OPS:
                continue
            if stmt.op in (Op.REDUCE_SUM, Op.REDUCE_MAX):
                continue
            if stmt.dst in carried_loop_regs:
                continue

            # Check if all args are defined outside the loop
            all_outside = True
            for arg in stmt.args:
                if isinstance(arg, int) and arg in inner_defined:
                    all_outside = False
                    break
            if all_outside and "mask_reg" in stmt.attrs:
                mr = stmt.attrs["mask_reg"]
                if isinstance(mr, int) and mr in inner_defined:
                    all_outside = False

            if all_outside:
                to_hoist.append(idx)

        # Hoist in reverse order to maintain indices
        hoisted = 0
        for idx in sorted(to_hoist):
            actual_idx = idx - hoisted
            node = loop.body.pop(actual_idx)
            parent_body.insert(loop_idx + hoisted, node)
            hoisted += 1
            changed = True


def _collect_defined(body, defined):
    """Collect all dst registers defined in a body (recursive)."""
    for stmt in body:
        if isinstance(stmt, IRNode) and stmt.dst >= 0:
            defined.add(stmt.dst)
        elif isinstance(stmt, ForLoop):
            defined.add(stmt.var_reg)
            _collect_defined(stmt.body, defined)


# ---------------------------------------------------------------------------
# Pass 7: Loop Unrolling
# ---------------------------------------------------------------------------

_UNROLL_THRESHOLD = 8


def loop_unroll(kernel_ir):
    """Fully unroll ForLoops with small known trip counts."""
    _unroll_body(kernel_ir.body, kernel_ir)


def _unroll_body(body, kernel_ir):
    """Process body list, unrolling eligible ForLoops in place."""
    i = 0
    while i < len(body):
        stmt = body[i]
        if isinstance(stmt, ForLoop):
            # First recurse into the loop body
            _unroll_body(stmt.body, kernel_ir)
            # Then try to unroll this loop
            unrolled = _try_unroll(stmt, kernel_ir, body)
            if unrolled is not None:
                body.pop(i)
                for j, node in enumerate(unrolled):
                    body.insert(i + j, node)
                i += len(unrolled)
                continue
        i += 1


def _try_unroll(loop, kernel_ir, parent_body):
    """Try to unroll a ForLoop. Returns list of replacement nodes, or None."""
    if not loop.is_start_imm:
        return None
    if not loop.is_end_imm:
        return None

    # Find the end value from the const_int node
    end_val = _find_const_value(loop.end, parent_body)
    if end_val is None:
        # Also search kernel_ir.body
        end_val = _find_const_value(loop.end, kernel_ir.body)
    if end_val is None:
        return None

    start_val = loop.start
    step = loop.step
    if step <= 0:
        return None

    trip_count = (end_val - start_val + step - 1) // step
    if trip_count <= 0 or trip_count > _UNROLL_THRESHOLD:
        return None

    result = []

    # For each iteration, clone the body with fresh registers
    prev_carried = {}  # name -> register from previous iteration
    for name, (init_reg, _loop_reg) in loop.carried_vars.items():
        prev_carried[name] = init_reg

    for it in range(trip_count):
        iter_val = start_val + it * step
        remap = {}

        # Map loop variable to a constant
        iter_const_reg = kernel_ir.alloc_reg("scalar_uint")
        result.append(IRNode(iter_const_reg, Op.CONST_INT, [], "scalar_uint",
                             {"value": iter_val}))
        remap[loop.var_reg] = iter_const_reg

        # Map carried var init_regs to previous iteration's output
        for name, (init_reg, _loop_reg) in loop.carried_vars.items():
            if it == 0:
                # First iteration uses the original init_reg
                pass
            else:
                remap[init_reg] = prev_carried[name]

        # Clone each node in the loop body
        for stmt in loop.body:
            if isinstance(stmt, ForLoop):
                # Don't unroll nested loops here, just clone
                cloned = copy.deepcopy(stmt)
                _remap_forloop(cloned, remap)
                result.append(cloned)
            elif isinstance(stmt, IRNode):
                cloned = IRNode(
                    dst=stmt.dst,
                    op=stmt.op,
                    args=[remap.get(a, a) if isinstance(a, int) else a for a in stmt.args],
                    vtype=stmt.vtype,
                    attrs=dict(stmt.attrs),
                )
                # Remap mask_reg
                if "mask_reg" in cloned.attrs and isinstance(cloned.attrs["mask_reg"], int):
                    cloned.attrs["mask_reg"] = remap.get(cloned.attrs["mask_reg"],
                                                         cloned.attrs["mask_reg"])

                # Allocate fresh dst register (except void ops)
                if cloned.dst >= 0:
                    new_dst = kernel_ir.alloc_reg(cloned.vtype)
                    remap[cloned.dst] = new_dst
                    cloned.dst = new_dst

                result.append(cloned)

        # Update carried vars for next iteration
        for name, (_init_reg, loop_reg) in loop.carried_vars.items():
            prev_carried[name] = remap.get(loop_reg, loop_reg)

    return result


def _find_const_value(reg, body):
    """Find the const_int value for a register in a body list."""
    for stmt in body:
        if isinstance(stmt, IRNode) and stmt.dst == reg and stmt.op == Op.CONST_INT:
            return stmt.attrs["value"]
    return None


def _remap_forloop(loop, remap):
    """Apply register remap to a ForLoop and its body."""
    if not loop.is_start_imm and isinstance(loop.start, int):
        loop.start = remap.get(loop.start, loop.start)
    if isinstance(loop.end, int):
        loop.end = remap.get(loop.end, loop.end)
    new_carried = {}
    for name, (init_reg, loop_reg) in loop.carried_vars.items():
        new_carried[name] = (remap.get(init_reg, init_reg), remap.get(loop_reg, loop_reg))
    loop.carried_vars = new_carried
    for stmt in loop.body:
        if isinstance(stmt, IRNode):
            stmt.args = [remap.get(a, a) if isinstance(a, int) else a for a in stmt.args]
            if "mask_reg" in stmt.attrs and isinstance(stmt.attrs["mask_reg"], int):
                stmt.attrs["mask_reg"] = remap.get(stmt.attrs["mask_reg"], stmt.attrs["mask_reg"])
        elif isinstance(stmt, ForLoop):
            _remap_forloop(stmt, remap)

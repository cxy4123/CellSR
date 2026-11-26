from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable
import copy
import math
import random

import numpy as np
import sympy as sp

ArrayMap = dict[str, np.ndarray]


def _finite(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return np.nan_to_num(values, nan=0.0, posinf=1e6, neginf=-1e6)


def _safe_div(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return _finite(a / np.where(np.abs(b) < 1e-6, np.where(b < 0, -1e-6, 1e-6), b))


def _safe_exp(a: np.ndarray) -> np.ndarray:
    return _finite(np.exp(np.clip(a, -20.0, 20.0)))


def _safe_log_abs(a: np.ndarray) -> np.ndarray:
    return _finite(np.log(np.abs(a) + 1e-6))


def _safe_sqrt_abs(a: np.ndarray) -> np.ndarray:
    return _finite(np.sqrt(np.abs(a)))


@dataclass(frozen=True)
class Operator:
    name: str
    arity: int
    numpy_func: Callable[..., np.ndarray]
    sympy_func: Callable[..., sp.Expr]
    complexity_cost: int = 1


OPERATORS: dict[str, Operator] = {
    "add": Operator("add", 2, lambda a, b: _finite(a + b), lambda a, b: a + b),
    "sub": Operator("sub", 2, lambda a, b: _finite(a - b), lambda a, b: a - b),
    "mul": Operator("mul", 2, lambda a, b: _finite(a * b), lambda a, b: a * b),
    "div": Operator("div", 2, _safe_div, lambda a, b: a / b, 2),
    "neg": Operator("neg", 1, lambda a: _finite(-a), lambda a: -a),
    "square": Operator("square", 1, lambda a: _finite(a * a), lambda a: a**2),
    "cube": Operator("cube", 1, lambda a: _finite(a * a * a), lambda a: a**3, 2),
    "sin": Operator("sin", 1, lambda a: _finite(np.sin(a)), sp.sin, 2),
    "cos": Operator("cos", 1, lambda a: _finite(np.cos(a)), sp.cos, 2),
    "exp": Operator("exp", 1, _safe_exp, sp.exp, 3),
    "log_abs": Operator("log_abs", 1, _safe_log_abs, lambda a: sp.log(sp.Abs(a) + sp.Float(1e-6)), 3),
    "sqrt_abs": Operator("sqrt_abs", 1, _safe_sqrt_abs, lambda a: sp.sqrt(sp.Abs(a)), 2),
    "abs": Operator("abs", 1, lambda a: _finite(np.abs(a)), sp.Abs, 2),
}


@dataclass
class ExpressionNode:
    kind: str
    value: str | float
    children: list[ExpressionNode] = field(default_factory=list)

    @staticmethod
    def variable(name: str) -> ExpressionNode:
        return ExpressionNode("var", name)

    @staticmethod
    def constant(value: float) -> ExpressionNode:
        return ExpressionNode("const", float(value))

    @staticmethod
    def op(name: str, children: Iterable[ExpressionNode]) -> ExpressionNode:
        return ExpressionNode("op", name, list(children))

    def copy(self) -> ExpressionNode:
        return copy.deepcopy(self)

    def evaluate(self, data: ArrayMap) -> np.ndarray:
        if self.kind == "var":
            return np.asarray(data[str(self.value)], dtype=float)
        if self.kind == "const":
            size = len(next(iter(data.values()))) if data else 1
            return np.full(size, float(self.value), dtype=float)
        operator = OPERATORS[str(self.value)]
        args = [child.evaluate(data) for child in self.children]
        return _finite(operator.numpy_func(*args))

    def to_sympy(self) -> sp.Expr:
        if self.kind == "var":
            return sp.Symbol(str(self.value))
        if self.kind == "const":
            value = float(self.value)
            if math.isfinite(value):
                return sp.Float(round(value, 8))
            return sp.Float(0.0)
        operator = OPERATORS[str(self.value)]
        return operator.sympy_func(*[child.to_sympy() for child in self.children])

    def complexity(self) -> int:
        if self.kind in {"var", "const"}:
            return 1
        operator = OPERATORS[str(self.value)]
        return operator.complexity_cost + sum(child.complexity() for child in self.children)

    def variables(self) -> set[str]:
        if self.kind == "var":
            return {str(self.value)}
        if self.kind == "const":
            return set()
        result: set[str] = set()
        for child in self.children:
            result.update(child.variables())
        return result

    def nodes(self) -> list[ExpressionNode]:
        result = [self]
        for child in self.children:
            result.extend(child.nodes())
        return result

    def simplified(self) -> sp.Expr:
        try:
            return sp.simplify(self.to_sympy())
        except Exception:
            return self.to_sympy()

    def raw_str(self) -> str:
        """Fast structural string without SymPy simplification.
        Used inside the search loop where sympy.simplify is too expensive.
        """
        if self.kind == "var":
            return str(self.value)
        if self.kind == "const":
            return f"{float(self.value):.6g}"
        args = ",".join(c.raw_str() for c in self.children)
        return f"{self.value}({args})"

    def __str__(self) -> str:
        return str(self.simplified())

    def extract_constants(self) -> list[float]:
        """Extract all numeric constants in depth-first order."""
        constants = []
        for node in self.nodes():
            if node.kind == "const":
                constants.append(float(node.value))
        return constants

    def topology_hash(self) -> str:
        """Canonical form with constants replaced by placeholder 'C'.
        Two expressions with same topology_hash have the same functional structure
        but may differ in numeric constants.
        """
        if self.kind == "var":
            return str(self.value)
        if self.kind == "const":
            return "C"
        args = ",".join(c.topology_hash() for c in self.children)
        return f"{self.value}({args})"

    def set_constants(self, values: list[float]) -> None:
        """Set constants in depth-first order from a list of values."""
        idx = [0]
        def _set(node: ExpressionNode):
            if node.kind == "const":
                if idx[0] < len(values):
                    node.value = values[idx[0]]
                    idx[0] += 1
            else:
                for child in node.children:
                    _set(child)
        _set(self)


@dataclass
class ExpressionGenome:
    root: ExpressionNode
    lineage: tuple[str, ...] = ()

    def evaluate(self, data: ArrayMap) -> np.ndarray:
        return self.root.evaluate(data)

    def to_sympy(self) -> sp.Expr:
        return self.root.simplified()

    def raw_str(self) -> str:
        """Fast structural string, no SymPy. Use inside search loops."""
        return self.root.raw_str()

    def complexity(self) -> int:
        return self.root.complexity()

    def copy(self) -> ExpressionGenome:
        return ExpressionGenome(self.root.copy(), self.lineage)

    def __str__(self) -> str:
        return str(self.to_sympy())


def random_terminal(variables: list[str], constant_range: tuple[float, float], rng: random.Random) -> ExpressionNode:
    if rng.random() < 0.75 and variables:
        return ExpressionNode.variable(rng.choice(variables))
    return ExpressionNode.constant(rng.uniform(*constant_range))


def random_expression(
    variables: list[str],
    operators: list[str],
    max_depth: int,
    constant_range: tuple[float, float],
    rng: random.Random,
) -> ExpressionNode:
    if max_depth <= 0 or rng.random() < 0.25:
        return random_terminal(variables, constant_range, rng)
    operator = OPERATORS[rng.choice(operators)]
    return ExpressionNode.op(
        operator.name,
        [random_expression(variables, operators, max_depth - 1, constant_range, rng) for _ in range(operator.arity)],
    )


def replace_random_subtree(
    root: ExpressionNode,
    replacement: ExpressionNode,
    rng: random.Random,
) -> ExpressionNode:
    root = root.copy()
    paths: list[tuple[int, ...]] = []

    def visit(node: ExpressionNode, path: tuple[int, ...]) -> None:
        paths.append(path)
        for index, child in enumerate(node.children):
            visit(child, path + (index,))

    visit(root, ())
    path = rng.choice(paths)
    if not path:
        return replacement.copy()
    parent = root
    for index in path[:-1]:
        parent = parent.children[index]
    parent.children[path[-1]] = replacement.copy()
    return root


def mutate_expression(
    root: ExpressionNode,
    variables: list[str],
    operators: list[str],
    max_depth: int,
    constant_range: tuple[float, float],
    rng: random.Random,
) -> ExpressionNode:
    if rng.random() < 0.2:
        return random_expression(variables, operators, max_depth, constant_range, rng)
    replacement = random_expression(variables, operators, max(1, max_depth // 2), constant_range, rng)
    return replace_random_subtree(root, replacement, rng)


def crossover_expression(left: ExpressionNode, right: ExpressionNode, rng: random.Random) -> ExpressionNode:
    right_nodes = right.nodes()
    replacement = rng.choice(right_nodes)
    return replace_random_subtree(left, replacement, rng)


# ---------------------------------------------------------------------------
# Concise-mode operators
# ---------------------------------------------------------------------------

ArrayMap = dict[str, np.ndarray]


def tree_depth(node: ExpressionNode) -> int:
    if not node.children:
        return 0
    return 1 + max(tree_depth(c) for c in node.children)


def shrink_mutation(
    root: ExpressionNode,
    x: ArrayMap,
    y: np.ndarray,
    rng: random.Random,
) -> ExpressionNode:
    """Replace a random subtree with a fitted constant (mean of subtree output)."""
    root = root.copy()
    nodes_with_paths: list[tuple[ExpressionNode, tuple[int, ...]]] = []

    def _collect(node: ExpressionNode, path: tuple[int, ...]) -> None:
        nodes_with_paths.append((node, path))
        for i, child in enumerate(node.children):
            _collect(child, path + (i,))

    _collect(root, ())
    # Only shrink non-root, non-leaf nodes
    candidates = [(n, p) for n, p in nodes_with_paths if p and n.children]
    if not candidates:
        candidates = [(n, p) for n, p in nodes_with_paths if p]
    if not candidates:
        return root

    _, path = rng.choice(candidates)
    # Navigate to parent and replace child with a constant
    parent = root
    for idx in path[:-1]:
        parent = parent.children[idx]
    target = parent.children[path[-1]]
    try:
        val = target.evaluate(x)
        c = float(np.nanmean(val))
        if not np.isfinite(c):
            c = 0.0
    except Exception:
        c = 0.0
    parent.children[path[-1]] = ExpressionNode.constant(c)
    return root


def hoist_mutation(root: ExpressionNode, rng: random.Random) -> ExpressionNode:
    """Replace the root with a random subtree (simplification)."""
    root = root.copy()
    all_nodes = root.nodes()
    # Only hoist if there are non-trivial subtrees
    candidates = [n for n in all_nodes if n is not root and n.children]
    if not candidates:
        candidates = [n for n in all_nodes if n is not root]
    if not candidates:
        return root
    return rng.choice(candidates).copy()


def constant_perturbation(
    root: ExpressionNode,
    rng: random.Random,
    scale: float = 0.2,
) -> ExpressionNode:
    """Perturb a random constant node by +/- scale * |value|."""
    root = root.copy()
    consts: list[ExpressionNode] = []

    def _find_consts(node: ExpressionNode) -> None:
        if node.kind == "const":
            consts.append(node)
        for child in node.children:
            _find_consts(child)

    _find_consts(root)
    if not consts:
        return root
    target = rng.choice(consts)
    val = float(target.value)
    delta = rng.gauss(0, max(abs(val) * scale, 0.1))
    target.value = val + delta
    return root


def depth_limited_crossover(
    left: ExpressionNode,
    right: ExpressionNode,
    max_depth: int,
    rng: random.Random,
    max_attempts: int = 5,
) -> ExpressionNode:
    """Crossover that rejects offspring exceeding max_depth."""
    for _ in range(max_attempts):
        candidate = crossover_expression(left.copy(), right, rng)
        if tree_depth(candidate) <= max_depth:
            return candidate
    # Fallback: just return a mutated copy of left
    return left.copy()

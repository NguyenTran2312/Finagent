"""Whitelisted formula language: parse, dimension inference and exact evaluation."""

from __future__ import annotations

import ast
from collections.abc import Callable
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation

from src.calculation.units import _DIV_TABLE, Dimension, Quantity, UnitError

_DAYS_IN_YEAR = 365
_ADD_SUB = (ast.Add, ast.Sub)
_BIN_OPS = (ast.Add, ast.Sub, ast.Mult, ast.Div)


class FormulaError(ValueError):
    """Raised when formula text is malformed or uses a construct outside the whitelist."""


@dataclass(frozen=True)
class Ref:
    """Reference to a node value; offset 0 is the current period, 1 is t-1, and so on."""

    node_id: str
    offset: int

    def __post_init__(self) -> None:
        if not isinstance(self.node_id, str) or not self.node_id:
            raise ValueError("Ref node_id must be a non-empty string")
        if isinstance(self.offset, bool) or not isinstance(self.offset, int) or self.offset < 0:
            raise ValueError("Ref offset must be an int >= 0")


def _reject(node: ast.AST, why: str = "") -> FormulaError:
    detail = f": {why}" if why else ""
    return FormulaError(f"unsupported construct {type(node).__name__}{detail}")


def _is_int_constant(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Constant)
        and isinstance(node.value, int)
        and not isinstance(node.value, bool)
    )


def _check_name(node: ast.Name) -> None:
    if "__" in node.id:
        raise _reject(node, "dunder names are not allowed")


def _parse_ref(node: ast.Subscript) -> Ref:
    """Accept only `Name[t - <int >= 1>]`."""
    name = node.value
    if not isinstance(name, ast.Name):
        raise _reject(node, "subscript base must be a name")
    _check_name(name)
    sl = node.slice
    if not (
        isinstance(sl, ast.BinOp)
        and isinstance(sl.op, ast.Sub)
        and isinstance(sl.left, ast.Name)
        and sl.left.id == "t"
        and _is_int_constant(sl.right)
        and sl.right.value >= 1
    ):
        raise _reject(node, "subscript must be [t - n] with integer n >= 1")
    return Ref(name.id, sl.right.value)


def _constant_value(node: ast.Constant, text: str) -> Decimal:
    """Exact Decimal of the literal as written (never routed through float repr)."""
    value = node.value
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _reject(node, "only numeric constants are allowed")
    if isinstance(value, float) and value in (float("inf"), float("-inf")):
        raise _reject(node, "non-finite constant")
    segment = ast.get_source_segment(text, node)
    try:
        return Decimal(segment) if segment is not None else Decimal(str(value))
    except InvalidOperation:
        return Decimal(str(value))


def _validate(node: ast.AST, text: str, refs: set[Ref]) -> None:
    if isinstance(node, ast.BinOp):
        if not isinstance(node.op, _BIN_OPS):
            raise _reject(node, type(node.op).__name__)
        _validate(node.left, text, refs)
        _validate(node.right, text, refs)
    elif isinstance(node, ast.UnaryOp):
        if not isinstance(node.op, ast.USub):
            raise _reject(node, type(node.op).__name__)
        _validate(node.operand, text, refs)
    elif isinstance(node, ast.Constant):
        _constant_value(node, text)
    elif isinstance(node, ast.Name):
        _check_name(node)
        refs.add(Ref(node.id, 0))
    elif isinstance(node, ast.Call):
        if not (
            isinstance(node.func, ast.Name)
            and node.func.id == "abs"
            and len(node.args) == 1
            and not isinstance(node.args[0], ast.Starred)
            and not node.keywords
        ):
            raise _reject(node, "only abs(x) is allowed")
        _validate(node.args[0], text, refs)
    elif isinstance(node, ast.Subscript):
        refs.add(_parse_ref(node))
    else:
        raise _reject(node)


def _days_form(body: ast.AST) -> bool:
    return (
        isinstance(body, ast.BinOp)
        and isinstance(body.op, ast.Mult)
        and _is_int_constant(body.right)
        and body.right.value == _DAYS_IN_YEAR
    )


class Formula:
    """A validated formula. Pickles by text and is re-parsed on load (D6)."""

    __slots__ = ("text", "refs", "is_days_form", "_body")

    def __init__(self, text: str, refs: frozenset[Ref], is_days_form: bool, body: ast.expr):
        self.text = text
        self.refs = refs
        self.is_days_form = is_days_form
        self._body = body

    def __reduce__(self):
        return (parse_formula, (self.text,))

    def __repr__(self) -> str:
        return f"Formula({self.text!r})"


def parse_formula(text: str) -> Formula:
    if not isinstance(text, str):
        raise FormulaError("formula text must be a string")
    try:
        tree = ast.parse(text, mode="eval")
        refs: set[Ref] = set()
        _validate(tree.body, text, refs)
    except FormulaError:
        raise
    except (SyntaxError, ValueError, RecursionError, MemoryError) as exc:
        raise FormulaError(f"invalid formula syntax ({type(exc).__name__})") from None
    return Formula(text, frozenset(refs), _days_form(tree.body), tree.body)


def _combine_dimensions(op: ast.operator, left: Dimension, right: Dimension) -> Dimension:
    """Dimension rules identical to Quantity's arithmetic, without any values."""
    if isinstance(op, _ADD_SUB):
        if left is not right:
            raise UnitError(f"cannot add/subtract {left} and {right}")
        return left
    if isinstance(op, ast.Mult):
        if right is Dimension.RATIO:
            return left
        if left is Dimension.RATIO:
            return right
        raise UnitError(f"cannot multiply {left} by {right}")
    dim = _DIV_TABLE.get((left, right))
    if dim is None:
        raise UnitError(f"cannot divide {left} by {right}")
    return dim


def _infer(node: ast.AST, dimension_of: Callable[[str], Dimension], days: bool) -> Dimension:
    if isinstance(node, ast.BinOp):
        if days:
            left = _infer(node.left, dimension_of, False)
            if left is not Dimension.RATIO:
                raise UnitError(f"days form requires a ratio, got {left}")
            return Dimension.DAYS
        return _combine_dimensions(
            node.op,
            _infer(node.left, dimension_of, False),
            _infer(node.right, dimension_of, False),
        )
    if isinstance(node, ast.UnaryOp):
        return _infer(node.operand, dimension_of, False)
    if isinstance(node, ast.Constant):
        return Dimension.RATIO
    if isinstance(node, ast.Name):
        return dimension_of(node.id)
    if isinstance(node, ast.Call):
        return _infer(node.args[0], dimension_of, False)
    if isinstance(node, ast.Subscript):
        return dimension_of(node.value.id)
    raise _reject(node)


def infer_dimension(formula: Formula, dimension_of: Callable[[str], Dimension]) -> Dimension:
    return _infer(formula._body, dimension_of, formula.is_days_form)


def _eval(node: ast.AST, text: str, resolve: Callable[[Ref], Quantity], days: bool) -> Quantity:
    if isinstance(node, ast.BinOp):
        left = _eval(node.left, text, resolve, False)
        if days:
            if left.dimension is not Dimension.RATIO:
                raise UnitError(f"days form requires a ratio, got {left.dimension}")
            return Quantity(left.value * _DAYS_IN_YEAR, Dimension.DAYS)
        right = _eval(node.right, text, resolve, False)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        return left / right
    if isinstance(node, ast.UnaryOp):
        return -_eval(node.operand, text, resolve, False)
    if isinstance(node, ast.Constant):
        return Quantity(_constant_value(node, text), Dimension.RATIO)
    if isinstance(node, ast.Name):
        return resolve(Ref(node.id, 0))
    if isinstance(node, ast.Call):
        return abs(_eval(node.args[0], text, resolve, False))
    if isinstance(node, ast.Subscript):
        return resolve(_parse_ref(node))
    raise _reject(node)


def evaluate(formula: Formula, resolve: Callable[[Ref], Quantity]) -> Quantity:
    return _eval(formula._body, formula.text, resolve, formula.is_days_form)

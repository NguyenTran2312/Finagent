"""Exact quantities with dimension algebra and display-unit conversion."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal, localcontext
from enum import StrEnum
from typing import Literal


class Dimension(StrEnum):
    MONEY = "money"
    RATIO = "ratio"
    MONEY_PER_SHARE = "money_per_share"
    SHARES = "shares"
    DAYS = "days"
    COUNT = "count"


class UnitError(ValueError):
    """Raised when an operation mixes incompatible dimensions or units."""


class CalculationError(ArithmeticError):
    """Raised when a calculation is undefined (e.g. division by zero)."""


_DIV_TABLE: dict[tuple[Dimension, Dimension], Dimension] = {
    (Dimension.MONEY, Dimension.SHARES): Dimension.MONEY_PER_SHARE,
}
for _dim in Dimension:
    _DIV_TABLE[(_dim, _dim)] = Dimension.RATIO
    _DIV_TABLE.setdefault((_dim, Dimension.RATIO), _dim)

_NUMBER_COMPARABLE = (Dimension.RATIO, Dimension.COUNT)


def _coerce(other: object) -> Quantity | None:
    """Treat int/Decimal as a dimensionless RATIO; None if unsupported."""
    if isinstance(other, Quantity):
        return other
    if isinstance(other, bool):
        raise TypeError("bool is not a valid numeric operand")
    if isinstance(other, (int, Decimal)):
        return Quantity(Decimal(other), Dimension.RATIO)
    if isinstance(other, float):
        raise TypeError("float operands are not allowed; use Decimal")
    return None


def _is_exact_zero(other: object) -> bool:
    """True for int 0 or a zero Decimal (never bool): the dimensionless additive identity."""
    if isinstance(other, bool):
        return False
    if isinstance(other, Decimal):
        return other.is_zero()
    return isinstance(other, int) and other == 0


@dataclass(frozen=True, slots=True)
class Quantity:
    value: Decimal
    dimension: Dimension

    def __post_init__(self) -> None:
        if not isinstance(self.value, Decimal):
            raise TypeError(f"Quantity value must be Decimal, got {type(self.value).__name__}")
        if not self.value.is_finite():
            raise UnitError("Quantity value must be finite")
        if not isinstance(self.dimension, Dimension):
            raise TypeError("Quantity dimension must be a Dimension")

    # --- additive ---
    def _additive(self, other: object, sign: int, reverse: bool) -> Quantity:
        if _is_exact_zero(other):
            # Exact zero is the identity for any dimension, so builtin sum() works on money.
            return -self if reverse and sign < 0 else self
        o = _coerce(other)
        if o is None:
            return NotImplemented
        if o.dimension is not self.dimension:
            raise UnitError(f"cannot add/subtract {self.dimension} and {o.dimension}")
        a, b = (o.value, self.value) if reverse else (self.value, o.value)
        return Quantity(a + sign * b, self.dimension)

    def __add__(self, other):
        return self._additive(other, 1, False)

    def __radd__(self, other):
        return self._additive(other, 1, True)

    def __sub__(self, other):
        return self._additive(other, -1, False)

    def __rsub__(self, other):
        return self._additive(other, -1, True)

    # --- multiplicative ---
    def __mul__(self, other):
        o = _coerce(other)
        if o is None:
            return NotImplemented
        if o.dimension is Dimension.RATIO:
            return Quantity(self.value * o.value, self.dimension)
        if self.dimension is Dimension.RATIO:
            return Quantity(self.value * o.value, o.dimension)
        raise UnitError(f"cannot multiply {self.dimension} by {o.dimension}")

    __rmul__ = __mul__

    def _divide(self, num: Quantity, den: Quantity) -> Quantity:
        dim = _DIV_TABLE.get((num.dimension, den.dimension))
        if dim is None:
            raise UnitError(f"cannot divide {num.dimension} by {den.dimension}")
        if den.value == 0:
            raise CalculationError("division by zero")
        return Quantity(num.value / den.value, dim)

    def __truediv__(self, other):
        o = _coerce(other)
        if o is None:
            return NotImplemented
        return self._divide(self, o)

    def __rtruediv__(self, other):
        o = _coerce(other)
        if o is None:
            return NotImplemented
        return self._divide(o, self)

    def __neg__(self) -> Quantity:
        return Quantity(-self.value, self.dimension)

    def __abs__(self) -> Quantity:
        return Quantity(abs(self.value), self.dimension)

    # --- ordering ---
    def _comparable(self, other: object) -> Quantity:
        if isinstance(other, Quantity):
            o = other
        else:
            o = _coerce(other)
            if o is None:
                return NotImplemented
            if self.dimension not in _NUMBER_COMPARABLE and not _is_exact_zero(other):
                raise UnitError(f"cannot compare {self.dimension} with a bare number")
            o = Quantity(o.value, self.dimension)
        if o.dimension is not self.dimension:
            raise UnitError(f"cannot compare {self.dimension} with {o.dimension}")
        return o

    def __lt__(self, other):
        o = self._comparable(other)
        return o if o is NotImplemented else self.value < o.value

    def __le__(self, other):
        o = self._comparable(other)
        return o if o is NotImplemented else self.value <= o.value

    def __gt__(self, other):
        o = self._comparable(other)
        return o if o is NotImplemented else self.value > o.value

    def __ge__(self, other):
        o = self._comparable(other)
        return o if o is NotImplemented else self.value >= o.value


MONEY_SCALES: dict[str, Decimal] = {
    "dong": Decimal(1),
    "nghin": Decimal(10) ** 3,
    "trieu": Decimal(10) ** 6,
    "ty": Decimal(10) ** 9,
    "tram_ty": Decimal(10) ** 11,
    "nghin_ty": Decimal(10) ** 12,
}

_NON_MONEY_KINDS = (
    "percent",
    "percentage_points",
    "times",
    "days",
    "money_per_share",
    "count",
    "shares",
)


@dataclass(frozen=True)
class DisplayUnit:
    kind: Literal[
        "money", "percent", "percentage_points", "times", "days",
        "money_per_share", "count", "shares",
    ]
    scale: Decimal = Decimal(1)


def display_unit(name: str) -> DisplayUnit:
    if name in MONEY_SCALES:
        return DisplayUnit("money", MONEY_SCALES[name])
    if name in _NON_MONEY_KINDS:
        return DisplayUnit(name)  # type: ignore[arg-type]
    raise UnitError(f"unknown display unit: {name!r}")


_KIND_DIMENSIONS: dict[str, tuple[Dimension, ...]] = {
    "money": (Dimension.MONEY, Dimension.MONEY_PER_SHARE),
    "percent": (Dimension.RATIO,),
    "percentage_points": (Dimension.RATIO,),
    "times": (Dimension.RATIO,),
    "days": (Dimension.DAYS,),
    "money_per_share": (Dimension.MONEY_PER_SHARE,),
    "count": (Dimension.COUNT,),
    "shares": (Dimension.SHARES,),
}


def to_display(q: Quantity, unit: DisplayUnit) -> Decimal:
    if q.dimension not in _KIND_DIMENSIONS[unit.kind]:
        raise UnitError(f"cannot display {q.dimension} as {unit.kind}")
    if unit.kind in ("percent", "percentage_points"):
        return q.value * 100
    if unit.kind in ("money", "money_per_share"):
        return q.value / unit.scale
    return q.value


def round_display(value: Decimal, places: int = 2) -> Decimal:
    with localcontext() as ctx:
        ctx.prec = max(ctx.prec, 100)
        return value.quantize(Decimal(1).scaleb(-places), rounding=ROUND_HALF_UP)

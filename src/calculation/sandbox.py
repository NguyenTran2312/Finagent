"""Run LLM-written `def solution(): ...` code under an AST whitelist in a spawned process.

The code is untrusted. Defence layers:

1. `check_code` accepts exactly one top-level `def solution():` built from a small whitelist
   of AST nodes (spec §10) that also compiles. Attribute access, names starting with `_` or
   containing `__` (except the bare name `_`), imports, loops other than `for`, lambdas,
   nested defs, `**`, star unpacking, f-strings and calls to anything but plain helper /
   allowed-builtin names are rejected. Helpers are
   closures whose `__closure__` reaches the FactSet and Trace, so forbidding every
   `Attribute` node is what keeps generated code away from those internals.
2. Every numeric literal is rewritten to `_num("<source text>")`, i.e. an exact `Decimal` (D3).
   Subscript indices and slice bounds are wrapped in `_index(...)`, which turns an integral
   Decimal into an int so `xs[0]` / `xs[1:3]` work (R13); `range` and `enumerate` accept
   integral Decimals the same way.
3. The code runs in a fresh `multiprocessing` "spawn" process whose `__builtins__` holds only
   `ALLOWED_BUILTINS`, so `__import__`, `open`, `getattr` etc. do not exist at runtime either.

The AST whitelist is the only real barrier. The child process is NOT isolated at the OS level
(same user, filesystem and network access as the parent) and on Windows has no memory cap;
restricted builtins and the separate process are a second line of defence, not a jail.

Child protocol (D7): the child sends ("started",) right before calling `solution()`, then
("ok", answer, trace) or ("error", kind, class_name); a failure while setting up sends
("startup_error", class_name) instead of "started". The parent allows `startup_timeout` for
the spawn and `timeout` for `solution()` itself; on expiry the child is terminated, then
killed. Exception text never leaves the child: failures carry the exception class name only.

Resource limits: the result is capped at `MAX_TRACE_STEPS` trace steps and `MAX_RESULT_BYTES`
pickled bytes. On POSIX the child applies `RLIMIT_AS` (`MEMORY_LIMIT_BYTES`) when the
`resource` module supports it. On Windows NO memory cap is enforced (documented risk): a
memory-hungry program is bounded only by the timeout and the AST whitelist.

Deployment note: "spawn" re-imports the parent's `__main__` module in the child (except
`python -m pkg` entry points). Entry scripts that call `run` must keep heavyweight imports
(torch, langchain, ...) under `if __name__ == "__main__":` or the child pays for them.
"""

from __future__ import annotations

import ast
import builtins
import math
import multiprocessing
import pickle
import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from multiprocessing import connection as mp_connection
from typing import Literal

from src.calculation.facts import FactSet
from src.calculation.helpers import Answer, ExecutionContext, Trace, build_helpers
from src.calculation.metric_tree import MetricTree
from src.calculation.units import CalculationError

ALLOWED_BUILTINS = ("len", "min", "max", "sum", "abs", "sorted", "range", "enumerate", "zip",
                    "list", "dict")
HELPER_NAMES = frozenset({
    "fact", "metric", "avg_balance", "growth_pct", "cagr", "pct_point_change", "mean", "median",
    "argmax", "argmin", "filter_by", "count_if", "money", "percent", "answer",
})
MAX_CODE_CHARS = 20_000
MAX_TRACE_STEPS = 10_000
MAX_RESULT_BYTES = 16 * 1024 * 1024
MEMORY_LIMIT_BYTES = 2 * 1024 ** 3  # POSIX only
_STOP_GRACE_SECONDS = 1.0

FailureKind = Literal["rejected", "timeout", "calculation_error", "runtime_error"]


@dataclass(frozen=True)
class SyntaxRejected:
    line: int
    node: str
    reason: str


@dataclass(frozen=True)
class ExecutionResult:
    answer: Answer
    trace: Trace


@dataclass(frozen=True)
class ExecutionFailure:
    kind: FailureKind
    detail: str
    rejections: tuple[SyntaxRejected, ...] = ()


# ---- static check ---------------------------------------------------------------------
_CALLABLE = HELPER_NAMES | frozenset(ALLOWED_BUILTINS)

_ALLOWED_NODES = (
    # statements
    ast.Assign, ast.AugAssign, ast.If, ast.For, ast.Return, ast.Pass, ast.Break, ast.Continue,
    ast.Expr,
    # expressions
    ast.BoolOp, ast.BinOp, ast.UnaryOp, ast.IfExp, ast.Dict, ast.List, ast.Tuple, ast.ListComp,
    ast.DictComp, ast.GeneratorExp, ast.Compare, ast.Call, ast.Constant, ast.Name,
    ast.Subscript, ast.Slice,
    # helpers, contexts and operators
    ast.comprehension, ast.keyword, ast.Load, ast.Store,
    ast.And, ast.Or, ast.Not, ast.UAdd, ast.USub, ast.Add, ast.Sub, ast.Mult, ast.Div,
    ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.In, ast.NotIn, ast.Is, ast.IsNot,
)

_FORBIDDEN_REASONS: dict[type, str] = {
    ast.Attribute: "attribute access is forbidden",
    ast.Import: "import is forbidden",
    ast.ImportFrom: "import is forbidden",
    ast.While: "while loops are forbidden",
    ast.Lambda: "lambda is forbidden",
    ast.Try: "try is forbidden",
    ast.With: "with is forbidden",
    ast.AsyncWith: "with is forbidden",
    ast.Global: "global is forbidden",
    ast.Nonlocal: "nonlocal is forbidden",
    ast.ClassDef: "class definitions are forbidden",
    ast.FunctionDef: "nested function definitions are forbidden",
    ast.AsyncFunctionDef: "async functions are forbidden",
    ast.JoinedStr: "f-strings are forbidden",
    ast.FormattedValue: "f-strings are forbidden",
    ast.Pow: "** is forbidden",
    ast.Starred: "star unpacking is forbidden",
    ast.NamedExpr: "assignment expressions are forbidden",
    ast.Delete: "del is forbidden",
    ast.Assert: "assert is forbidden",
    ast.Raise: "raise is forbidden",
    ast.Yield: "yield is forbidden",
    ast.YieldFrom: "yield is forbidden",
    ast.Await: "await is forbidden",
    ast.Match: "match is forbidden",
}

_BAD_NAME = "names starting with '_' or containing '__' are forbidden"


def _bad_identifier(name: str) -> bool:
    # The bare throwaway name `_` is allowed; it is not an injected internal (_num, _index).
    return name != "_" and (name.startswith("_") or "__" in name)


class _SourceLines:
    """Source text of single-line nodes in linear total time (R15).

    `ast.get_source_segment` re-splits the whole source on every call, which is quadratic over
    many literals. Lines are split once (on the same line breaks as the tokenizer) and each is
    encoded lazily, because AST column offsets are UTF-8 byte offsets.
    """

    def __init__(self, code: str):
        parts = re.split(r"(\r\n|\r|\n)", code)
        self._lines = parts[::2]
        self._encoded: dict[int, bytes] = {}

    def segment(self, node: ast.expr) -> str | None:
        end_line, end_col = getattr(node, "end_lineno", None), getattr(node, "end_col_offset", None)
        index = node.lineno - 1
        if end_line != node.lineno or end_col is None or not 0 <= index < len(self._lines):
            return None
        line = self._encoded.get(index)
        if line is None:
            line = self._encoded[index] = self._lines[index].encode("utf-8")
        try:
            return line[node.col_offset:end_col].decode("utf-8")
        except UnicodeDecodeError:
            return None


class _Checker:
    def __init__(self, code: str):
        self.source = _SourceLines(code)
        self.issues: list[SyntaxRejected] = []

    def visit(self, node: ast.AST, line: int) -> None:
        line = getattr(node, "lineno", line)
        reason = self._problem(node)
        if reason is not None:
            self.issues.append(SyntaxRejected(line, type(node).__name__, reason))
            return  # one rejection per offending subtree
        for child in ast.iter_child_nodes(node):
            self.visit(child, line)

    def _problem(self, node: ast.AST) -> str | None:
        reason = _FORBIDDEN_REASONS.get(type(node))
        if reason is not None:
            return reason
        if not isinstance(node, _ALLOWED_NODES):
            return f"{type(node).__name__} is not allowed"
        if isinstance(node, ast.Name):
            if _bad_identifier(node.id):
                return _BAD_NAME
            if isinstance(node.ctx, ast.Store) and node.id in _CALLABLE:
                return f"cannot assign to helper or builtin name {node.id!r}"
        elif isinstance(node, ast.Call):
            if not (isinstance(node.func, ast.Name) and node.func.id in _CALLABLE):
                return "only helpers and allowed builtins may be called"
        elif isinstance(node, ast.keyword):
            if node.arg is None:
                return "** unpacking is forbidden"
            if _bad_identifier(node.arg):
                return _BAD_NAME
        elif isinstance(node, ast.Dict):
            if any(key is None for key in node.keys):
                return "** unpacking is forbidden"
        elif isinstance(node, ast.comprehension):
            if node.is_async:
                return "async comprehensions are forbidden"
        elif isinstance(node, ast.Constant):
            return self._constant_problem(node)
        return None

    def _constant_problem(self, node: ast.Constant) -> str | None:
        value = node.value
        if value is None or isinstance(value, (bool, str)):  # strings are plain values (R14)
            return None
        if isinstance(value, (int, float)):
            text = self.source.segment(node)
            try:
                if text is None or not Decimal(text).is_finite():
                    raise InvalidOperation
            except InvalidOperation:
                return "numeric literals must be plain decimal numbers"
            return None
        return f"{type(value).__name__} literals are forbidden"


def _signature_problems(func: ast.FunctionDef) -> list[SyntaxRejected]:
    issues = []
    args = func.args
    if args.posonlyargs or args.args or args.vararg or args.kwonlyargs or args.kwarg:
        issues.append(SyntaxRejected(func.lineno, "arguments", "solution() must take no parameters"))
    if func.decorator_list:
        issues.append(SyntaxRejected(func.lineno, "FunctionDef", "decorators are forbidden"))
    if func.returns is not None:
        issues.append(SyntaxRejected(func.lineno, "FunctionDef", "annotations are forbidden"))
    if getattr(func, "type_params", None):
        issues.append(SyntaxRejected(func.lineno, "FunctionDef", "type parameters are forbidden"))
    return issues


def check_code(code: str) -> list[SyntaxRejected]:
    """Every whitelist violation in `code`; an empty list means the code may run."""
    if not isinstance(code, str):
        raise TypeError(f"code must be a str, got {type(code).__name__}")
    if len(code) > MAX_CODE_CHARS:
        return [SyntaxRejected(0, "Module", f"code too long (max {MAX_CODE_CHARS} characters)")]
    try:
        module = ast.parse(code, "<solution>", "exec")
    except SyntaxError as exc:
        return [SyntaxRejected(exc.lineno or 0, "SyntaxError", "invalid Python syntax")]
    except (ValueError, RecursionError, MemoryError) as exc:
        return [SyntaxRejected(0, type(exc).__name__, "code cannot be parsed")]

    issues: list[SyntaxRejected] = []
    solution: ast.FunctionDef | None = None
    for node in module.body:
        if solution is None and isinstance(node, ast.FunctionDef) and node.name == "solution":
            solution = node
        else:
            issues.append(SyntaxRejected(node.lineno, type(node).__name__,
                                         "only one top-level def solution() is allowed"))
    if solution is None:
        issues.append(SyntaxRejected(1, "Module", "missing def solution()"))
        return issues
    issues.extend(_signature_problems(solution))
    checker = _Checker(code)
    try:
        for statement in solution.body:
            checker.visit(statement, solution.lineno)
    except RecursionError:
        checker.issues.append(SyntaxRejected(solution.lineno, "Module", "code is nested too deeply"))
    issues += checker.issues
    if not issues:  # compiler-only errors, e.g. `break` outside a loop
        try:
            compile(module, "<solution>", "exec", dont_inherit=True)
        except SyntaxError as exc:
            issues.append(SyntaxRejected(exc.lineno or 0, "Module", f"invalid syntax: {exc.msg}"))
        except (ValueError, RecursionError, MemoryError) as exc:
            issues.append(SyntaxRejected(0, type(exc).__name__, "code cannot be compiled"))
    return issues


# ---- child process --------------------------------------------------------------------
def _num(text: str) -> Decimal:
    """Exact value of a numeric literal from its source text (D3)."""
    return Decimal(text)


def _call(name: str, arg: ast.expr) -> ast.Call:
    return ast.copy_location(ast.Call(ast.Name(name, ast.Load()), [arg], []), arg)


class _LiteralRewriter(ast.NodeTransformer):
    """Numeric literals -> `_num("<source text>")` (D3); indices and slice bounds -> `_index(...)` (R13)."""

    def __init__(self, code: str):
        self.source = _SourceLines(code)

    def visit_Constant(self, node: ast.Constant) -> ast.AST:
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            return node
        text = self.source.segment(node)
        if text is None:
            raise ValueError("numeric literal without source position")
        return ast.copy_location(ast.Call(ast.Name("_num", ast.Load()), [ast.Constant(text)], []),
                                 node)

    def visit_Subscript(self, node: ast.Subscript) -> ast.AST:
        self.generic_visit(node)  # rewrites literals; Slice bounds are wrapped by visit_Slice
        index = node.slice
        if isinstance(index, ast.Tuple) and any(isinstance(e, ast.Slice) for e in index.elts):
            index.elts = [e if isinstance(e, ast.Slice) else _call("_index", e) for e in index.elts]
        elif not isinstance(index, ast.Slice):
            node.slice = _call("_index", index)
        return node

    def visit_Slice(self, node: ast.Slice) -> ast.AST:
        self.generic_visit(node)
        for name in ("lower", "upper", "step"):
            bound = getattr(node, name)
            if bound is not None:
                setattr(node, name, _call("_index", bound))
        return node


def _rewrite_literals(code: str) -> ast.Module:
    """Parse `code`, rewrite numeric literals to `_num(...)` and wrap indices in `_index(...)`."""
    module = _LiteralRewriter(code).visit(ast.parse(code, "<solution>", "exec"))
    return ast.fix_missing_locations(module)


def _int_argument(value: object, what: str) -> object:
    """An integral Decimal (D3 literal) as int; other Decimals raise, non-Decimals pass through."""
    if isinstance(value, Decimal):
        if not value.is_finite() or value != value.to_integral_value():
            raise TypeError(f"{what} needs integers")
        if not value.is_zero() and value.adjusted() > 18:
            raise OverflowError(f"{what} value too large")
        return int(value)
    return value


def _index(value: object) -> object:
    """Subscript key: an integral Decimal becomes an int (R13); anything else is unchanged.

    Non-integral or huge Decimals stay Decimal, so a list raises TypeError while a dict still
    finds a Decimal key. int(Decimal(2023)) hashes like Decimal(2023), so dict lookups by year
    work either way.
    """
    try:
        return _int_argument(value, "index")
    except (TypeError, OverflowError):
        return value


def _range(*args: object) -> range:
    return range(*(_int_argument(a, "range()") for a in args))


def _enumerate(iterable: object, start: object = 0) -> enumerate:
    return enumerate(iterable, _int_argument(start, "enumerate()"))


def _restricted_builtins() -> dict[str, object]:
    allowed = {name: getattr(builtins, name) for name in ALLOWED_BUILTINS}
    allowed["range"] = _range
    allowed["enumerate"] = _enumerate
    return allowed


def _limit_memory() -> None:
    try:
        import resource
    except ImportError:  # Windows: no memory cap (documented risk)
        return
    if hasattr(resource, "RLIMIT_AS"):
        _, hard = resource.getrlimit(resource.RLIMIT_AS)
        limit = MEMORY_LIMIT_BYTES if hard == resource.RLIM_INFINITY else min(MEMORY_LIMIT_BYTES, hard)
        resource.setrlimit(resource.RLIMIT_AS, (limit, hard))


def _execute(solution, trace: Trace) -> tuple:
    try:
        result = solution()
    except CalculationError as exc:
        return ("error", "calculation_error", type(exc).__name__)
    except BaseException as exc:  # noqa: BLE001 - only the class name leaves the child
        return ("error", "runtime_error", type(exc).__name__)
    if not isinstance(result, Answer):
        return ("error", "runtime_error", "InvalidReturn")
    if len(trace.steps) > MAX_TRACE_STEPS:
        return ("error", "runtime_error", "TraceOverflow")
    return ("ok", result, trace)


def _encode(message: tuple) -> bytes:
    try:
        payload = pickle.dumps(message)
    except Exception:  # noqa: BLE001 - e.g. a generator recorded in the trace
        return pickle.dumps(("error", "runtime_error", "UnpicklableResult"))
    if len(payload) > MAX_RESULT_BYTES:
        return pickle.dumps(("error", "runtime_error", "ResultTooLarge"))
    return payload


def _child_main(conn, code: str, tree: MetricTree, facts: FactSet,
                context: ExecutionContext) -> None:
    """Entry point of the spawned process (module level so spawn can pickle it)."""
    try:
        try:
            _limit_memory()
            trace = Trace()
            env = {"__builtins__": _restricted_builtins(),
                   **build_helpers(tree, facts, context, trace), "_num": _num, "_index": _index}
            exec(compile(_rewrite_literals(code), "<solution>", "exec", dont_inherit=True), env)
            solution = env["solution"]
        except BaseException as exc:  # noqa: BLE001 - only the class name leaves the child
            conn.send_bytes(pickle.dumps(("startup_error", type(exc).__name__)))
            return
        conn.send_bytes(pickle.dumps(("started",)))
        conn.send_bytes(_encode(_execute(solution, trace)))
    except OSError:
        return  # parent gone or pipe closed; nothing left to report
    finally:
        conn.close()


# ---- parent ---------------------------------------------------------------------------
_TIMED_OUT = object()
_DIED = object()
_UNREADABLE = object()


def _receive(conn, process, timeout: float) -> object:
    """Next message from the child, or one of the sentinels above."""
    try:
        ready = mp_connection.wait([conn, process.sentinel], timeout)
        if not ready:
            return _TIMED_OUT
        if conn not in ready and not conn.poll(0):
            return _DIED
        data = conn.recv_bytes(MAX_RESULT_BYTES)
    except (EOFError, OSError):
        return _DIED
    try:
        return pickle.loads(data)
    except Exception:  # noqa: BLE001
        return _UNREADABLE


def _interpret(message: object) -> ExecutionResult | ExecutionFailure:
    if message is _TIMED_OUT:
        return ExecutionFailure("timeout", "Timeout")
    if message is _DIED:
        return ExecutionFailure("runtime_error", "ProcessDied")
    if message is _UNREADABLE:
        return ExecutionFailure("runtime_error", "UnpicklableResult")
    if (isinstance(message, tuple) and len(message) == 3 and message[0] == "ok"
            and isinstance(message[1], Answer) and isinstance(message[2], Trace)):
        return ExecutionResult(message[1], message[2])
    if (isinstance(message, tuple) and len(message) == 3 and message[0] == "error"
            and message[1] in ("calculation_error", "runtime_error")
            and isinstance(message[2], str)):
        return ExecutionFailure(message[1], message[2])
    return ExecutionFailure("runtime_error", "ProtocolError")


def _stop(process, *, graceful: bool) -> None:
    if graceful:
        process.join(_STOP_GRACE_SECONDS)
    if process.is_alive():
        process.terminate()
        process.join(1)
    if process.is_alive():
        process.kill()
    process.join()


def _seconds(value: object, what: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{what} must be a number of seconds")
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{what} must be positive and finite")
    return float(value)


def run(code: str, tree: MetricTree, facts: FactSet, context: ExecutionContext, *,
        timeout: float = 2.0, startup_timeout: float = 20.0) -> ExecutionResult | ExecutionFailure:
    """Check and execute generated `solution()` code; every failure is a typed value."""
    timeout = _seconds(timeout, "timeout")
    startup_timeout = _seconds(startup_timeout, "startup_timeout")
    rejections = check_code(code)
    if rejections:
        first = rejections[0]
        return ExecutionFailure("rejected", f"{first.node} at line {first.line}: {first.reason}",
                                tuple(rejections))

    ctx = multiprocessing.get_context("spawn")
    reader, writer = ctx.Pipe(duplex=False)
    process = ctx.Process(target=_child_main, args=(writer, code, tree, facts, context),
                          name="calculation-sandbox", daemon=True)
    started = finished = False
    try:
        process.start()
        started = True
        writer.close()  # the child holds its own copy; EOF must reach us if it dies
        greeting = _receive(reader, process, startup_timeout)
        if greeting != ("started",):
            if (isinstance(greeting, tuple) and len(greeting) == 2
                    and greeting[0] == "startup_error" and isinstance(greeting[1], str)):
                return ExecutionFailure("runtime_error", f"StartupFailure:{greeting[1]}")
            return ExecutionFailure("runtime_error", "StartupFailure")
        message = _receive(reader, process, timeout)
        finished = message is not _TIMED_OUT
        return _interpret(message)
    finally:
        writer.close()
        if started:
            _stop(process, graceful=finished)
        process.close()
        reader.close()

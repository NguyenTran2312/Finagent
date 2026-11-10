import ast
import multiprocessing
import os
import time
import unittest
from decimal import Decimal as D
from unittest import mock

from src.calculation import sandbox
from src.calculation.facts import FactSet
from src.calculation.helpers import Answer, Trace, build_helpers
from src.calculation.sandbox import (
    ALLOWED_BUILTINS,
    HELPER_NAMES,
    ExecutionFailure,
    ExecutionResult,
    SyntaxRejected,
    check_code,
    run,
)
from tests.test_calculation_helpers import M, VJC_2023, context, fact, make_tree

ESCAPES = {
 "import": "import os\ndef solution(): return 1",
 "dunder_import": "def solution(): return __import__('os')",
 "class_chain": "def solution(): return ().__class__.__bases__",
 "getattr": "def solution(): return getattr(1, 'real')",
 "open": "def solution(): return open('x')",
 "eval": "def solution(): return eval('1')",
 "while": "def solution():\n    while True: pass",
 "lambda": "def solution(): return (lambda: 1)()",
 "pow": "def solution(): return 10 ** 10 ** 10",
 "try": "def solution():\n    try: return 1\n    except: return 2",
 "two_defs": "def solution(): return 1\ndef other(): return 2",
 "no_solution": "x = 1",
 "fstring": "def solution(): return f'{1}'",
 "round": "def solution(): return round(1)",
 "nested_def": "def solution():\n    def g(): return 1\n    return g()",
}

VALID = """\
def solution():
    years = [2021, 2022, 2023]
    icr = {y: metric("interest_coverage", "VJC", y) for y in years}
    best = argmax(icr)
    peers = dict(VJC=metric("ebit", "VJC", 2023), HPG=metric("ebit", "HPG", 2023))
    low = filter_by(peers, "<", money(20, "ty"))
    total = sum(peers[t] for t in peers) if False else 0
    values = sorted([abs(low[t]) for t in low])[0:2]
    n = 0
    for i, y in enumerate(years):
        if y > 2021 and not (y == 2023) or len(values) >= 1:
            n += 1
        elif y in years:
            continue
        else:
            break
    pair = (n, -n)
    a, b = pair
    x = 1.5 * percent(10) - 0.1 / 2 + max(a, b) - min(1, 2)
    return answer(x if n else percent(0), "percent", ) if best else answer(n, "count")
"""


def rejected_nodes(code):
    return {r.node for r in check_code(code)}


class CheckCodeTests(unittest.TestCase):
    def test_escapes_rejected_statically(self):
        for name, code in ESCAPES.items():
            with self.subTest(name):
                issues = check_code(code)
                self.assertTrue(issues)
                self.assertTrue(all(isinstance(i, SyntaxRejected) for i in issues))

    def test_escapes_run_returns_rejected_without_spawning(self):
        with mock.patch.object(sandbox.multiprocessing, "get_context",
                               side_effect=AssertionError("spawned")):
            for name, code in ESCAPES.items():
                with self.subTest(name):
                    result = run(code, make_tree(), FactSet(VJC_2023), context())
                    self.assertIsInstance(result, ExecutionFailure)
                    self.assertEqual(result.kind, "rejected")
                    self.assertTrue(result.rejections)
                    self.assertEqual(list(result.rejections), check_code(code))

    def test_rejection_names_node_and_line(self):
        self.assertEqual(check_code(ESCAPES["while"]),
                         [SyntaxRejected(2, "While", "while loops are forbidden")])
        self.assertIn("Attribute", rejected_nodes(ESCAPES["class_chain"]))
        self.assertIn("Pow", rejected_nodes(ESCAPES["pow"]))
        self.assertIn("JoinedStr", rejected_nodes(ESCAPES["fstring"]))
        self.assertIn("Import", rejected_nodes(ESCAPES["import"]))
        self.assertEqual(check_code(ESCAPES["two_defs"])[0].line, 2)
        failure = run(ESCAPES["while"], make_tree(), FactSet(VJC_2023), context())
        self.assertEqual(failure.detail, "While at line 2: while loops are forbidden")

    def test_valid_program_accepted(self):
        self.assertEqual(check_code(VALID), [])

    def test_closure_access_rejected(self):
        for code in ("def solution(): return metric.__closure__",
                     "def solution(): return metric.__closure__[0].cell_contents",
                     "def solution(): return answer.__globals__"):
            with self.subTest(code):
                self.assertIn("Attribute", rejected_nodes(code))

    def test_underscore_names_rejected(self):
        for code in ("def solution(): return _num('1')",
                     "def solution(): return _num",
                     "def solution():\n    _num = 1\n    return 1",
                     "def solution():\n    for _x in [1]: pass",
                     "def solution(): return __builtins__",
                     "def solution(): return answer(1, unit__x='count')",
                     "def solution(): return dict(_a=1)"):
            with self.subTest(code):
                self.assertTrue(check_code(code))

    def test_helper_and_builtin_shadowing_rejected(self):
        for code in ("def solution():\n    metric = 1\n    return metric",
                     "def solution():\n    sum = 1\n    return sum",
                     "def solution():\n    for answer in [1]: pass",
                     "def solution():\n    len += 1",
                     "def solution():\n    a, money = 1, 2",
                     "def solution(): return [1 for range in [1]]"):
            with self.subTest(code):
                self.assertIn("Name", rejected_nodes(code))

    def test_forbidden_statements_rejected(self):
        cases = {
            "Global": "def solution():\n    global x",
            "Nonlocal": "def solution():\n    nonlocal x",
            "ImportFrom": "def solution():\n    from os import path",
            "Delete": "def solution():\n    x = 1\n    del x",
            "Assert": "def solution():\n    assert 1",
            "Raise": "def solution():\n    raise ValueError",
            "With": "def solution():\n    with x: pass",
            "ClassDef": "def solution():\n    class A: pass",
            "NamedExpr": "def solution(): return (x := 1)",
            "Match": "def solution():\n    match 1:\n        case 1: pass",
            "Yield": "def solution():\n    yield 1",
            "YieldFrom": "def solution():\n    yield from [1]",
            "AsyncFunctionDef": "async def solution(): return 1",
            "Set": "def solution(): return {1}",
            "AnnAssign": "def solution():\n    x: int = 1",
        }
        for node, code in cases.items():
            with self.subTest(node):
                self.assertIn(node, rejected_nodes(code))
        self.assertTrue(check_code("async def f():\n    await x\ndef solution(): return 1"))

    def test_starred_and_double_star_rejected(self):
        for code in ("def solution(): return max(*[1, 2])",
                     "def solution(): return [*[1]]",
                     "def solution():\n    a, *b = [1, 2]",
                     "def solution(): return dict(**{})",
                     "def solution(): return {**{}}"):
            with self.subTest(code):
                self.assertTrue(check_code(code))
        self.assertIn("Starred", rejected_nodes("def solution(): return max(*[1, 2])"))

    def test_only_plain_whitelisted_calls(self):
        for code in ("def solution(): return metric('x')('y')",
                     "def solution(): return [len][0]([1])",
                     "def solution(): return foo()",
                     "def solution(): return solution()",
                     "def solution(): return exec('1')"):
            with self.subTest(code):
                self.assertIn("Call", rejected_nodes(code))

    def test_solution_signature(self):
        for code in ("def solution(x): return 1",
                     "def solution(x=1): return 1",
                     "def solution(*a): return 1",
                     "def solution(**k): return 1",
                     "def solution(*, k): return 1",
                     "@dict\ndef solution(): return 1",
                     "def solution() -> int: return 1",
                     "def solution[T](): return 1",
                     "def solution(): return 1\ndef solution(): return 2"):
            with self.subTest(code):
                self.assertTrue(check_code(code))

    def test_string_constants_allowed_as_plain_values(self):
        # R14: strings are ordinary values; only f-strings / t-strings stay forbidden.
        for code in ("def solution():\n    ts = ['VJC', 'HPG']\n    return ts",
                     "def solution():\n    d = dict(VJC=1)\n    return d['VJC']",
                     "def solution():\n    t = 'VJC'\n    return t == 'VJC'",
                     "def solution(): return answer('VJC' if c else 'HPG', 'ticker')",
                     "def solution(): return {'VJC': ('a', 'b')}",
                     "def solution():\n    '''docstrings are plain string values'''\n    return 1"):
            with self.subTest(code):
                self.assertEqual(check_code(code), [])
        self.assertIn("JoinedStr", rejected_nodes("def solution(): return f'{x}'"))
        self.assertTrue(check_code("def solution(): return t'{x}'"))

    def test_code_length_bound(self):
        code = "def solution(): return 1" + " " * (20_001 - 24)
        self.assertEqual(len(code), 20_001)
        started = time.monotonic()
        self.assertEqual(check_code(code), [SyntaxRejected(0, "Module",
                                                           "code too long (max 20000 characters)")])
        self.assertLess(time.monotonic() - started, 0.5)
        self.assertEqual(run(code, make_tree(), FactSet(()), context()).kind, "rejected")

    def test_many_literals_checked_in_linear_time(self):
        code = ("def solution():\n    xs = [" + ", ".join(["1.5"] * 2000)
                + "]\n    return answer(len(xs), 'count')")
        self.assertLess(len(code), 20_000)
        started = time.monotonic()
        self.assertEqual(check_code(code), [])
        sandbox._rewrite_literals(code)
        self.assertLess(time.monotonic() - started, 2.0)

    def test_literal_text_uses_utf8_columns(self):
        code = "def solution():\n    return f('đồng tỷ', 0.1, 'ữ', 2.50)\r\n# x\n"
        text = ast.unparse(sandbox._rewrite_literals(code))
        self.assertIn("_num('0.1')", text)
        self.assertIn("_num('2.50')", text)

    def test_literals(self):
        self.assertTrue(check_code("def solution(): return 1j"))
        self.assertTrue(check_code("def solution(): return 0x10"))
        self.assertTrue(check_code("def solution(): return b'x'"))
        self.assertTrue(check_code("def solution(): return ..."))
        self.assertTrue(check_code("def solution(): return 7 % 2"))
        self.assertTrue(check_code("def solution(): return 7 // 2"))
        self.assertTrue(check_code("def solution(): return ~1"))
        self.assertEqual(check_code("def solution(): return answer(1_000, 'count', ) or None"), [])
        self.assertEqual(check_code("def solution(): return money(1, scale='ty') if True else 1"), [])

    def test_unparseable_code_rejected(self):
        self.assertEqual(check_code("def solution(:")[0].node, "SyntaxError")
        self.assertTrue(check_code("def solution():\n    return " + "-" * 100000 + "1"))
        with self.assertRaises(TypeError):
            check_code(b"def solution(): return 1")

    def test_helper_names_match_build_helpers(self):
        built = build_helpers(make_tree(), FactSet(()), context(), Trace())
        self.assertEqual(HELPER_NAMES, frozenset(built))
        self.assertEqual(ALLOWED_BUILTINS, ("len", "min", "max", "sum", "abs", "sorted", "range",
                                            "enumerate", "zip", "list", "dict"))

    def test_numeric_literals_rewritten_from_source_text(self):
        module = sandbox._rewrite_literals("def solution():\n    return f('đồng', 0.1, True, -2, 1_0)")
        text = ast.unparse(module)
        self.assertIn("_num('0.1')", text)
        self.assertIn("-_num('2')", text)
        self.assertIn("_num('1_0')", text)
        self.assertIn("True", text)
        self.assertEqual(sandbox._num("0.1"), D("0.1"))

    def test_subscripts_wrapped_in_index(self):
        code = "def solution():\n    xs[0] = xs[len(xs) - 1] + xs[1:-1:2][0] + d['VJC'] + xs[:]"
        text = ast.unparse(sandbox._rewrite_literals(code))
        self.assertIn("xs[_index(_num('0'))] =", text)
        self.assertIn("xs[_index(len(xs) - _num('1'))]", text)
        self.assertIn("xs[_index(_num('1')):_index(-_num('1')):_index(_num('2'))]", text)
        self.assertIn("d[_index('VJC')]", text)
        self.assertIn("xs[:]", text)

    def test_index_conversion(self):
        self.assertEqual(sandbox._index(D("2")), 2)
        self.assertIsInstance(sandbox._index(D("2.0")), int)
        self.assertEqual(sandbox._index(D("-1")), -1)
        for value in ("VJC", D("0.5"), D("1E+40"), 3, True, None):
            with self.subTest(value):
                self.assertIs(sandbox._index(value), value)
        self.assertEqual({2023: "a"}[sandbox._index(D(2023))], "a")
        self.assertEqual({D(2023): "a"}[sandbox._index(D(2023))], "a")
        with self.assertRaises(TypeError):
            [1, 2][sandbox._index(D("0.5"))]
        self.assertEqual(list(sandbox._enumerate(["a"], D(1))), [(1, "a")])
        with self.assertRaises(TypeError):
            sandbox._enumerate(["a"], D("1.5"))

    def test_run_validates_timeouts(self):
        with self.assertRaises(ValueError):
            run(VALID, make_tree(), FactSet(()), context(), timeout=0)
        with self.assertRaises(ValueError):
            run(VALID, make_tree(), FactSet(()), context(), startup_timeout=float("nan"))


class _ExitOnUnpickle:
    """Unpickling this in the child exits the process before solution() starts."""

    def __reduce__(self):
        return (os._exit, (3,))


class SandboxRunTests(unittest.TestCase):
    """Each test spawns one process (~1-3 s on Windows)."""

    def tearDown(self):
        self.assertEqual(multiprocessing.active_children(), [])

    def run_code(self, code, facts=VJC_2023, **kwargs):
        return run(code, make_tree(), FactSet(facts), context(), **kwargs)

    def assertFailure(self, result, kind, detail):
        self.assertIsInstance(result, ExecutionFailure)
        self.assertEqual((result.kind, result.detail), (kind, detail))

    def test_success_returns_answer_and_trace(self):
        result = self.run_code(
            "def solution():\n"
            "    icr = metric('interest_coverage', 'VJC', 2023)\n"
            "    return answer(icr, 'times')")
        self.assertIsInstance(result, ExecutionResult)
        self.assertIsInstance(result.answer, Answer)
        self.assertIsInstance(result.trace, Trace)
        self.assertEqual(result.answer.kind, "quantity")
        self.assertGreater(result.answer.value.value, 1)
        self.assertIn("interest_coverage", result.trace.metrics_used())
        self.assertEqual(result.trace.steps[-1].helper, "answer")

    def test_literals_are_decimal(self):
        result = self.run_code(
            "def solution(): return answer(money(0.1,'dong') + money(0.2,'dong'), 'dong')")
        self.assertIsInstance(result, ExecutionResult)
        self.assertEqual(result.answer.value, M("0.3"))
        self.assertEqual(result.answer.value.value, D("0.3"))

    def test_unknown_name_runtime_error(self):
        self.assertFailure(self.run_code("def solution(): return foo"), "runtime_error", "NameError")

    def test_timeout(self):
        started = time.monotonic()
        result = self.run_code("def solution():\n    return sum(i for i in range(1000000000000))",
                               timeout=1.0)
        self.assertFailure(result, "timeout", "Timeout")
        self.assertLess(time.monotonic() - started, 20)

    def test_calculation_error_kind(self):
        self.assertFailure(
            self.run_code("def solution(): return answer(money(1,'dong') / money(0,'dong'), 'dong')"),
            "calculation_error", "CalculationError")

    def test_detail_has_no_exception_text(self):
        # answer() raises ValueError("unknown answer unit 'secret'"); only the class name leaves.
        result = self.run_code("def solution(): return answer(money(1, 'dong'), 'secret')")
        self.assertFailure(result, "runtime_error", "ValueError")
        self.assertNotIn("secret", repr(result))

    def test_invalid_return(self):
        self.assertFailure(self.run_code("def solution(): return 1"), "runtime_error", "InvalidReturn")

    def test_trace_overflow(self):
        code = ("def solution():\n"
                "    for i in range(10001):\n"
                "        x = percent(1)\n"
                "    return answer(x, 'percent')")
        self.assertFailure(self.run_code(code), "runtime_error", "TraceOverflow")

    def test_unpicklable_result(self):
        # a generator recorded in the trace cannot cross the process boundary
        code = ("def solution():\n"
                "    best = argmax({(i for i in [1]): percent(1)})\n"
                "    return answer(percent(1), 'percent')")
        self.assertFailure(self.run_code(code), "runtime_error", "UnpicklableResult")

    def test_child_dying_before_start_is_startup_failure(self):
        result = run("def solution(): return 1", make_tree(), _ExitOnUnpickle(), context())
        self.assertFailure(result, "runtime_error", "StartupFailure")

    def test_setup_error_names_class(self):
        # `break` outside a loop passes the AST check but fails to compile in the child
        result = self.run_code("def solution():\n    break")
        self.assertFailure(result, "runtime_error", "StartupFailure:SyntaxError")

    def test_unlisted_builtin_is_name_error(self):
        result = self.run_code("def solution():\n    x = getattr\n    return x")
        self.assertFailure(result, "runtime_error", "NameError")

    def test_indexing_slicing_and_enumerate_start(self):
        code = ("def solution():\n"
                "    xs = sorted([percent(30), percent(10), percent(20)])\n"
                "    first = sorted(xs)[0]\n"
                "    last = xs[len(xs) - 1]\n"
                "    assert_last = xs[-1]\n"
                "    total = percent(0)\n"
                "    for i, x in enumerate(xs[0:2], 1):\n"
                "        total = total + x * i\n"
                "    return answer(total + last - first + assert_last - last, 'percent')")
        result = self.run_code(code)
        self.assertIsInstance(result, ExecutionResult)
        # 0.10*1 + 0.20*2 + 0.30 - 0.10 = 0.70
        self.assertEqual(result.answer.value.value, D("0.70"))

    def test_valid_program_runs(self):
        facts = [fact(m, v, ticker=t, year=y)
                 for t in ("VJC", "HPG") for y in (2021, 2022, 2023)
                 for m, v in (("profit_before_tax", 100 + y), ("interest_expense", -50))]
        result = self.run_code(VALID, facts)
        self.assertIsInstance(result, ExecutionResult)
        # n = 3 -> 1.5 * 0.10 - 0.1 / 2 + max(3, -3) - min(1, 2) = 2.10
        self.assertEqual(result.answer.unit, "percent")
        self.assertEqual(result.answer.value.value, D("2.10"))

    def test_ticker_list_argmax(self):
        facts = [fact("profit_before_tax", 100, ticker="VJC"),
                 fact("interest_expense", -50, ticker="VJC"),
                 fact("profit_before_tax", 900, ticker="HPG"),
                 fact("interest_expense", -50, ticker="HPG")]
        code = ("def solution():\n"
                "    ts = ['VJC', 'HPG']\n"
                "    m = {}\n"
                "    for t in ts:\n"
                "        m[t] = metric('ebit', t, 2023)\n"
                "    best = 'VJC' if m['VJC'] > m['HPG'] else argmax(m)\n"
                "    return answer(argmax(m) if best == 'HPG' else 'none', 'ticker')")
        result = self.run_code(code, facts)
        self.assertIsInstance(result, ExecutionResult)
        self.assertEqual((result.answer.kind, result.answer.value), ("ticker", "HPG"))

    def test_startup_timeout_is_startup_failure(self):
        result = self.run_code("def solution(): return 1", startup_timeout=0.01)
        self.assertFailure(result, "runtime_error", "StartupFailure")


if __name__ == "__main__":
    unittest.main()

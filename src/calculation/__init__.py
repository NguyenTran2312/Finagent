"""Calculation core: exact units, metric tree, sandbox and gates.

Importing this package loads only the standard library, PyYAML and the light text helpers in
`src.preprocess.cleaner` / `src.indexing.table_semantics`; no model, database or network
dependency.
"""

from src.calculation.expressions import (
    Formula,
    FormulaError,
    Ref,
    evaluate,
    infer_dimension,
    parse_formula,
)
from src.calculation.facts import (
    Fact,
    FactConflict,
    FactKey,
    FactSet,
    FactSource,
    MissingFact,
    PeriodKind,
    Scope,
)
from src.calculation.gates import (
    ALLOWED_CONSTANTS,
    GateIssue,
    GateReport,
    compare_answers,
    question_literals,
    requested_answer_kind,
    requested_unit,
    run_gates,
)
from src.calculation.helpers import (
    Answer,
    AnswerKind,
    ExecutionContext,
    MetricUnavailable,
    Trace,
    TraceStep,
    build_helpers,
)
from src.calculation.metric_tree import (
    DEFAULT_METRICS_DIR,
    TEMPLATES,
    DerivedDef,
    LeafDef,
    LeafRequest,
    MetricTree,
    Template,
    TreeError,
)
from src.calculation.proposals import ProposalResult, validate_proposal
from src.calculation.sandbox import (
    ALLOWED_BUILTINS,
    ExecutionFailure,
    ExecutionResult,
    SyntaxRejected,
    check_code,
    run,
)
from src.calculation.units import (
    MONEY_SCALES,
    CalculationError,
    Dimension,
    DisplayUnit,
    Quantity,
    UnitError,
    display_unit,
    round_display,
    to_display,
)

__all__ = [
    # units (Task 1)
    "Dimension", "UnitError", "CalculationError", "Quantity", "MONEY_SCALES", "DisplayUnit",
    "display_unit", "to_display", "round_display",
    # expressions (Task 2)
    "FormulaError", "Ref", "Formula", "parse_formula", "infer_dimension", "evaluate",
    # metric tree (Task 3)
    "Template", "TEMPLATES", "LeafDef", "DerivedDef", "TreeError", "LeafRequest",
    "DEFAULT_METRICS_DIR", "MetricTree",
    # facts (Task 4)
    "Scope", "PeriodKind", "FactSource", "FactKey", "Fact", "FactConflict", "MissingFact",
    "FactSet",
    # helpers (Task 6)
    "AnswerKind", "Answer", "ExecutionContext", "TraceStep", "Trace", "MetricUnavailable",
    "build_helpers",
    # proposals (Task 7)
    "ProposalResult", "validate_proposal",
    # sandbox (Task 8)
    "SyntaxRejected", "check_code", "ExecutionResult", "ExecutionFailure", "ALLOWED_BUILTINS",
    "run",
    # gates (Task 9)
    "ALLOWED_CONSTANTS", "GateIssue", "GateReport", "question_literals",
    "requested_answer_kind", "requested_unit", "compare_answers", "run_gates",
]

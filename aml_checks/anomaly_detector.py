"""Statement-integrity checks for monthly financial submissions.

Flags patterns that look engineered rather than reported: unit costs held
to the cent across multiple months, a stated shock (e.g. an FX rate) whose
magnitude doesn't reconcile with the cost movement it's blamed for, revenue
component ratios held suspiciously constant, static contractual figures
(earn-out/covenant targets), and anomalies whose timing lines up with a
cumulative-revenue threshold being crossed.

Each check returns a list of Finding, so results can be filtered, sorted,
or rendered by any caller.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from statistics import pstdev


@dataclass
class Finding:
    severity: str  # "info" | "warning" | "high"
    check: str
    month: str | None
    message: str
    detail: dict = field(default_factory=dict)


def _rel_change(prev: float, curr: float) -> float | None:
    if prev == 0:
        return None
    return (curr - prev) / prev


def detect_precision_clusters(
    rows: list[dict], numerator: str, denominator: str, min_run: int = 3, tolerance: float = 1e-6
) -> list[Finding]:
    """Flag runs of >= min_run consecutive months where numerator/denominator
    is constant to within `tolerance` relative error -- real unit costs drift
    month to month; a value held flat to the cent across a multi-month run
    reads as a formula, not an actual.
    """
    findings = []
    ratios = []
    for r in rows:
        denom = r[denominator]
        ratios.append(r[numerator] / denom if denom else None)

    run_start = 0
    i = 1
    while i <= len(ratios):
        same_as_prev = (
            i < len(ratios)
            and ratios[i] is not None
            and ratios[run_start] is not None
            and abs(ratios[i] - ratios[run_start]) <= tolerance * max(abs(ratios[run_start]), 1e-9)
        )
        if not same_as_prev:
            run_len = i - run_start
            if run_len >= min_run and ratios[run_start] is not None:
                findings.append(
                    Finding(
                        severity="warning",
                        check="precision_cluster",
                        month=rows[run_start]["Month"],
                        message=(
                            f"{numerator}/{denominator} held exactly constant at "
                            f"{ratios[run_start]:,.4f} for {run_len} consecutive months "
                            f"({rows[run_start]['Month']}–{rows[i - 1]['Month']})"
                        ),
                        detail={"value": ratios[run_start], "months": run_len},
                    )
                )
            run_start = i
        i += 1
    return findings


def detect_shock_reconciliation(
    rows: list[dict], shock_col: str, cost_col: str, unit_col: str | None = None, mismatch_tolerance: float = 0.15
) -> list[Finding]:
    """When shock_col steps to a new value, the dependent cost metric's
    step ratio should move by roughly the same multiple. Flag when it doesn't.
    """
    findings = []
    for i in range(1, len(rows)):
        prev_shock, curr_shock = rows[i - 1][shock_col], rows[i][shock_col]
        if prev_shock == curr_shock or prev_shock in (0, None):
            continue
        shock_ratio = curr_shock / prev_shock

        if unit_col:
            prev_cost = rows[i - 1][cost_col] / rows[i - 1][unit_col]
            curr_cost = rows[i][cost_col] / rows[i][unit_col]
        else:
            prev_cost, curr_cost = rows[i - 1][cost_col], rows[i][cost_col]
        if prev_cost == 0:
            continue
        cost_ratio = curr_cost / prev_cost

        mismatch = abs(cost_ratio - shock_ratio) / shock_ratio
        if mismatch > mismatch_tolerance:
            findings.append(
                Finding(
                    severity="high",
                    check="shock_reconciliation",
                    month=rows[i]["Month"],
                    message=(
                        f"{shock_col} moved {shock_ratio:.2f}x at {rows[i]['Month']} but "
                        f"{cost_col}{'/' + unit_col if unit_col else ''} moved {cost_ratio:.2f}x "
                        f"-- the stated shock does not explain the cost change "
                        f"({mismatch:.0%} mismatch)"
                    ),
                    detail={"shock_ratio": shock_ratio, "cost_ratio": cost_ratio},
                )
            )
    return findings


def detect_static_columns(rows: list[dict], exclude: tuple[str, ...] = ("Month",)) -> list[Finding]:
    """Flag columns whose value never changes across the whole submission --
    typically a contractual figure (covenant/earn-out target) rather than an
    operating metric, and worth cross-referencing against timing anomalies.
    """
    findings = []
    if not rows:
        return findings
    for col in rows[0]:
        if col in exclude:
            continue
        values = {r[col] for r in rows}
        if len(values) == 1:
            findings.append(
                Finding(
                    severity="info",
                    check="static_column",
                    month=None,
                    message=f"{col} is identical every month ({values.pop():,}) -- likely a contractual target, not an actual",
                    detail={"column": col},
                )
            )
    return findings


def detect_fixed_ratio(
    rows: list[dict], col_a: str, col_b: str, max_relative_stdev: float = 0.01
) -> list[Finding]:
    """Flag when col_b/col_a is held to a near-constant ratio across every
    month -- a real business's revenue-component mix should drift with deal
    composition, not hold to a fixed split.
    """
    ratios = [r[col_b] / r[col_a] for r in rows if r[col_a]]
    if len(ratios) < 3:
        return []
    mean = sum(ratios) / len(ratios)
    rel_stdev = pstdev(ratios) / mean if mean else 0
    if rel_stdev <= max_relative_stdev:
        return [
            Finding(
                severity="warning",
                check="fixed_ratio",
                month=None,
                message=(
                    f"{col_b}/{col_a} held to a near-constant ratio of {mean:.3f} "
                    f"across all {len(ratios)} months (relative stdev {rel_stdev:.2%})"
                ),
                detail={"mean_ratio": mean, "relative_stdev": rel_stdev},
            )
        ]
    return []


def detect_milestone_crossing_correlation(
    rows: list[dict], revenue_cols: list[str], milestone_col: str, other_findings: list[Finding], window: int = 1
) -> list[Finding]:
    """Find the month cumulative revenue crosses the (typically static)
    milestone target, then check whether any other flagged anomaly lands
    within `window` months of the crossing -- that correlation is the
    strongest single signal of the group.
    """
    findings = []
    milestone = rows[0][milestone_col]
    cumulative = 0.0
    crossing_month = None
    crossing_index = None
    for i, r in enumerate(rows):
        cumulative += sum(r[c] for c in revenue_cols)
        if crossing_month is None and cumulative >= milestone:
            crossing_month = r["Month"]
            crossing_index = i
            break

    if crossing_month is None:
        return findings

    nearby = [
        f
        for f in other_findings
        if f.month is not None
        and abs(_month_index(rows, f.month) - crossing_index) <= window
        and f.check != "static_column"
    ]
    if nearby:
        findings.append(
            Finding(
                severity="high",
                check="milestone_timing_correlation",
                month=crossing_month,
                message=(
                    f"Cumulative {'+'.join(revenue_cols)} crosses the {milestone_col} "
                    f"target ({milestone:,.0f}) at {crossing_month}, within {window} month(s) of "
                    f"{len(nearby)} other flagged anomal{'y' if len(nearby) == 1 else 'ies'} "
                    f"({', '.join(sorted({f.check for f in nearby}))}) -- "
                    "the cost/margin break coincides with the contractual trigger, not an "
                    "independent operating event"
                ),
                detail={"crossing_month": crossing_month, "correlated_checks": [f.check for f in nearby]},
            )
        )
    return findings


def _month_index(rows: list[dict], month: str) -> int:
    for i, r in enumerate(rows):
        if r["Month"] == month:
            return i
    return -1


def run_all_checks(rows: list[dict], config: dict) -> list[Finding]:
    findings: list[Finding] = []

    for unit_cost_check in config.get("precision_clusters", []):
        findings += detect_precision_clusters(rows, **unit_cost_check)

    for reconciliation in config.get("shock_reconciliations", []):
        findings += detect_shock_reconciliation(rows, **reconciliation)

    for ratio_check in config.get("fixed_ratios", []):
        findings += detect_fixed_ratio(rows, **ratio_check)

    static_findings = detect_static_columns(rows, exclude=config.get("exclude_static", ("Month",)))
    findings += static_findings

    if "milestone" in config:
        findings += detect_milestone_crossing_correlation(
            rows,
            revenue_cols=config["milestone"]["revenue_cols"],
            milestone_col=config["milestone"]["milestone_col"],
            other_findings=findings,
            window=config["milestone"].get("window", 1),
        )

    return findings

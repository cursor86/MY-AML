"""Inline (per-row) adversarial checks, for use inside an agent's step loop --
as opposed to anomaly_detector.py's batch/offline review of a whole CSV.

Both share the same two primitives (zero-variance unit costs, a margin
collapse synchronized with a contractual milestone) so an online agent and
an offline audit reach the same verdict on the same data.
"""
from __future__ import annotations


def zero_variance_penalty(
    unit_costs: list[float], min_regime_len: int = 3, tolerance: float = 0.0
) -> str | None:
    """Flag a run of >= min_regime_len consecutive months where unit cost is
    unadjusted (variance exactly 0, or within `tolerance`) despite volume
    changing underneath it. Real unit economics drift; a flat run is a
    formula, not an actual.
    """
    if len(unit_costs) < min_regime_len:
        return None
    run_len = 1
    for i in range(1, len(unit_costs)):
        if abs(unit_costs[i] - unit_costs[i - 1]) <= tolerance:
            run_len += 1
            if run_len >= min_regime_len:
                return (
                    f"unit cost held to {unit_costs[i]:,.2f} with 0% variance "
                    f"across {run_len} consecutive periods ending at index {i}"
                )
        else:
            run_len = 1
    return None


def detect_contractual_fraud(cumulative_rev, milestone, margin_delta, month, on_trigger=None):
    """Flags a margin collapse landing in the same period cumulative revenue
    crosses a static contractual milestone (earn-out, covenant threshold).
    on_trigger, if given, is called with `month` before returning -- the hook
    a caller uses to actually halt the pipeline and page compliance.
    """
    if cumulative_rev >= milestone and margin_delta <= -0.15:
        if on_trigger:
            on_trigger(month)
        return "WARNING: High probability of engineered cost step or contract manipulation."
    return None

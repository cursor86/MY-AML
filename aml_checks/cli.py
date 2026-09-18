"""Run statement-integrity checks against a monthly financial CSV.

Usage:
    python -m aml_checks.cli path/to/submission.csv
    python -m aml_checks.cli path/to/submission.csv --profile bundled_contracts
"""
from __future__ import annotations

import argparse
import csv
import sys

from .anomaly_detector import run_all_checks

# Column-mapping profiles for submission formats seen so far. A new monthly
# format just needs a new profile here -- the checks themselves are generic.
PROFILES = {
    "bundled_contracts": {
        "precision_clusters": [
            {"numerator": "Reported_COGS", "denominator": "Hardware_Units_Shipped"},
        ],
        "shock_reconciliations": [
            {
                "shock_col": "Historical_FX_Hedge_Rate",
                "cost_col": "Reported_COGS",
                "unit_col": "Hardware_Units_Shipped",
            },
        ],
        "fixed_ratios": [
            {"col_a": "Raw_Bundled_Contracts", "col_b": "Consulting_Component"},
        ],
        "exclude_static": ("Month", "Bank_Cash_Balance"),
        "milestone": {
            "revenue_cols": ["Raw_Bundled_Contracts", "Consulting_Component"],
            "milestone_col": "Target_Earn_Out_Milestone",
            "window": 1,
        },
    },
    "ar_actuals": {
        "precision_clusters": [
            {"numerator": "Raw_COGS", "denominator": "Gross_Revenue"},
        ],
        "shock_reconciliations": [],
        "fixed_ratios": [],
        "exclude_static": ("Month", "Starting_Cash_Runway"),
        "milestone": {
            "revenue_cols": ["Gross_Revenue"],
            "milestone_col": "Total_Bank_Debt",
            "window": 1,
        },
    },
}

SEVERITY_ORDER = {"high": 0, "warning": 1, "info": 2}


def load_rows(path: str) -> list[dict]:
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        rows = []
        for raw in reader:
            row = {}
            for key, value in raw.items():
                if key == "Month":
                    row[key] = value
                    continue
                try:
                    row[key] = float(value)
                except (TypeError, ValueError):
                    row[key] = value
            rows.append(row)
        return rows


def guess_profile(rows: list[dict]) -> str:
    columns = set(rows[0].keys()) if rows else set()
    best, best_score = None, -1
    for name, config in PROFILES.items():
        referenced = set()
        for group in ("precision_clusters", "shock_reconciliations", "fixed_ratios"):
            for entry in config.get(group, []):
                referenced.update(v for k, v in entry.items() if k != "min_run" and k != "mismatch_tolerance")
        score = len(referenced & columns)
        if score > best_score:
            best, best_score = name, score
    return best


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path")
    parser.add_argument("--profile", choices=sorted(PROFILES), default=None)
    args = parser.parse_args(argv)

    rows = load_rows(args.csv_path)
    if not rows:
        print("No rows found.")
        return 1

    profile = args.profile or guess_profile(rows)
    print(f"Using profile: {profile}\n")
    findings = run_all_checks(rows, PROFILES[profile])
    findings.sort(key=lambda f: (SEVERITY_ORDER[f.severity], f.month or ""))

    if not findings:
        print("No anomalies flagged.")
        return 0

    for f in findings:
        tag = f.severity.upper()
        where = f" [{f.month}]" if f.month else ""
        print(f"[{tag}]{where} ({f.check}) {f.message}")

    high = sum(1 for f in findings if f.severity == "high")
    print(f"\n{len(findings)} finding(s): {high} high severity.")
    return 1 if high else 0


if __name__ == "__main__":
    sys.exit(main())

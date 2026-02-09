# bias_analysis/cli.py
from __future__ import annotations

import argparse
import json
from typing import Any, Dict, List, Optional

import pandas as pd

from .io import read_table
from .cohort import CohortSpec, build_cohort_table
from .contingency import build_2x2
from .metrics import compute_bias_metrics


def _parse_filters_json(s: Optional[str]) -> List[Dict[str, Any]]:
    """
    filters_json format (list of dicts):
      [
        {"col":"ethnicity","op":"in","value":["Black","White"]}
      ]
    """
    if not s:
        return []

    if not isinstance(s, str):
        raise ValueError("--filters-json must be a JSON string. Prefer --filters-file filters.json")

    # Strip whitespace and optional UTF-8 BOM
    s = s.strip().lstrip("\ufeff")

    try:
        obj = json.loads(s)
    except json.JSONDecodeError as e:
        raise ValueError(
            "Could not parse filters as JSON. "
            f"Tip: use --filters-file filters.json. Details: {e}"
        ) from e

    if not isinstance(obj, list):
        raise ValueError("Filters must be a JSON list of objects.")
    return obj


def _load_filters(filters_json: Optional[str], filters_file: Optional[str]) -> List[Dict[str, Any]]:
    if filters_file:
        with open(filters_file, "r", encoding="utf-8-sig") as f:
            txt = f.read()
        return _parse_filters_json(txt)
    return _parse_filters_json(filters_json)


def apply_filters(df: pd.DataFrame, filters: List[Dict[str, Any]]) -> pd.DataFrame:
    out = df.copy()
    for f in filters:
        col = f["col"]
        op = f.get("op", "in")
        val = f.get("value")

        if col not in out.columns:
            raise KeyError(f"Filter column not found: {col}")

        if op == "in":
            if not isinstance(val, list):
                raise ValueError(f"Filter op 'in' requires list value for {col}")
            out = out[out[col].isin(val)]
        elif op == "eq":
            out = out[out[col] == val]
        elif op == "neq":
            out = out[out[col] != val]
        else:
            raise ValueError(f"Unsupported filter op: {op}")
    return out


def outcome_from_spec(
    df: pd.DataFrame,
    *,
    col: str,
    positive: Optional[str],
    thr: Optional[float],
    op: Optional[str],
) -> pd.Series:
    """
    Builds a 0/1 outcome series from:
      - categorical match: df[col] == positive
      - numeric threshold: df[col] (numeric) compared to thr with op
    """
    if col not in df.columns:
        raise KeyError(f"Outcome column not found in demographics: {col}")

    s = df[col]

    # Categorical mode
    if positive is not None:
        return (s.astype(str) == str(positive)).astype(int)

    # Numeric threshold mode
    if thr is None or op is None:
        raise ValueError("Provide either --outcome-positive OR (--outcome-threshold and --threshold-op).")

    x = pd.to_numeric(s, errors="coerce")

    if op == "ge":
        y = x >= thr
    elif op == "gt":
        y = x > thr
    elif op == "le":
        y = x <= thr
    elif op == "lt":
        y = x < thr
    elif op == "eq":
        y = x == thr
    elif op == "ne":
        y = x != thr
    else:
        raise ValueError(f"Unsupported threshold-op: {op}")

    return y.astype(int)


def _contingency_to_dict(t: Any) -> Dict[str, Any]:
    if hasattr(t, "to_dict") and callable(getattr(t, "to_dict")):
        return t.to_dict()

    d: Dict[str, Any] = {}
    for k in ["a", "b", "c", "d", "A", "B", "C", "D"]:
        if hasattr(t, k):
            d[k.lower()] = getattr(t, k)
    return d or {"repr": repr(t)}


def _load_covariates_spec(args: argparse.Namespace) -> tuple[List[str], List[str], List[str]]:
    """
    Resolve covariates from:
      - --covariates col1 col2 ...
      - --covariates-file covariates.json

    covariates.json format:
      {
        "covariates": ["age", "county"],
        "force_numeric": ["age"],
        "force_categorical": ["county"]
      }
    """
    covariates: List[str] = []
    force_numeric: List[str] = []
    force_categorical: List[str] = []

    if getattr(args, "covariates", None):
        covariates.extend(args.covariates)

    if getattr(args, "covariates_file", None):
        with open(args.covariates_file, "r", encoding="utf-8-sig") as f:
            spec = json.load(f)
        covariates.extend(spec.get("covariates", []))
        force_numeric.extend(spec.get("force_numeric", []))
        force_categorical.extend(spec.get("force_categorical", []))

    # de-dupe, preserve order, drop empties
    covariates = list(dict.fromkeys([c for c in covariates if c]))

    return covariates, force_numeric, force_categorical


def main() -> None:
    ap = argparse.ArgumentParser("bias-analysis")

    ap.add_argument("--demographics", required=True)

    # Optional: commitments inputs 
    ap.add_argument("--current", default=None, help="Optional: current_commitments.csv/xlsx")
    ap.add_argument("--prior", default=None, help="Optional: prior_commitments.csv/xlsx")

    # Optional: restrict to a subset of CDC IDs 
    ap.add_argument("--cdc-ids", nargs="*", default=None, help="Optional: restrict to these CDC IDs.")

    ap.add_argument("--id-col", default="cdcno")
    ap.add_argument("--group-col", default="ethnicity")
    ap.add_argument("--exposed", required=True)
    ap.add_argument("--unexposed", required=True)

    # Outcome specification 
    ap.add_argument("--outcome-col", required=True)
    ap.add_argument("--outcome-positive", default=None, help="Categorical: outcome=1 if outcome-col equals this value.")
    ap.add_argument("--outcome-threshold", type=float, default=None, help="Numeric: threshold value.")
    ap.add_argument(
        "--threshold-op",
        choices=["ge", "gt", "le", "lt", "eq", "ne"],
        default=None,
        help="Numeric compare op for threshold outcomes (ge, gt, le, lt, eq, ne).",
    )

    # filters: use file to avoid PowerShell escaping pain
    ap.add_argument("--filters-file", default=None, help="Path to filters.json (recommended).")
    ap.add_argument("--filters-json", default=None, help="JSON string (harder in PowerShell).")

    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--min-cases", type=int, default=15)

    # No silent imputation: only if explicitly provided (requires updated metrics.py)
    ap.add_argument(
        "--continuity-correction",
        type=float,
        default=None,
        help="Optional continuity correction (e.g., 0.5). If omitted, zero cells yield NaN metrics.",
    )

    ap.add_argument(
        "--chi2-yates",
        dest="chi2_yates",
        action="store_true",
        help="Use Yates continuity correction for chi-square (2x2). Default: on.",
    )
    ap.add_argument(
        "--no-chi2-yates",
        dest="chi2_yates",
        action="store_false",
        help="Disable Yates correction for chi-square.",
    )
    ap.set_defaults(chi2_yates=True)

    # NEW: Logistic regression mode
    ap.add_argument(
        "--mode",
        choices=["2x2", "logit"],
        default="2x2",
        help="Analysis mode: 2x2 (default) or logit",
    )

    ap.add_argument(
        "--covariates",
        nargs="*",
        default=None,
        help="Covariate column names for logistic regression",
    )

    ap.add_argument(
        "--covariates-file",
        default=None,
        help="JSON file specifying covariates and encoding rules",
    )

    ap.add_argument(
        "--drop-missing",
        choices=["any", "outcome", "covariates", "none"],
        default="any",
        help="How to drop rows with missing values (logit mode)",
    )

    ap.add_argument(
        "--no-intercept",
        action="store_true",
        help="Disable intercept in logit model (default: intercept ON).",
    )

    args = ap.parse_args()

    # Resolve covariates early (needed for keep_cols)
    covariates, force_numeric, force_categorical = _load_covariates_spec(args)

    # Load inputs 
    demo = read_table(args.demographics)

    # Load commitments if provided (even if unused now)
    cur = read_table(args.current) if args.current else None
    pri = read_table(args.prior) if args.prior else None

    # Load/apply filters (app Step 2)
    filters = _load_filters(args.filters_json, args.filters_file)
    demo = apply_filters(demo, filters)

    if len(demo) < args.min_cases:
        raise ValueError(f"Filtered dataset has {len(demo)} rows (< {args.min_cases}).")

    #  Build cohort table 
    spec = CohortSpec(id_col=args.id_col, group_col=args.group_col, outcome_col="outcome")

    cohort = build_cohort_table(
        cdc_ids=args.cdc_ids if args.cdc_ids else None,
        demographics=demo,
        current_commitments=cur,
        prior_commitments=pri,
        spec=spec,
        outcome_fn=lambda dsub, c, p: outcome_from_spec(
            dsub,
            col=args.outcome_col,
            positive=args.outcome_positive,
            thr=args.outcome_threshold,
            op=args.threshold_op,
        ),
        keep_cols=list(dict.fromkeys([args.outcome_col] + covariates)),
    )

    # Restrict to the two groups (matches app behavior after selecting groups)
    cohort = cohort[cohort[spec.group_col].isin([args.exposed, args.unexposed])].copy()

    # Sanity checks
    if cohort.empty:
        available = sorted(set(demo[args.group_col].dropna().astype(str).unique()))
        raise ValueError(
            f"No rows left after selecting groups [{args.exposed}, {args.unexposed}]. "
            f"Available values in '{args.group_col}': {available[:30]}{'...' if len(available) > 30 else ''}"
        )

    groups_present = set(cohort[spec.group_col].unique())
    missing = [g for g in [args.exposed, args.unexposed] if g not in groups_present]
    if missing:
        raise ValueError(f"Missing group(s) in filtered cohort: {missing}. Present: {sorted(groups_present)}")

    # MODE SWITCH (after cohort)
    
    if args.mode == "2x2":
        t = build_2x2(
            cohort,
            group_col=spec.group_col,
            outcome_col=spec.outcome_col,
            exposed_value=args.exposed,
            unexposed_value=args.unexposed,
        )

        t_dict = _contingency_to_dict(t)

        metrics = compute_bias_metrics(
            t,
            alpha=args.alpha,
            continuity_correction=args.continuity_correction,
            chi2_yates=args.chi2_yates,
        )

        out = {
            "inputs": {
                "mode": args.mode,
                "id_col": args.id_col,
                "group_col": args.group_col,
                "exposed": args.exposed,
                "unexposed": args.unexposed,
                "outcome_col": args.outcome_col,
                "outcome_positive": args.outcome_positive,
                "outcome_threshold": args.outcome_threshold,
                "threshold_op": args.threshold_op,
                "n_filtered_rows": int(len(demo)),
                "n_used_rows": int(len(cohort)),
                "min_cases": int(args.min_cases),
                "filters": filters,
                "cdc_ids_restricted": bool(args.cdc_ids),
                "current_loaded": bool(args.current),
                "prior_loaded": bool(args.prior),
                "continuity_correction": args.continuity_correction,
                "chi2_yates": bool(args.chi2_yates),
            },
            "table": t_dict,
            "metrics": metrics,
        }

    else:
        from .logistic import fit_logit

        cohort = cohort.copy()
        cohort["group"] = (cohort[spec.group_col].astype(str) == str(args.exposed)).astype(int)

        logit_out = fit_logit(
            df=cohort,
            outcome_col=spec.outcome_col,  # derived 0/1 "outcome"
            group_indicator_col="group",
            covariates=covariates,
            drop_missing=args.drop_missing,
            add_intercept=(not args.no_intercept),
            force_numeric=force_numeric or None,
            force_categorical=force_categorical or None,
            robust_se=True,
        )

        out = {
            "inputs": {
                "mode": args.mode,
                "id_col": args.id_col,
                "group_col": args.group_col,
                "exposed": args.exposed,
                "unexposed": args.unexposed,
                "outcome_col": args.outcome_col,
                "outcome_positive": args.outcome_positive,
                "outcome_threshold": args.outcome_threshold,
                "threshold_op": args.threshold_op,
                "n_filtered_rows": int(len(demo)),
                "n_used_rows": int(len(cohort)),
                "min_cases": int(args.min_cases),
                "filters": filters,
                "cdc_ids_restricted": bool(args.cdc_ids),
                "current_loaded": bool(args.current),
                "prior_loaded": bool(args.prior),
                "covariates": covariates,
                "covariates_file": args.covariates_file,
                "drop_missing": args.drop_missing,
                "intercept": (not args.no_intercept),
                "force_numeric": force_numeric,
                "force_categorical": force_categorical,
            },
            "logit": logit_out,
        }

    print(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    main()

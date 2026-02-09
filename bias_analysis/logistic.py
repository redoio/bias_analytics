# bias_analysis/logistic.py
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import statsmodels.api as sm
from statsmodels.tools.sm_exceptions import PerfectSeparationError


def _infer_covariate_types(
    df: pd.DataFrame,
    covariates: Sequence[str],
    force_categorical: Optional[Sequence[str]] = None,
    force_numeric: Optional[Sequence[str]] = None,
) -> Tuple[List[str], List[str]]:
    force_categorical = set(force_categorical or [])
    force_numeric = set(force_numeric or [])

    numeric: List[str] = []
    categorical: List[str] = []

    for c in covariates:
        if c in force_numeric and c in force_categorical:
            raise ValueError(f"Covariate '{c}' is in both force_numeric and force_categorical.")
        if c in force_numeric:
            numeric.append(c)
            continue
        if c in force_categorical:
            categorical.append(c)
            continue

        s = df[c]
        if pd.api.types.is_numeric_dtype(s):
            numeric.append(c)
            continue

        # try coercion: if >=90% of non-null become numeric, treat as numeric
        coerced = pd.to_numeric(s, errors="coerce")
        non_null = s.notna().sum()
        ok = coerced.notna().sum()
        if non_null > 0 and (ok / non_null) >= 0.90:
            numeric.append(c)
        else:
            categorical.append(c)

    return numeric, categorical


def build_design_matrix(
    df: pd.DataFrame,
    outcome_col: str,
    group_indicator_col: str,
    covariates: Sequence[str],
    drop_missing: str = "any",
    add_intercept: bool = True,
    force_categorical: Optional[Sequence[str]] = None,
    force_numeric: Optional[Sequence[str]] = None,
) -> Tuple[pd.Series, pd.DataFrame, Dict[str, Any]]:
    if outcome_col not in df.columns:
        raise ValueError(f"Missing outcome column: {outcome_col}")
    if group_indicator_col not in df.columns:
        raise ValueError(f"Missing group indicator column: {group_indicator_col}")

    covariates = list(covariates)
    for c in covariates:
        if c not in df.columns:
            raise ValueError(f"Missing covariate column: {c}")

    base_cols = [outcome_col, group_indicator_col] + covariates
    work = df[base_cols].copy()

    # Outcome and group must be numeric 0/1
    work[outcome_col] = pd.to_numeric(work[outcome_col], errors="coerce")
    work[group_indicator_col] = pd.to_numeric(work[group_indicator_col], errors="coerce")

    numeric_covs, cat_covs = _infer_covariate_types(
        work, covariates, force_categorical=force_categorical, force_numeric=force_numeric
    )

    for c in numeric_covs:
        work[c] = pd.to_numeric(work[c], errors="coerce")

    if cat_covs:
        dummies = pd.get_dummies(
            work[cat_covs].astype("string"),
            drop_first=True,
            dummy_na=False,
        )
    else:
        dummies = pd.DataFrame(index=work.index)

    X_parts = [work[[group_indicator_col]]]
    if numeric_covs:
        X_parts.append(work[numeric_covs])
    if not dummies.empty:
        X_parts.append(dummies)

    X = pd.concat(X_parts, axis=1)

    if add_intercept:
        X = sm.add_constant(X, has_constant="add")

    # Ensure design matrix is numeric
    X = X.apply(pd.to_numeric, errors="coerce")

    before_n = len(work)

    if drop_missing not in {"any", "outcome", "covariates", "none"}:
        raise ValueError(f"Invalid drop_missing: {drop_missing}")

    if drop_missing != "none":
        if drop_missing == "any":
            mask = ~(work[outcome_col].isna() | X.isna().any(axis=1))
        elif drop_missing == "outcome":
            mask = ~work[outcome_col].isna()
        else:  # covariates
            mask = ~X.isna().any(axis=1)

        y = work.loc[mask, outcome_col]
        X = X.loc[mask, :]
    else:
        y = work[outcome_col]

    meta = {
        "n_before": int(before_n),
        "n_used": int(len(y)),
        "n_dropped": int(before_n - len(y)),
        "numeric_covariates": numeric_covs,
        "categorical_covariates": cat_covs,
        "terms": list(X.columns),
        "add_intercept": bool(add_intercept),
        "drop_missing": drop_missing,
    }
    return y, X, meta


def _apply_robust_covariance(res: Any, model: Any, cov_type: str = "HC1") -> Any:
    """
    Apply robust covariance in a way that works across statsmodels versions.

    Some versions:
      - expose get_robustcov_results (returns results)
      - expose _get_robustcov_results (may return results OR None and mutate in-place)
      - allow model.fit(..., cov_type="HC1")
    """
    # Preferred public API (if present)
    if hasattr(res, "get_robustcov_results"):
        out = res.get_robustcov_results(cov_type=cov_type)
        return out if out is not None else res

    # Private fallback (may mutate in-place and return None)
    if hasattr(res, "_get_robustcov_results"):
        out = res._get_robustcov_results(cov_type=cov_type)
        return out if out is not None else res

    # Last resort: refit with cov_type
    try:
        out = model.fit(disp=False, cov_type=cov_type)
        return out if out is not None else res
    except TypeError:
        return res


def fit_logit(
    df: pd.DataFrame,
    outcome_col: str,
    group_indicator_col: str,
    covariates: Sequence[str],
    drop_missing: str = "any",
    add_intercept: bool = True,
    force_categorical: Optional[Sequence[str]] = None,
    force_numeric: Optional[Sequence[str]] = None,
    robust_se: bool = True,
) -> Dict[str, Any]:
    y, X, meta = build_design_matrix(
        df=df,
        outcome_col=outcome_col,
        group_indicator_col=group_indicator_col,
        covariates=covariates,
        drop_missing=drop_missing,
        add_intercept=add_intercept,
        force_categorical=force_categorical,
        force_numeric=force_numeric,
    )

    # keep as float for statsmodels
    y = y.astype(float)
    X = X.astype(float)

    model = sm.Logit(y, X)

    used_regularized = False
    try:
        res = model.fit(disp=False)

        if robust_se:
            res = _apply_robust_covariance(res=res, model=model, cov_type="HC1")

        params = res.params
        pvals = res.pvalues
        conf = res.conf_int()
        conf.columns = ["ci_low", "ci_high"]

    except PerfectSeparationError:
        used_regularized = True
        res = model.fit_regularized(method="l1", alpha=1e-6, disp=False)
        params = res.params
        pvals = pd.Series(np.nan, index=params.index)
        conf = pd.DataFrame({"ci_low": np.nan, "ci_high": np.nan}, index=params.index)

    rows: List[Dict[str, Any]] = []
    for term in params.index:
        coef = float(params.loc[term])
        or_value = float(np.exp(coef))

        ci_low_v = conf.loc[term, "ci_low"]
        ci_high_v = conf.loc[term, "ci_high"]
        ci_low = float(np.exp(ci_low_v)) if pd.notna(ci_low_v) else float("nan")
        ci_high = float(np.exp(ci_high_v)) if pd.notna(ci_high_v) else float("nan")

        pv = pvals.loc[term]
        p_value = float(pv) if pd.notna(pv) else float("nan")

        rows.append(
            {
                "term": str(term),
                "coef": coef,
                "or": or_value,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "p_value": p_value,
            }
        )

    return {"meta": meta, "used_regularized": used_regularized, "results": rows}

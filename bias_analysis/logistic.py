from __future__ import annotations
from dataclasses import dataclass
import pandas as pd
import statsmodels.api as sm

@dataclass
class LogisticResult:
    params: dict
    conf_int: dict
    pvalues: dict

def fit_logistic(cohort: pd.DataFrame, *, outcome_col: str, predictors: list[str]) -> LogisticResult:
    """
    Simple logistic regression scaffold.
    cohort[outcome_col] should be 0/1; predictors can include group encoding + covariates.
    """
    df = cohort[[outcome_col, *predictors]].dropna().copy()
    y = df[outcome_col].astype(float)
    X = df[predictors]
    X = sm.add_constant(X, has_constant="add")
    model = sm.Logit(y, X).fit(disp=False)

    ci = model.conf_int()
    return LogisticResult(
        params=model.params.to_dict(),
        conf_int={k: (float(ci.loc[k, 0]), float(ci.loc[k, 1])) for k in ci.index},
        pvalues=model.pvalues.to_dict(),
    )

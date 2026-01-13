# bias_analysis/cohort.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Iterable, Optional
import pandas as pd

@dataclass
class CohortSpec:
    id_col: str = "cdc_id"
    group_col: str = "race_eth"
    outcome_col: str = "outcome"

def build_cohort_table(
    cdc_ids: Optional[Iterable[str]],
    demographics: pd.DataFrame,
    current_commitments: Optional[pd.DataFrame],
    prior_commitments: Optional[pd.DataFrame],
    *,
    spec: CohortSpec,
    outcome_fn: Callable[[pd.DataFrame, Optional[pd.DataFrame], Optional[pd.DataFrame]], pd.Series],
    keep_cols: Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    Returns a cohort table with columns:
      - id_col
      - group_col
      - outcome_col
    If cdc_ids is provided, restricts to those IDs; otherwise uses all rows.
    """

    demo = demographics.copy()

    if spec.id_col not in demo.columns:
        raise KeyError(f"ID column '{spec.id_col}' not found in demographics.")

    demo[spec.id_col] = demo[spec.id_col].astype(str)

    if cdc_ids:
        ids = pd.Series(list(cdc_ids), dtype="string").dropna().astype(str).unique().tolist()
        demo_sub = demo[demo[spec.id_col].isin(ids)].copy()
    else:
        demo_sub = demo.copy()

    if spec.group_col not in demo_sub.columns:
        raise KeyError(f"Group column '{spec.group_col}' not found in demographics.")

    outcome = outcome_fn(demo_sub, current_commitments, prior_commitments)

    out = demo_sub[[spec.id_col, spec.group_col]].copy()
    # Align outcome to rows
    out[spec.outcome_col] = outcome.values if len(outcome) == len(out) else outcome.reindex(out.index).values
    out[spec.outcome_col] = out[spec.outcome_col].astype(float)

    if keep_cols:
        for c in keep_cols:
            if c in demo_sub.columns:
                out[c] = demo_sub[c].values

    return out

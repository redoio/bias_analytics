from .io import read_table
from .cohort import build_cohort_table
from .contingency import build_2x2
from .metrics import (
    odds_ratio_and_ci,
    rate_ratio_and_ci,
    relative_risk_and_ci,
    compute_bias_metrics,
)

__all__ = [
    "read_table",
    "build_cohort_table",
    "build_2x2",
    "odds_ratio_and_ci",
    "rate_ratio_and_ci",
    "relative_risk_and_ci",
    "compute_bias_metrics",
]

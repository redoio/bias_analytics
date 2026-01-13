from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd

@dataclass(frozen=True)
class Contingency2x2:
    # A=exposed group, B=unexposed group; outcome=1 is event
    a: int  # exposed & event
    b: int  # exposed & no-event
    c: int  # unexposed & event
    d: int  # unexposed & no-event

    def as_array(self) -> np.ndarray:
        return np.array([[self.a, self.b], [self.c, self.d]], dtype=int)

def build_2x2(
    cohort: pd.DataFrame,
    *,
    group_col: str,
    outcome_col: str,
    exposed_value: str,
    unexposed_value: str,
    outcome_positive_value: float = 1.0,
) -> Contingency2x2:
    """
    Builds a 2x2 table:
        outcome=1   outcome=0
    exp     a         b
    unexp   c         d
    """
    df = cohort[[group_col, outcome_col]].copy()
    df = df.dropna(subset=[group_col, outcome_col])

    exp = df[df[group_col].astype(str) == str(exposed_value)]
    unexp = df[df[group_col].astype(str) == str(unexposed_value)]

    def _counts(x: pd.DataFrame) -> tuple[int, int]:
        ev = (x[outcome_col] == outcome_positive_value).sum()
        nev = (x[outcome_col] != outcome_positive_value).sum()
        return int(ev), int(nev)

    a, b = _counts(exp)
    c, d = _counts(unexp)

    return Contingency2x2(a=a, b=b, c=c, d=d)

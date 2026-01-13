from __future__ import annotations
from pathlib import Path
import pandas as pd

def read_table(path: str) -> pd.DataFrame:
    p = Path(path)
    suf = p.suffix.lower()
    if suf in [".csv"]:
        return pd.read_csv(p)
    if suf in [".xlsx", ".xls"]:
        return pd.read_excel(p)
    raise ValueError(f"Unsupported file type: {suf}. Expected .csv or .xlsx")

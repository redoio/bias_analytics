import numpy as np
import pandas as pd

from bias_analysis.logistic import build_design_matrix, fit_logit


def test_build_design_matrix_onehot_and_numeric():
    df = pd.DataFrame(
        {
            "outcome": [0, 1, 0, 1, 0, 1],
            "group":   [0, 0, 1, 1, 0, 1],
            "age":     [30, 40, 35, 50, 29, 60],
            "county":  ["A", "A", "B", "B", "C", "C"],
        }
    )

    y, X, meta = build_design_matrix(
        df=df,
        outcome_col="outcome",
        group_indicator_col="group",
        covariates=["age", "county"],
        drop_missing="any",
        add_intercept=True,
    )

    # outcome used
    assert len(y) == 6
    # intercept present
    assert "const" in X.columns
    # group present
    assert "group" in X.columns
    # numeric kept
    assert "age" in X.columns
    # one-hot present for county (drop_first=True => A dropped, B/C created)
    assert any(c.startswith("county_") for c in X.columns)
    assert set(meta["numeric_covariates"]) == {"age"}
    assert set(meta["categorical_covariates"]) == {"county"}


def test_fit_logit_group_effect_positive():
    rng = np.random.default_rng(0)
    n = 2000

    group = rng.integers(0, 2, size=n)
    age = rng.normal(40, 10, size=n)
    county = rng.choice(["A", "B", "C"], size=n, p=[0.4, 0.3, 0.3])

    # True DGP: group increases log-odds
    lin = -2.0 + 0.7 * group + 0.02 * age + 0.3 * (county == "B")
    p = 1 / (1 + np.exp(-lin))
    y = rng.binomial(1, p, size=n)

    df = pd.DataFrame({"outcome": y, "group": group, "age": age, "county": county})

    out = fit_logit(
        df=df,
        outcome_col="outcome",
        group_indicator_col="group",
        covariates=["age", "county"],
        drop_missing="any",
        add_intercept=True,
        robust_se=True,
    )

    res = {r["term"]: r for r in out["results"]}
    assert "group" in res
    assert res["group"]["coef"] > 0
    assert res["group"]["or"] > 1


def test_drop_missing_any_drops_rows():
    df = pd.DataFrame(
        {
            "outcome": [0, 1, np.nan, 1],
            "group":   [0, 1, 1, 0],
            "age":     [30, np.nan, 40, 50],
            "county":  ["A", "B", "B", None],
        }
    )

    y, X, meta = build_design_matrix(
        df=df,
        outcome_col="outcome",
        group_indicator_col="group",
        covariates=["age", "county"],
        drop_missing="any",
        add_intercept=True,
    )

    # With drop_missing="any", rows missing outcome OR any X term are removed
    assert meta["n_used"] < meta["n_before"]
    assert len(y) == meta["n_used"]
    assert not np.isnan(y.to_numpy()).any()
    assert not X.isna().any().any()

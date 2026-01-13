import math

from bias_analysis.contingency import Contingency2x2
from bias_analysis.metrics import odds_ratio_and_ci


def test_or_zero_cell_nan_by_default():
    t = Contingency2x2(a=0, b=10, c=5, d=20)
    res = odds_ratio_and_ci(t)

    assert math.isnan(res["odds_ratio"])
    assert math.isnan(res["odds_ratio_ci_low"])
    assert math.isnan(res["odds_ratio_ci_high"])


def test_or_zero_cell_with_cc_finite():
    t = Contingency2x2(a=0, b=10, c=5, d=20)
    res = odds_ratio_and_ci(t, continuity_correction=0.5)

    assert math.isfinite(res["odds_ratio"])
    assert math.isfinite(res["odds_ratio_ci_low"])
    assert math.isfinite(res["odds_ratio_ci_high"])
    assert res["odds_ratio_ci_low"] <= res["odds_ratio"] <= res["odds_ratio_ci_high"]

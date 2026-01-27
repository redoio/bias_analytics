import math

from bias_analysis.contingency import Contingency2x2
from bias_analysis.metrics import odds_ratio_and_ci, chi_square_test


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


def test_chi_square_returns_valid_p_value():
    t = Contingency2x2(a=10, b=20, c=30, d=40)
    res = chi_square_test(t, yates=True)
    assert 0.0 <= res["chi2_p_value"] <= 1.0
    assert res["chi2_dof"] == 1


def test_chi_square_yates_flag_present():
    t = Contingency2x2(a=10, b=20, c=30, d=40)
    res = chi_square_test(t, yates=False)
    assert res["chi2_yates_correction"] is False


def test_chi_square_yates_changes_statistic():
    t = Contingency2x2(a=10, b=20, c=30, d=40)
    res_y = chi_square_test(t, yates=True)
    res_n = chi_square_test(t, yates=False)
    assert res_y["chi2_stat"] != res_n["chi2_stat"]

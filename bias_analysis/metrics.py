from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, Optional
import math

from scipy.stats import norm

from .contingency import Contingency2x2


def _z(alpha: float) -> float:
    return float(norm.ppf(1.0 - alpha / 2.0))


def _apply_cc(
    t: Contingency2x2, cc: Optional[float]
) -> tuple[float, float, float, float]:
    """
    Apply continuity correction ONLY if explicitly provided.
    If any cell is zero and cc is None -> return NaNs to signal undefined ratios.
    """
    a, b, c, d = t.a, t.b, t.c, t.d
    if min(a, b, c, d) == 0:
        if cc is None:
            return (float("nan"), float("nan"), float("nan"), float("nan"))
        return (a + cc, b + cc, c + cc, d + cc)
    return (float(a), float(b), float(c), float(d))


def odds_ratio_and_ci(
    t: Contingency2x2,
    *,
    alpha: float = 0.05,
    continuity_correction: Optional[float] = None,
) -> Dict[str, Any]:
    a, b, c, d = _apply_cc(t, continuity_correction)
    if any(math.isnan(x) for x in (a, b, c, d)) or b == 0 or c == 0:
        return {
            "odds_ratio": float("nan"),
            "odds_ratio_ci_low": float("nan"),
            "odds_ratio_ci_high": float("nan"),
            "alpha": alpha,
            "continuity_correction": continuity_correction,
        }

    or_ = (a * d) / (b * c)
    se = math.sqrt(1.0 / a + 1.0 / b + 1.0 / c + 1.0 / d)
    z = _z(alpha)
    lo = math.exp(math.log(or_) - z * se)
    hi = math.exp(math.log(or_) + z * se)

    return {
        "odds_ratio": float(or_),
        "odds_ratio_ci_low": float(lo),
        "odds_ratio_ci_high": float(hi),
        "alpha": alpha,
        "continuity_correction": continuity_correction,
    }


def relative_risk_and_ci(
    t: Contingency2x2,
    *,
    alpha: float = 0.05,
    continuity_correction: Optional[float] = None,
) -> Dict[str, Any]:
    a, b, c, d = _apply_cc(t, continuity_correction)
    if any(math.isnan(x) for x in (a, b, c, d)) or (a + b) <= 0 or (c + d) <= 0:
        return {
            "relative_risk": float("nan"),
            "relative_risk_ci_low": float("nan"),
            "relative_risk_ci_high": float("nan"),
            "alpha": alpha,
            "continuity_correction": continuity_correction,
        }

    risk_exp = a / (a + b)
    risk_unexp = c / (c + d)
    if risk_unexp <= 0:
        return {
            "relative_risk": float("nan"),
            "relative_risk_ci_low": float("nan"),
            "relative_risk_ci_high": float("nan"),
            "alpha": alpha,
            "continuity_correction": continuity_correction,
        }

    rr = risk_exp / risk_unexp

    # Katz log(RR) SE
    se = math.sqrt((1.0 / a) - (1.0 / (a + b)) + (1.0 / c) - (1.0 / (c + d)))
    z = _z(alpha)
    lo = math.exp(math.log(rr) - z * se)
    hi = math.exp(math.log(rr) + z * se)

    return {
        "relative_risk": float(rr),
        "relative_risk_ci_low": float(lo),
        "relative_risk_ci_high": float(hi),
        "alpha": alpha,
        "continuity_correction": continuity_correction,
    }


def rate_ratio_and_ci(
    *,
    events_exposed: int,
    time_exposed: float,
    events_unexposed: int,
    time_unexposed: float,
    alpha: float = 0.05,
    continuity_correction: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Rate ratio (incidence rate ratio) for Poisson events with person-time.

    RR = (events_exposed / time_exposed) / (events_unexposed / time_unexposed)

    Log CI approx:
      SE(log RR) = sqrt(1/e1 + 1/e0)
    where e1,e0 are event counts (with optional continuity correction if 0).
    """
    if time_exposed <= 0 or time_unexposed <= 0:
        raise ValueError("time_exposed and time_unexposed must be > 0.")
    if events_exposed < 0 or events_unexposed < 0:
        raise ValueError("events must be >= 0.")

    e1 = float(events_exposed)
    e0 = float(events_unexposed)

    if (events_exposed == 0 or events_unexposed == 0):
        if continuity_correction is None:
            return {
                "rate_ratio": float("nan"),
                "rate_ratio_ci_low": float("nan"),
                "rate_ratio_ci_high": float("nan"),
                "alpha": alpha,
                "continuity_correction": continuity_correction,
            }
        e1 += continuity_correction
        e0 += continuity_correction

    r1 = e1 / time_exposed
    r0 = e0 / time_unexposed
    if r0 <= 0:
        return {
            "rate_ratio": float("nan"),
            "rate_ratio_ci_low": float("nan"),
            "rate_ratio_ci_high": float("nan"),
            "alpha": alpha,
            "continuity_correction": continuity_correction,
        }

    rr = r1 / r0
    se = math.sqrt((1.0 / e1) + (1.0 / e0))
    z = _z(alpha)
    lo = math.exp(math.log(rr) - z * se)
    hi = math.exp(math.log(rr) + z * se)

    return {
        "rate_ratio": float(rr),
        "rate_ratio_ci_low": float(lo),
        "rate_ratio_ci_high": float(hi),
        "alpha": alpha,
        "continuity_correction": continuity_correction,
    }


def compute_bias_metrics(
    t: Contingency2x2,
    *,
    alpha: float = 0.05,
    continuity_correction: Optional[float] = None,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {"table": asdict(t)}
    out.update(odds_ratio_and_ci(t, alpha=alpha, continuity_correction=continuity_correction))
    out.update(relative_risk_and_ci(t, alpha=alpha, continuity_correction=continuity_correction))
    # rate ratio requires person-time inputs, so not computed here
    return out

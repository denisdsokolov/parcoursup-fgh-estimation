"""Integer thresholds induced by the paper's minimum target shares.

Processed Parcoursup tables store shares in percent (0..100), whereas alpha
in the manuscript lies in [0, 1]. Only this boundary performs the conversion.
"""

from decimal import Decimal, ROUND_CEILING
from fractions import Fraction
from math import isfinite, isnan


def target_count(share_percent: float | Fraction | Decimal | None, cohort_size: int) -> int:
    """Return ceil(share_percent * cohort_size / 100), without float overshoot.

    A missing share means no target. The cohort may be an attainable stage
    cohort or final enrollment; capacity is used only for its induced count.
    Decimal arithmetic preserves declared decimal percentages at boundaries.
    """
    if cohort_size < 0 or int(cohort_size) != cohort_size:
        raise ValueError("cohort_size must be a nonnegative integer")
    share = 0.0 if share_percent is None else float(share_percent)
    if isnan(share):
        share = 0.0
    if not isfinite(share) or not 0 <= share <= 100:
        raise ValueError("target share must lie in [0, 100] percent")
    if isinstance(share_percent, Fraction):
        numerator = share_percent.numerator * int(cohort_size)
        denominator = share_percent.denominator * 100
        return -(-numerator // denominator)
    declared = share if share_percent is None or isnan(float(share_percent)) else share_percent
    value = Decimal(str(declared)) * int(cohort_size) / Decimal(100)
    return int(value.to_integral_value(rounding=ROUND_CEILING))

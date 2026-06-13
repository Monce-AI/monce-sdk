"""
Distribution candle for a Snake lookalike set.

A candle is a pure-distribution object computed from a list of float targets
(the y values of a sample's lookalikes). No temporal ordering — just the
shape of belief: high/low (extremes), q1/q3 (body / IQR), median (central),
mean and std (point estimate + dispersion), n (sample size).

Usage (driven by Snake.get_candle / get_batch_candles):
    from algorithmeai import Candle, compute_candle
    candle = compute_candle([1.2, 3.4, 2.1, ...])
    candle.median, candle.q1, candle.q3, candle.high, candle.low
"""

from dataclasses import dataclass, asdict


@dataclass
class Candle:
    high: float
    q3: float
    median: float
    q1: float
    low: float
    mean: float
    iqr_mean: float
    std: float
    n: int

    def to_dict(self):
        return asdict(self)


def _percentile(sorted_xs, p):
    if not sorted_xs:
        return float("nan")
    if len(sorted_xs) == 1:
        return float(sorted_xs[0])
    k = (len(sorted_xs) - 1) * p
    lo = int(k)
    hi = lo + 1 if lo + 1 < len(sorted_xs) else lo
    frac = k - lo
    return float(sorted_xs[lo] * (1 - frac) + sorted_xs[hi] * frac)


def compute_candle(ys):
    """Build a Candle from a list of numeric values.

    Coerces entries to float; non-numeric values are skipped. Returns a
    Candle with NaNs and n=0 if the input contains no numeric values."""
    xs = []
    for v in ys:
        try:
            xs.append(float(v))
        except (TypeError, ValueError):
            continue
    n = len(xs)
    if n == 0:
        nan = float("nan")
        return Candle(high=nan, q3=nan, median=nan, q1=nan, low=nan,
                      mean=nan, iqr_mean=nan, std=nan, n=0)
    xs_sorted = sorted(xs)
    mean = sum(xs) / n
    var = sum((x - mean) ** 2 for x in xs) / n
    std = var ** 0.5
    q1 = _percentile(xs_sorted, 0.25)
    q3 = _percentile(xs_sorted, 0.75)
    # IQR-trimmed mean: average over [Q1, Q3]. Robust point estimate for
    # regression — discards the wicks (extreme lookalikes routed by accident).
    iqr_segment = [x for x in xs_sorted if q1 <= x <= q3]
    if not iqr_segment:
        iqr_segment = xs_sorted
    iqr_mean = sum(iqr_segment) / len(iqr_segment)
    return Candle(
        high=float(xs_sorted[-1]),
        q3=q3,
        median=_percentile(xs_sorted, 0.50),
        q1=q1,
        low=float(xs_sorted[0]),
        mean=float(mean),
        iqr_mean=float(iqr_mean),
        std=float(std),
        n=n,
    )

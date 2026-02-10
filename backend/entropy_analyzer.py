"""
Entropy-Based Regime Detection for Dip Finder.

Uses information theory to detect institutional accumulation during dips.

Core Signals:
  1. Shannon Entropy — measures randomness of price action
     - High entropy (~1.0) = random retail noise
     - Low entropy (<0.6) during a dip = structured institutional accumulation
  2. Entropy Rate of Change — speed of entropy shift
     - Sharp drop = sudden institutional interest (urgent)
     - Gradual drop = slow accumulation (still bullish)
  3. Hurst Exponent — mean-reversion probability via R/S analysis
     - H < 0.5 = mean-reverting (dip likely bounces)
     - H ≈ 0.5 = random walk
     - H > 0.5 = trending (dip may continue)

The composite "dip quality" score combines all three into a single
0-100 rating of how likely a dip is to be a genuine buying opportunity
backed by institutional activity.
"""

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ── Dataclass ────────────────────────────────────────────────────────────────

@dataclass
class EntropyAnalysis:
    """Results of entropy-based regime detection."""
    # Shannon Entropy (0-1 normalized, 1 = max randomness)
    entropy: float = 1.0
    entropy_percentile: float = 50.0     # Where current entropy sits vs history

    # Entropy dynamics
    entropy_roc: float = 0.0             # Rate of change (negative = entropy dropping)
    entropy_dropping: bool = False       # True if entropy declining significantly

    # Hurst Exponent
    hurst_exponent: float = 0.5          # 0-1, <0.5 = mean-reverting
    mean_reverting: bool = False         # True if H < 0.45
    trending: bool = False               # True if H > 0.55

    # Regime Classification
    regime: str = "random"               # "accumulation", "distribution", "trending", "random"
    regime_confidence: float = 0.0       # 0-1

    # Composite Score (0-100, higher = better dip quality)
    dip_quality_score: float = 0.0

    def to_dict(self) -> dict:
        return {
            "entropy": round(self.entropy, 4),
            "entropy_percentile": round(self.entropy_percentile, 1),
            "entropy_roc": round(self.entropy_roc, 4),
            "entropy_dropping": self.entropy_dropping,
            "hurst_exponent": round(self.hurst_exponent, 4),
            "mean_reverting": self.mean_reverting,
            "trending": self.trending,
            "regime": self.regime,
            "regime_confidence": round(self.regime_confidence, 2),
            "dip_quality_score": round(self.dip_quality_score, 1),
        }


# ── Shannon Entropy ──────────────────────────────────────────────────────────

def calculate_shannon_entropy(returns: np.ndarray, n_bins: int = 10) -> float:
    """
    Calculate Shannon entropy of a return distribution.

    Discretizes continuous returns into bins, calculates probability
    distribution, then computes H = -Σ p(x) * log2(p(x)).

    Returns normalized entropy (0-1), where:
      1.0 = maximum randomness (uniform distribution)
      0.0 = perfectly predictable (single bin)

    Args:
      returns: Array of log returns
      n_bins: Number of discretization bins (more bins = finer resolution)
    """
    if len(returns) < 5:
        return 1.0  # Not enough data, assume random

    # Discretize returns into bins
    counts, _ = np.histogram(returns, bins=n_bins)

    # Convert to probability distribution
    total = counts.sum()
    if total == 0:
        return 1.0

    probs = counts / total

    # Remove zero-probability bins (log(0) is undefined)
    probs = probs[probs > 0]

    # Shannon entropy: H = -Σ p(x) * log2(p(x))
    entropy = -np.sum(probs * np.log2(probs))

    # Normalize to 0-1 range (max entropy = log2(n_bins))
    max_entropy = np.log2(n_bins)
    if max_entropy > 0:
        entropy = entropy / max_entropy

    return float(entropy)


def calculate_rolling_entropy(
    returns: np.ndarray, window: int = 20, n_bins: int = 10
) -> np.ndarray:
    """
    Calculate rolling Shannon entropy over a window of returns.

    Returns array of entropy values, one per bar (NaN for first `window-1` bars).
    """
    n = len(returns)
    result = np.full(n, np.nan)

    for i in range(window - 1, n):
        window_returns = returns[i - window + 1 : i + 1]
        result[i] = calculate_shannon_entropy(window_returns, n_bins)

    return result


# ── Hurst Exponent (Rescaled Range) ──────────────────────────────────────────

def calculate_hurst_exponent(series: np.ndarray, max_lag: int = 40) -> float:
    """
    Calculate the Hurst exponent using Rescaled Range (R/S) analysis
    on LOG RETURNS (not raw prices) to remove inherent trend bias.

    The Hurst exponent characterizes the long-term memory of a time series:
      H < 0.5 → Anti-persistent (mean-reverting): dips tend to bounce
      H = 0.5 → Random walk: no predictable tendency
      H > 0.5 → Persistent (trending): dips tend to continue

    Args:
      series: Price series (will be converted to log returns internally)
      max_lag: Maximum sub-series length to test

    Returns:
      Hurst exponent (0-1), or 0.5 on failure
    """
    if len(series) < 20:
        return 0.5  # Not enough data

    try:
        series = np.array(series, dtype=float)

        # ── KEY FIX: Convert prices to log returns ──────────────────
        # Raw prices have an inherent trend that biases H toward 1.0.
        # Log returns are stationary and give meaningful H values.
        valid = series[series > 0]
        if len(valid) < 20:
            return 0.5
        log_returns = np.diff(np.log(valid))

        if len(log_returns) < 20 or np.std(log_returns) < 1e-10:
            return 0.5

        # ── Capped max_lag (avoid overshoot) ────────────────────────
        min_lag = 8
        max_lag = min(max_lag, len(log_returns) // 4)  # Cap at 1/4 of data
        if max_lag <= min_lag:
            return 0.5

        # Use logarithmically spaced lag values for better fit
        lags = np.unique(np.logspace(
            np.log10(min_lag), np.log10(max_lag), num=15
        ).astype(int))
        lags = lags[lags >= min_lag]

        rs_values = []
        lag_values = []

        for lag in lags:
            n_subseries = len(log_returns) // lag
            if n_subseries < 1:
                continue

            rs_list = []
            for k in range(n_subseries):
                subseries = log_returns[k * lag : (k + 1) * lag]
                if len(subseries) < 2:
                    continue

                # Mean-adjusted cumulative deviations
                mean = np.mean(subseries)
                deviations = subseries - mean
                cumdev = np.cumsum(deviations)

                # Rescaled range
                r = np.max(cumdev) - np.min(cumdev)
                s = np.std(subseries, ddof=1)

                if s > 1e-10:
                    rs_list.append(r / s)

            if rs_list:
                rs_values.append(np.mean(rs_list))
                lag_values.append(lag)

        if len(rs_values) < 3:
            return 0.5

        # Fit log(R/S) = H * log(n) + c
        log_lags = np.log(np.array(lag_values, dtype=float))
        log_rs = np.log(np.array(rs_values, dtype=float))

        coeffs = np.polyfit(log_lags, log_rs, 1)
        hurst = float(coeffs[0])

        # Clamp to [0, 1]
        return max(0.0, min(1.0, hurst))

    except Exception as e:
        logger.warning(f"Hurst exponent calculation failed: {e}")
        return 0.5


# ── Regime Classification ────────────────────────────────────────────────────

def classify_regime(
    entropy: float,
    entropy_roc: float,
    hurst: float,
    price_dropping: bool,
) -> tuple[str, float]:
    """
    Classify the current market regime for this stock.

    Returns: (regime_name, confidence)

    Regimes:
      "accumulation" — Entropy dropping + price dropping + mean-reverting
                       → Institutions buying the dip (strongest signal)
      "distribution" — Entropy dropping + price rising + trending
                       → Institutions selling into strength
      "trending"     — High Hurst + high entropy
                       → Genuine trend (not institutional)
      "random"       — High entropy + Hurst ≈ 0.5
                       → No detectable pattern
    """
    confidence = 0.0

    # Accumulation: low/dropping entropy + price drop + mean-reverting Hurst
    if entropy < 0.7 and price_dropping and hurst < 0.5:
        confidence = (
            (0.7 - entropy) * 2 +            # Lower entropy = more confident
            max(0, -entropy_roc) * 5 +        # Faster drop = more confident
            (0.5 - hurst) * 2                 # More mean-reverting = more confident
        )
        confidence = min(1.0, confidence)
        return "accumulation", confidence

    # Distribution: structured selling (entropy low + price up + trending)
    if entropy < 0.7 and not price_dropping and hurst > 0.55:
        confidence = (0.7 - entropy) * 1.5
        confidence = min(1.0, confidence)
        return "distribution", confidence

    # Trending: clear trend (Hurst high, entropy can be anything)
    if hurst > 0.6:
        confidence = (hurst - 0.5) * 2
        confidence = min(1.0, confidence)
        return "trending", confidence

    # Random: no detectable institutional pattern
    confidence = entropy * 0.8  # Higher entropy = more confident it's random
    return "random", min(1.0, confidence)


# ── Composite Dip Quality Score ──────────────────────────────────────────────

def calculate_dip_quality(
    entropy: float,
    entropy_roc: float,
    entropy_percentile: float,
    hurst: float,
    regime: str,
    regime_confidence: float,
) -> float:
    """
    Calculate composite dip quality score (0-100).

    Components:
      Entropy Signal (40 pts):  Low entropy during dip = institutional footprint
      Hurst Signal (35 pts):    Mean-reverting = dip likely bounces
      Regime Bonus (25 pts):    "accumulation" regime = strongest confirmation

    Higher = better quality dip (more likely to bounce with institutional backing).
    """
    score = 0.0

    # ── Entropy Signal (40 pts max) ──────────────────────────────────
    # Lower entropy = more structured = more institutional
    if entropy < 0.8:
        entropy_pts = 40 * max(0, (0.8 - entropy) / 0.6)  # 0.2 entropy = max
        score += min(40, entropy_pts)

    # Bonus for rapidly dropping entropy (institutions just started)
    if entropy_roc < -0.02:
        roc_bonus = 10 * min(1.0, abs(entropy_roc) / 0.1)
        score += min(10, roc_bonus)

    # ── Hurst Signal (35 pts max) ────────────────────────────────────
    # Lower Hurst = more mean-reverting = dip more likely to bounce
    if hurst < 0.5:
        hurst_pts = 35 * max(0, (0.5 - hurst) / 0.3)  # 0.2 Hurst = max
        score += min(35, hurst_pts)
    elif hurst > 0.6:
        # Penalty for trending (dip likely continues)
        trend_penalty = 15 * min(1.0, (hurst - 0.6) / 0.2)
        score -= trend_penalty

    # ── Regime Bonus (25 pts max) ────────────────────────────────────
    if regime == "accumulation":
        regime_pts = 25 * regime_confidence
        score += regime_pts
    elif regime == "distribution":
        score -= 15  # Active distribution = bearish

    # Clamp to 0-100
    return max(0, min(100, score))


# ── Main Analyzer ────────────────────────────────────────────────────────────

def analyze_entropy(
    df: pd.DataFrame,
    entropy_window: int = 20,
    entropy_bins: int = 10,
    hurst_max_lag: int = 40,
) -> EntropyAnalysis:
    """
    Perform complete entropy-based regime detection on a stock's price data.

    Args:
      df: OHLCV DataFrame (minimum 50 bars recommended)
      entropy_window: Rolling window for entropy calculation
      entropy_bins: Number of discretization bins
      hurst_max_lag: Maximum lag for Hurst exponent R/S analysis

    Returns:
      EntropyAnalysis with entropy, Hurst, regime, and dip quality score
    """
    result = EntropyAnalysis()

    try:
        if len(df) < max(entropy_window + 5, 30):
            logger.debug("Insufficient data for entropy analysis (need 30+ bars)")
            return result

        close = df["Close"].values.astype(float)

        # ── Log Returns ──────────────────────────────────────────────
        log_returns = np.diff(np.log(close))

        # ── Rolling Entropy ──────────────────────────────────────────
        rolling_ent = calculate_rolling_entropy(log_returns, entropy_window, entropy_bins)
        valid_entropy = rolling_ent[~np.isnan(rolling_ent)]

        if len(valid_entropy) < 5:
            return result

        current_entropy = float(valid_entropy[-1])
        result.entropy = current_entropy

        # Entropy percentile (where does current entropy sit vs recent history)
        result.entropy_percentile = float(
            np.sum(valid_entropy < current_entropy) / len(valid_entropy) * 100
        )

        # ── Entropy Rate of Change ───────────────────────────────────
        if len(valid_entropy) >= 5:
            ent_5_bars_ago = float(valid_entropy[-5])
            result.entropy_roc = float(current_entropy - ent_5_bars_ago)
            result.entropy_dropping = result.entropy_roc < -0.03  # Significant drop

        # ── Hurst Exponent ───────────────────────────────────────────
        result.hurst_exponent = calculate_hurst_exponent(close, hurst_max_lag)
        result.mean_reverting = result.hurst_exponent < 0.45
        result.trending = result.hurst_exponent > 0.55

        # ── Regime Classification ────────────────────────────────────
        # Determine if price is currently dropping
        price_dropping = float(close[-1]) < float(close[-5]) if len(close) >= 5 else False

        result.regime, result.regime_confidence = classify_regime(
            entropy=current_entropy,
            entropy_roc=result.entropy_roc,
            hurst=result.hurst_exponent,
            price_dropping=price_dropping,
        )

        # ── Dip Quality Score ────────────────────────────────────────
        result.dip_quality_score = calculate_dip_quality(
            entropy=current_entropy,
            entropy_roc=result.entropy_roc,
            entropy_percentile=result.entropy_percentile,
            hurst=result.hurst_exponent,
            regime=result.regime,
            regime_confidence=result.regime_confidence,
        )

    except Exception as e:
        logger.error(f"Entropy analysis failed: {e}")

    return result


# ── CLI Test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import yfinance as yf

    print("🧠 Entropy-Based Regime Detection")
    print("=" * 60)

    for ticker in ["AAPL", "TSLA", "NVDA", "SPY"]:
        data = yf.download(ticker, period="6mo", interval="1d", progress=False)
        if data.empty:
            print(f"  {ticker}: No data")
            continue

        result = analyze_entropy(data)
        print(f"\n  {ticker}:")
        print(f"    Entropy:        {result.entropy:.4f} (p{result.entropy_percentile:.0f})")
        print(f"    Entropy RoC:    {result.entropy_roc:+.4f} {'⬇️ DROPPING' if result.entropy_dropping else ''}")
        print(f"    Hurst:          {result.hurst_exponent:.4f} ({'Mean-Reverting ✅' if result.mean_reverting else 'Trending ⚠️' if result.trending else 'Random'})")
        print(f"    Regime:         {result.regime.upper()} (confidence: {result.regime_confidence:.0%})")
        print(f"    Dip Quality:    {result.dip_quality_score:.0f}/100")

"""
Walk-Forward Backtester for Dip Finder v3 + Entropy.

Tests the ACTUAL scoring pipeline against historical data.
For each trading day, uses only trailing data (no future leak),
runs the full indicator + entropy + scoring pipeline, and tracks
forward returns at 5/10/20 days.

Usage:
  python -m backend.walk_forward_backtest --tickers AAPL,TSLA,NVDA --period 1y
"""

import logging
import argparse
import sys
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

from backend.indicators import (
    IndicatorResult,
    calculate_rsi,
    calculate_bollinger_bands,
    calculate_macd,
    detect_volume_spike,
    calculate_pct_drop_from_high,
    calculate_atr,
    calculate_vwap,
    calculate_roc,
    detect_candle_direction,
    calculate_trend,
    calculate_support_resistance,
    calculate_tp_sl,
)
from backend.entropy_analyzer import analyze_entropy
from backend.scoring_v3 import score_technical_v3
from backend.technical_analysis_v3 import EnhancedTechnicalAnalysis

logger = logging.getLogger(__name__)


# ── Data Structures ──────────────────────────────────────────────────────────

@dataclass
class BacktestSignal:
    """A single signal captured during backtesting."""
    ticker: str
    date: str                    # Entry date
    entry_price: float
    score: float                 # Technical score at entry
    entropy: float               # Shannon entropy at entry
    hurst: float                 # Hurst exponent at entry
    regime: str                  # Entropy regime at entry
    dip_quality: float           # Entropy dip quality score

    # Forward returns (filled after simulation)
    return_5d: Optional[float] = None
    return_10d: Optional[float] = None
    return_20d: Optional[float] = None
    max_drawdown: Optional[float] = None
    max_gain: Optional[float] = None
    hit_tp1: bool = False
    hit_sl: bool = False

    # Targets
    stop_loss: Optional[float] = None
    take_profit_1: Optional[float] = None
    
    # ML Features: Full indicator snapshot at signal time
    features: dict = field(default_factory=dict)

    @property
    def won_5d(self) -> bool:
        return (self.return_5d or 0) > 0

    @property
    def won_10d(self) -> bool:
        return (self.return_10d or 0) > 0

    @property
    def won_20d(self) -> bool:
        return (self.return_20d or 0) > 0

    @property
    def tier(self) -> str:
        if self.score >= 85:
            return "A (85+)"
        elif self.score >= 72:
            return "B (72-84)"
        elif self.score >= 55:
            return "C (55-71)"
        return "D (<55)"


@dataclass
class BacktestResult:
    """Aggregated backtest results for one ticker."""
    ticker: str
    period: str
    total_bars: int
    signals: list[BacktestSignal] = field(default_factory=list)

    @property
    def total_signals(self) -> int:
        return len(self.signals)


# ── Lightweight Indicator Builder ────────────────────────────────────────────

def build_indicators_from_window(ticker: str, df: pd.DataFrame) -> IndicatorResult:
    """
    Build indicators from a DataFrame slice WITHOUT external API calls.
    This is the backtesting-safe version — no earnings checks, no network.
    """
    result = IndicatorResult(ticker=ticker)

    try:
        if len(df) < 30:
            return result

        close = df["Close"]
        result.price = float(close.iloc[-1])

        # RSI
        result.rsi_value = calculate_rsi(df)
        result.rsi_oversold = (result.rsi_value or 100) < 30

        # Bollinger Bands
        bb = calculate_bollinger_bands(df)
        if bb:
            result.bb_lower, result.bb_middle, result.bb_upper = bb
            result.bb_breached = result.price < result.bb_lower if result.bb_lower else False

        # MACD
        macd = calculate_macd(df)
        if macd:
            result.macd_value, result.macd_signal, result.macd_histogram, result.macd_bearish_cross = macd

        # Volume
        vol = detect_volume_spike(df)
        if vol:
            result.volume_current, result.volume_avg_20, result.volume_spike = vol

        # % Drop from high
        drop = calculate_pct_drop_from_high(df)
        if drop:
            result.recent_high, result.pct_drop_from_high = drop

        # Candle direction
        result.candle_direction = detect_candle_direction(df)

        # VWAP
        result.vwap = calculate_vwap(df)

        # Rate of Change
        result.roc = calculate_roc(df)

        # ATR
        result.atr = calculate_atr(df)

        # Trend
        trend_data = calculate_trend(df)
        if trend_data:
            result.sma_50 = trend_data.get("sma_50")
            result.sma_200 = trend_data.get("sma_200")
            result.trend = trend_data.get("trend", "neutral")
            result.golden_cross = trend_data.get("golden_cross", False)
            result.death_cross = trend_data.get("death_cross", False)
            result.above_sma_50 = trend_data.get("above_sma_50", False)
            result.above_sma_200 = trend_data.get("above_sma_200", False)

        # Support/Resistance
        sr = calculate_support_resistance(df)
        if sr:
            result.support_levels = sr.get("support_levels", [])
            result.resistance_levels = sr.get("resistance_levels", [])
            result.nearest_support = sr.get("nearest_support")
            result.nearest_resistance = sr.get("nearest_resistance")
            result.at_support = sr.get("at_support", False)

        # TP/SL
        tp_sl = calculate_tp_sl(
            price=result.price,
            atr=result.atr,
            nearest_support=result.nearest_support,
            nearest_resistance=result.nearest_resistance,
            recent_high=result.recent_high,
            support_levels=result.support_levels,
        )
        result.stop_loss = tp_sl["stop_loss"]
        result.take_profit_1 = tp_sl["take_profit_1"]
        result.take_profit_2 = tp_sl["take_profit_2"]
        result.take_profit_3 = tp_sl["take_profit_3"]
        result.risk_reward_ratio = tp_sl["risk_reward_ratio"]

        # Entropy
        try:
            entropy_result = analyze_entropy(df)
            result.entropy_value = entropy_result.entropy
            result.hurst_exponent = entropy_result.hurst_exponent
            result.entropy_regime = entropy_result.regime
            result.dip_quality_score = entropy_result.dip_quality_score
        except Exception:
            pass

    except Exception as e:
        logger.debug("build_indicators error for %s: %s", ticker, e)

    return result


def build_tech_analysis_stub() -> EnhancedTechnicalAnalysis:
    """Create a minimal EnhancedTechnicalAnalysis with neutral defaults for backtesting."""
    return EnhancedTechnicalAnalysis(
        ticker="BACKTEST",
        adx_value=15.0,
        adx_trending=False,
        ichimoku_trend="neutral",
        stochastic_signal="neutral",
        stochastic_oversold=False,
        mtf_result=None,
        bullish_divergence=False,
        bearish_divergence=False,
    )


# ── Feature Extraction for ML ───────────────────────────────────────────────

def extract_ml_features(indicators: IndicatorResult, window_df: pd.DataFrame = None) -> dict:
    """
    Extract ML training features from IndicatorResult + raw price data.
    
    Returns dict with 32 features for XGBoost.
    """
    price = indicators.price or 0
    
    # Bollinger depth: how far below BB lower band
    bb_depth = 0.0
    if indicators.bb_lower and price > 0:
        bb_depth = ((indicators.bb_lower - price) / price) * 100
    
    # Volume ratio
    volume_ratio = 1.0
    if indicators.volume_current and indicators.volume_avg_20 and indicators.volume_avg_20 > 0:
        volume_ratio = indicators.volume_current / indicators.volume_avg_20
    
    # ATR as % of price
    atr_pct = 0.0
    if indicators.atr and price > 0:
        atr_pct = (indicators.atr / price) * 100
    
    # SMA distances
    sma50_dist = 0.0
    sma200_dist = 0.0
    if indicators.sma_50 and price > 0:
        sma50_dist = ((price - indicators.sma_50) / price) * 100
    if indicators.sma_200 and price > 0:
        sma200_dist = ((price - indicators.sma_200) / price) * 100
    
    features = {
        # === Original 21 features ===
        "rsi": indicators.rsi_value or 50.0,
        "bb_breached": 1 if indicators.bb_breached else 0,
        "bb_depth": bb_depth,
        "macd_histogram": indicators.macd_histogram or 0.0,
        "macd_bearish_cross": 1 if indicators.macd_bearish_cross else 0,
        "volume_ratio": volume_ratio,
        "volume_spike": 1 if indicators.volume_spike else 0,
        "pct_drop": indicators.pct_drop_from_high or 0.0,
        "at_support": 1 if indicators.at_support else 0,
        "pct_above_support": indicators.pct_above_support or 0.0,
        "rr_ratio": indicators.risk_reward_ratio or 1.0,
        "atr_pct": atr_pct,
        "sma50_dist": sma50_dist,
        "sma200_dist": sma200_dist,
        "golden_cross": 1 if indicators.golden_cross else 0,
        "above_sma_200": 1 if indicators.above_sma_200 else 0,
        "roc": indicators.roc or 0.0,
        "candle_green": 1 if getattr(indicators, 'candle_direction', None) == "green" else 0,
        "entropy": indicators.entropy_value or 1.0,
        "hurst": indicators.hurst_exponent or 0.5,
        "dip_quality": indicators.dip_quality_score or 0.0,
    }
    
    # === Advanced features (from raw price data) ===
    if window_df is not None and len(window_df) >= 20:
        close = window_df["Close"]
        
        # Pre-dip momentum (returns BEFORE the signal — how fast was the selloff?)
        c = close.values.astype(float)
        curr = c[-1]
        features["momentum_5d"] = ((curr / c[-6]) - 1) * 100 if len(c) >= 6 else 0.0
        features["momentum_10d"] = ((curr / c[-11]) - 1) * 100 if len(c) >= 11 else 0.0
        features["momentum_20d"] = ((curr / c[-21]) - 1) * 100 if len(c) >= 21 else 0.0
        
        # Consecutive red days (how many days in a row has price fallen?)
        red_count = 0
        for j in range(len(c) - 1, 0, -1):
            if c[j] < c[j - 1]:
                red_count += 1
            else:
                break
        features["consecutive_red_days"] = red_count
        
        # Bounce strength (how far did today's close recover from today's low?)
        low_today = float(window_df["Low"].iloc[-1])
        high_today = float(window_df["High"].iloc[-1])
        day_range = high_today - low_today
        if day_range > 0:
            features["bounce_strength"] = (curr - low_today) / day_range
        else:
            features["bounce_strength"] = 0.5
        
        # RSI slope (is RSI turning up or still falling?)
        rsi_5d_ago = calculate_rsi(window_df.iloc[:-5]) if len(window_df) > 35 else None
        rsi_now = indicators.rsi_value
        if rsi_5d_ago and rsi_now:
            features["rsi_slope"] = rsi_now - rsi_5d_ago
        else:
            features["rsi_slope"] = 0.0
        
        # MACD slope (momentum direction)
        if indicators.macd_histogram is not None:
            # Use MACD histogram as proxy — positive slope means momentum improving
            macd_vals = close.ewm(span=12).mean() - close.ewm(span=26).mean()
            signal_line = macd_vals.ewm(span=9).mean()
            hist = macd_vals - signal_line
            if len(hist) >= 5:
                features["macd_slope"] = float(hist.iloc[-1] - hist.iloc[-5])
            else:
                features["macd_slope"] = 0.0
        else:
            features["macd_slope"] = 0.0
        
        # Volatility regime (recent vs historical volatility)
        if len(c) >= 60:
            vol_recent = float(np.std(np.diff(np.log(c[-20:]))))
            vol_hist = float(np.std(np.diff(np.log(c[-60:]))))
            features["vol_regime"] = vol_recent / vol_hist if vol_hist > 0 else 1.0
        else:
            features["vol_regime"] = 1.0
        
        # Distance from VWAP
        if indicators.vwap and price > 0:
            features["vwap_dist"] = ((price - indicators.vwap) / price) * 100
        else:
            features["vwap_dist"] = 0.0
    else:
        # Defaults if no window data
        features.update({
            "momentum_5d": 0.0, "momentum_10d": 0.0, "momentum_20d": 0.0,
            "consecutive_red_days": 0, "bounce_strength": 0.5,
            "rsi_slope": 0.0, "macd_slope": 0.0, "vol_regime": 1.0,
            "vwap_dist": 0.0,
        })
    
    return features


# ── Quality Filters ──────────────────────────────────────────────────────────
def passes_quality_filters(indicators: IndicatorResult, quality_mode: bool = True) -> bool:
    """
    Focused quality gate: buy dips in confirmed uptrends.

    Strategy: The strongest alpha comes from buying oversold dips in
    macro uptrends (golden cross). 4 focused gates:
      1. RSI ≤ 40 (oversold — a dip is happening)
      2. Golden cross + above SMA200 (macro uptrend confirmed)
      3. Non-bearish candle (recovery starting, not still falling)
      4. Not distribution regime (smart money isn't exiting)
    """
    if not quality_mode:
        return True

    # ── Gate 1: RSI Oversold ─────────────────────────────────────────
    rsi = indicators.rsi_value or 100
    if rsi > 40:
        return False

    # ── Gate 2: Confirmed Uptrend ────────────────────────────────────
    # golden_cross = SMA50 > SMA200 (macro trend is up)
    # above_sma_200 = price still above the 200 SMA (not crashed)
    if not indicators.golden_cross:
        return False
    if not indicators.above_sma_200:
        return False

    # ── Gate 3: Recovery Candle ──────────────────────────────────────
    candle = getattr(indicators, 'candle_direction', None)
    if candle == "bearish":
        return False  # Still falling — wait for buyers

    # ── Gate 4: Not Distribution ─────────────────────────────────────
    regime = getattr(indicators, 'entropy_regime', None)
    if regime == "distribution":
        return False

    return True


# ── Walk-Forward Engine ──────────────────────────────────────────────────────

def run_backtest(
    ticker: str,
    period: str = "1y",
    lookback_window: int = 250,
    min_score: float = 55.0,
    cooldown_days: int = 5,
    quality_mode: bool = True,
) -> BacktestResult:
    """
    Walk-forward backtest on a single ticker.

    For each trading day:
      1. Slice trailing data window (no future leak)
      2. Build indicators + entropy
      3. Apply quality filters (dip, confluence, trend, regime)
      4. Score via score_technical_v3
      5. If score >= min_score → record signal + track 5/10/20-day returns

    Args:
      ticker: Stock symbol
      period: yfinance period string ('1y', '2y', etc.)
      lookback_window: Trailing bars for indicator calculation
      min_score: Minimum technical score to record
      cooldown_days: Min days between signals to avoid clustering
      quality_mode: If True, apply all quality filters (default). False = raw signals.
    """
    result = BacktestResult(ticker=ticker, period=period, total_bars=0)

    try:
        # Download historical data
        df = yf.download(ticker, period=period, interval="1d", progress=False)

        if df.empty or len(df) < lookback_window + 25:
            logger.warning("Insufficient data for %s (%d bars)", ticker, len(df))
            return result

        # Flatten MultiIndex columns if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        result.total_bars = len(df)
        close = df["Close"].values
        tech_stub = build_tech_analysis_stub()

        # Walk forward: from lookback_window → end-20 (need 20 days of future)
        start_idx = lookback_window
        end_idx = len(df) - 20
        last_signal_idx = -cooldown_days  # Allow first signal immediately

        for i in range(start_idx, end_idx):
            # Cooldown: skip if too soon after last signal
            if i - last_signal_idx < cooldown_days:
                continue

            # ── Trailing window only (NO future data) ────────────────
            window_df = df.iloc[i - lookback_window : i + 1].copy()

            # Build indicators on the window
            indicators = build_indicators_from_window(ticker, window_df)

            if indicators.price is None:
                continue

            # ── Quality gates (BEFORE scoring to save computation) ───
            if not passes_quality_filters(indicators, quality_mode):
                continue

            # Score
            score, breakdown = score_technical_v3(indicators, tech_stub)

            if score < min_score:
                continue

            # ── Record signal ────────────────────────────────────────
            entry_price = float(close[i])
            signal_date = str(df.index[i].date())

            signal = BacktestSignal(
                ticker=ticker,
                date=signal_date,
                entry_price=entry_price,
                score=round(score, 1),
                entropy=getattr(indicators, 'entropy_value', None) or 1.0,
                hurst=getattr(indicators, 'hurst_exponent', None) or 0.5,
                regime=getattr(indicators, 'entropy_regime', None) or "unknown",
                dip_quality=getattr(indicators, 'dip_quality_score', None) or 0.0,
                stop_loss=indicators.stop_loss,
                take_profit_1=indicators.take_profit_1,
                features=extract_ml_features(indicators, window_df),  # Extract ML features
            )

            # ── Forward returns (look ahead) ─────────────────────────
            future_prices = close[i + 1 : i + 21]  # Next 20 trading days

            if len(future_prices) >= 5:
                signal.return_5d = round(
                    ((float(future_prices[4]) - entry_price) / entry_price) * 100, 2
                )
            if len(future_prices) >= 10:
                signal.return_10d = round(
                    ((float(future_prices[9]) - entry_price) / entry_price) * 100, 2
                )
            if len(future_prices) >= 20:
                signal.return_20d = round(
                    ((float(future_prices[19]) - entry_price) / entry_price) * 100, 2
                )

            # Max drawdown & max gain within 20-day holding window
            if len(future_prices) > 0:
                future_returns_pct = (future_prices.astype(float) - entry_price) / entry_price * 100
                signal.max_drawdown = round(float(np.min(future_returns_pct)), 2)
                signal.max_gain = round(float(np.max(future_returns_pct)), 2)

            # TP1 / SL hit detection
            if signal.take_profit_1 and len(future_prices) > 0:
                signal.hit_tp1 = bool(np.any(future_prices.astype(float) >= signal.take_profit_1))
            if signal.stop_loss and len(future_prices) > 0:
                signal.hit_sl = bool(np.any(future_prices.astype(float) <= signal.stop_loss))

            result.signals.append(signal)
            last_signal_idx = i

    except Exception as e:
        logger.error("Backtest failed for %s: %s", ticker, e, exc_info=True)

    return result


# ── Statistics ───────────────────────────────────────────────────────────────

def compute_statistics(signals: list[BacktestSignal]) -> dict:
    """Compute performance statistics grouped by score tier and entropy regime."""
    if not signals:
        return {"total_signals": 0}

    stats = {"total_signals": len(signals)}

    # ── Overall win rates ────────────────────────────────────────────
    for period_name, period_days in [("5d", 5), ("10d", 10), ("20d", 20)]:
        attr_name = f"return_{period_name}"
        valid = [s for s in signals if getattr(s, attr_name) is not None]
        if valid:
            returns = [getattr(s, attr_name) for s in valid]
            wins = sum(1 for r in returns if r > 0)
            stats[f"win_rate_{period_name}"] = round(wins / len(valid) * 100, 1)
            stats[f"avg_return_{period_name}"] = round(float(np.mean(returns)), 2)
            stats[f"median_return_{period_name}"] = round(float(np.median(returns)), 2)

    # Sharpe (20d, annualized approx: ~12 non-overlapping 20d periods per year)
    valid_20d = [s.return_20d for s in signals if s.return_20d is not None]
    if len(valid_20d) > 1:
        std = float(np.std(valid_20d))
        if std > 0:
            stats["sharpe_20d"] = round(float(np.mean(valid_20d)) / std * np.sqrt(12), 2)

    # Max DD / Gain
    dd = [s.max_drawdown for s in signals if s.max_drawdown is not None]
    gains = [s.max_gain for s in signals if s.max_gain is not None]
    if dd:
        stats["avg_max_drawdown"] = round(float(np.mean(dd)), 2)
    if gains:
        stats["avg_max_gain"] = round(float(np.mean(gains)), 2)

    # Profit factor: sum of gains / sum of losses
    wins_total = sum(r for r in valid_20d if r > 0) if valid_20d else 0
    losses_total = abs(sum(r for r in valid_20d if r < 0)) if valid_20d else 0
    if losses_total > 0:
        stats["profit_factor_20d"] = round(wins_total / losses_total, 2)

    # TP/SL hit rates
    tp = [s for s in signals if s.take_profit_1 is not None]
    sl = [s for s in signals if s.stop_loss is not None]
    if tp:
        stats["tp1_hit_rate"] = round(sum(1 for s in tp if s.hit_tp1) / len(tp) * 100, 1)
    if sl:
        stats["sl_hit_rate"] = round(sum(1 for s in sl if s.hit_sl) / len(sl) * 100, 1)

    # ── Performance by Score Tier ────────────────────────────────────
    stats["by_tier"] = {}
    for tier_name in ["A (85+)", "B (72-84)", "C (55-71)", "D (<55)"]:
        tier_signals = [s for s in signals if s.tier == tier_name]
        if not tier_signals:
            continue
        tier_20d = [s for s in tier_signals if s.return_20d is not None]
        stats["by_tier"][tier_name] = {
            "count": len(tier_signals),
            "win_rate_20d": round(sum(1 for s in tier_20d if s.won_20d) / max(1, len(tier_20d)) * 100, 1) if tier_20d else 0,
            "avg_return_20d": round(float(np.mean([s.return_20d for s in tier_20d])), 2) if tier_20d else 0,
            "avg_score": round(float(np.mean([s.score for s in tier_signals])), 1),
        }

    # ── Performance by Entropy Regime ────────────────────────────────
    stats["by_regime"] = {}
    for regime in sorted(set(s.regime for s in signals)):
        regime_signals = [s for s in signals if s.regime == regime]
        regime_20d = [s for s in regime_signals if s.return_20d is not None]
        if not regime_20d:
            continue
        stats["by_regime"][regime] = {
            "count": len(regime_signals),
            "win_rate_20d": round(sum(1 for s in regime_20d if s.won_20d) / len(regime_20d) * 100, 1),
            "avg_return_20d": round(float(np.mean([s.return_20d for s in regime_20d])), 2),
            "avg_entropy": round(float(np.mean([s.entropy for s in regime_signals])), 3),
            "avg_hurst": round(float(np.mean([s.hurst for s in regime_signals])), 3),
        }

    return stats


# ── Report Printer ───────────────────────────────────────────────────────────

def print_report(results: list[BacktestResult], quality_mode: bool = True) -> dict:
    """Print a formatted backtest report."""
    mode_label = "QUALITY" if quality_mode else "RAW (no filters)"
    print("\n" + "=" * 70)
    print(f"  DIP FINDER v3 — BACKTEST [{mode_label}]")
    print("=" * 70)

    all_signals = []
    for r in results:
        all_signals.extend(r.signals)
        print(f"\n  {r.ticker} — {r.period} ({r.total_bars} bars, {r.total_signals} signals)")

    if not all_signals:
        print("\n  No signals generated. Try lowering --min-score or extending --period.")
        return {"total_signals": 0}

    stats = compute_statistics(all_signals)

    # ── Overall Performance ──────────────────────────────────────────
    print(f"\n{'─' * 70}")
    print(f"  OVERALL PERFORMANCE ({stats['total_signals']} signals across {len(results)} stocks)")
    print(f"{'─' * 70}")

    print(f"\n  {'Horizon':<12} {'Win Rate':>10} {'Avg Return':>12} {'Median Ret':>12}")
    print(f"  {'─'*12} {'─'*10} {'─'*12} {'─'*12}")
    for period in ["5d", "10d", "20d"]:
        wr = stats.get(f"win_rate_{period}", "—")
        ar = stats.get(f"avg_return_{period}", "—")
        mr = stats.get(f"median_return_{period}", "—")
        wr_str = f"{wr}%" if isinstance(wr, (int, float)) else wr
        ar_str = f"{ar:+.2f}%" if isinstance(ar, (int, float)) else ar
        mr_str = f"{mr:+.2f}%" if isinstance(mr, (int, float)) else mr
        print(f"  {period:<12} {wr_str:>10} {ar_str:>12} {mr_str:>12}")

    if "sharpe_20d" in stats:
        print(f"\n  Sharpe Ratio (20d, annualized): {stats['sharpe_20d']}")
    if "profit_factor_20d" in stats:
        print(f"  Profit Factor (20d):            {stats['profit_factor_20d']}")
    if "avg_max_drawdown" in stats:
        print(f"  Avg Max Drawdown (20d window):  {stats['avg_max_drawdown']:+.2f}%")
        print(f"  Avg Max Gain (20d window):      {stats['avg_max_gain']:+.2f}%")
    if "tp1_hit_rate" in stats:
        print(f"  TP1 Hit Rate:                   {stats['tp1_hit_rate']}%")
    if "sl_hit_rate" in stats:
        print(f"  SL Hit Rate:                    {stats['sl_hit_rate']}%")

    # ── By Score Tier ────────────────────────────────────────────────
    if stats.get("by_tier"):
        print(f"\n{'─' * 70}")
        print(f"  PERFORMANCE BY SCORE TIER")
        print(f"{'─' * 70}")
        print(f"\n  {'Tier':<15} {'Count':>6} {'Win Rate 20d':>14} {'Avg Ret 20d':>13} {'Avg Score':>10}")
        print(f"  {'─'*15} {'─'*6} {'─'*14} {'─'*13} {'─'*10}")
        for tier, d in sorted(stats["by_tier"].items()):
            print(f"  {tier:<15} {d['count']:>6} {d['win_rate_20d']:>13.1f}% {d['avg_return_20d']:>+12.2f}% {d['avg_score']:>10.1f}")

    # ── By Entropy Regime ────────────────────────────────────────────
    if stats.get("by_regime"):
        print(f"\n{'─' * 70}")
        print(f"  PERFORMANCE BY ENTROPY REGIME")
        print(f"{'─' * 70}")
        print(f"\n  {'Regime':<16} {'Count':>6} {'Win Rate 20d':>14} {'Avg Ret 20d':>13} {'Hurst':>7} {'Entropy':>8}")
        print(f"  {'─'*16} {'─'*6} {'─'*14} {'─'*13} {'─'*7} {'─'*8}")
        for regime, d in sorted(stats["by_regime"].items()):
            print(f"  {regime:<16} {d['count']:>6} {d['win_rate_20d']:>13.1f}% {d['avg_return_20d']:>+12.2f}% {d['avg_hurst']:>7.3f} {d['avg_entropy']:>8.3f}")

    # ── Top Signals ──────────────────────────────────────────────────
    by_score = sorted(all_signals, key=lambda s: s.score, reverse=True)
    print(f"\n{'─' * 70}")
    print(f"  TOP 5 SIGNALS BY SCORE")
    print(f"{'─' * 70}")
    print(f"\n  {'Ticker':<7} {'Date':<12} {'Score':>6} {'Regime':<14} {'5d':>7} {'10d':>7} {'20d':>7}")
    print(f"  {'─'*7} {'─'*12} {'─'*6} {'─'*14} {'─'*7} {'─'*7} {'─'*7}")
    for s in by_score[:5]:
        r5 = f"{s.return_5d:+.1f}%" if s.return_5d is not None else "—"
        r10 = f"{s.return_10d:+.1f}%" if s.return_10d is not None else "—"
        r20 = f"{s.return_20d:+.1f}%" if s.return_20d is not None else "—"
        print(f"  {s.ticker:<7} {s.date:<12} {s.score:>6.1f} {s.regime:<14} {r5:>7} {r10:>7} {r20:>7}")

    # ── Best regime signals ──────────────────────────────────────────
    accum_signals = [s for s in all_signals if s.regime == "accumulation" and s.return_20d is not None]
    if accum_signals:
        print(f"\n{'─' * 70}")
        print(f"  ACCUMULATION REGIME SIGNALS ({len(accum_signals)} found)")
        print(f"{'─' * 70}")
        accum_sorted = sorted(accum_signals, key=lambda s: s.dip_quality, reverse=True)
        print(f"\n  {'Ticker':<7} {'Date':<12} {'Score':>6} {'DipQ':>6} {'5d':>7} {'10d':>7} {'20d':>7}")
        print(f"  {'─'*7} {'─'*12} {'─'*6} {'─'*6} {'─'*7} {'─'*7} {'─'*7}")
        for s in accum_sorted[:5]:
            r5 = f"{s.return_5d:+.1f}%" if s.return_5d is not None else "—"
            r10 = f"{s.return_10d:+.1f}%" if s.return_10d is not None else "—"
            r20 = f"{s.return_20d:+.1f}%" if s.return_20d is not None else "—"
            print(f"  {s.ticker:<7} {s.date:<12} {s.score:>6.1f} {s.dip_quality:>6.1f} {r5:>7} {r10:>7} {r20:>7}")

    print(f"\n{'=' * 70}\n")
    return stats


# ── CLI Entry Point ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Dip Finder v3 Walk-Forward Backtester")
    parser.add_argument("--tickers", type=str, default="AAPL,MSFT,NVDA,TSLA,AMZN",
                        help="Comma-separated tickers (default: AAPL,MSFT,NVDA,TSLA,AMZN)")
    parser.add_argument("--period", type=str, default="1y",
                        help="Historical period: 1y, 2y, etc. (default: 1y)")
    parser.add_argument("--min-score", type=float, default=20.0,
                        help="Min technical score to record signal (default: 20)")
    parser.add_argument("--cooldown", type=int, default=5,
                        help="Min days between signals per stock (default: 5)")
    parser.add_argument("--window", type=int, default=250,
                        help="Trailing lookback window in bars (default: 250)")
    parser.add_argument("--raw", action="store_true",
                        help="Disable quality filters (raw signal mode for comparison)")
    args = parser.parse_args()

    tickers = [t.strip().upper() for t in args.tickers.split(",")]
    quality_mode = not args.raw

    mode_label = "QUALITY MODE" if quality_mode else "RAW MODE (no filters)"
    print(f"\n  Walk-Forward Backtest [{mode_label}]")
    print(f"   Tickers:  {', '.join(tickers)}")
    print(f"   Period:   {args.period}")
    print(f"   Min Score: {args.min_score}")
    print(f"   Cooldown: {args.cooldown}d | Window: {args.window} bars\n")

    results = []
    for ticker in tickers:
        print(f"  {ticker}...", end=" ", flush=True)
        r = run_backtest(
            ticker=ticker,
            period=args.period,
            lookback_window=args.window,
            min_score=args.min_score,
            cooldown_days=args.cooldown,
            quality_mode=quality_mode,
        )
        results.append(r)
        print(f"done ({r.total_signals} signals in {r.total_bars} bars)")

    print_report(results, quality_mode)


if __name__ == "__main__":
    main()

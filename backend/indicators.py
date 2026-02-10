"""
Technical indicator calculations for dip detection.
Uses the `ta` library for RSI, Bollinger Bands, and MACD.
Also implements:
  - Volume spike detection
  - Percent drop from high
  - Earnings calendar check (catalyst filter)
  - Trend analysis (50/200 SMA, uptrend/downtrend)
  - Support / resistance level detection
"""

import logging
import datetime
from dataclasses import dataclass
from typing import Optional

import pandas as pd
import numpy as np
import yfinance as yf
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
from ta.trend import MACD

from backend.config import (
    RSI_OVERSOLD,
    BOLLINGER_PERIOD,
    BOLLINGER_STD,
    VOLUME_SPIKE_MULTIPLIER,
    PERCENT_DROP_LOOKBACK,
    SMA_50_PERIOD,
    SMA_200_PERIOD,
    EARNINGS_LOOKBACK_DAYS,
    SUPPORT_LOOKBACK,
    SUPPORT_PROXIMITY_PCT,
)
from backend.entropy_analyzer import analyze_entropy

logger = logging.getLogger(__name__)


@dataclass
class IndicatorResult:
    """Results of all indicator calculations for a single stock."""
    ticker: str
    price: Optional[float] = None

    # RSI
    rsi_value: Optional[float] = None
    rsi_oversold: bool = False

    # Bollinger Bands
    bb_lower: Optional[float] = None
    bb_middle: Optional[float] = None
    bb_upper: Optional[float] = None
    bb_breached: bool = False

    # MACD
    macd_value: Optional[float] = None
    macd_signal: Optional[float] = None
    macd_histogram: Optional[float] = None
    macd_bearish_cross: bool = False

    # Volume
    volume_current: Optional[float] = None
    volume_avg_20: Optional[float] = None
    volume_spike: bool = False

    # Percent drop from recent high
    recent_high: Optional[float] = None
    pct_drop_from_high: Optional[float] = None

    # ── NEW: Earnings Catalyst ──────────────────────────────────────────
    had_recent_earnings: bool = False       # True if earnings within EARNINGS_LOOKBACK_DAYS
    earnings_date: Optional[str] = None     # The date of the most recent earnings
    days_since_earnings: Optional[int] = None

    # ── NEW: Trend Analysis ─────────────────────────────────────────────
    sma_50: Optional[float] = None
    sma_200: Optional[float] = None
    above_sma_50: bool = False              # price > 50 SMA
    above_sma_200: bool = False             # price > 200 SMA
    trend: str = "neutral"                  # "uptrend", "downtrend", "neutral"
    golden_cross: bool = False              # 50 SMA > 200 SMA
    death_cross: bool = False               # 50 SMA < 200 SMA

    # ── NEW: Support / Resistance ───────────────────────────────────────
    support_levels: list = None             # list of support price levels
    resistance_levels: list = None          # list of resistance price levels
    nearest_support: Optional[float] = None
    nearest_resistance: Optional[float] = None
    at_support: bool = False                # within SUPPORT_PROXIMITY_PCT of support
    pct_above_support: Optional[float] = None

    # ── NEW: ATR & TP/SL ─────────────────────────────────────────────────
    atr: Optional[float] = None              # Average True Range (14-day)
    stop_loss: Optional[float] = None        # Suggested stop loss price
    take_profit_1: Optional[float] = None    # Conservative TP (nearest resistance)
    take_profit_2: Optional[float] = None    # Moderate TP (50% retracement to high)
    take_profit_3: Optional[float] = None    # Aggressive TP (recent high)
    risk_reward_ratio: Optional[float] = None  # R:R using TP1

    # ── NEW: VWAP, ROC, Candle Direction ─────────────────────────────────
    vwap: Optional[float] = None              # Volume-Weighted Average Price
    roc: Optional[float] = None               # Rate of Change (14-period %)
    candle_direction: Optional[str] = None    # "red", "green", or "doji"

    # ── NEW: Entropy-Based Regime Detection ──────────────────────────────
    entropy_value: Optional[float] = None     # Shannon entropy (0-1, lower = more structured)
    hurst_exponent: Optional[float] = None    # Hurst (< 0.5 = mean-reverting)
    entropy_regime: Optional[str] = None      # "accumulation", "distribution", "trending", "random"
    dip_quality_score: Optional[float] = None # Entropy-based dip quality (0-100)

    def __post_init__(self):
        if self.support_levels is None:
            self.support_levels = []
        if self.resistance_levels is None:
            self.resistance_levels = []


# ── Core indicators (unchanged) ─────────────────────────────────────────────

def calculate_rsi(df: pd.DataFrame, period: int = 14) -> Optional[float]:
    """Calculate the latest RSI(14) value."""
    try:
        rsi = RSIIndicator(close=df["Close"], window=period)
        rsi_series = rsi.rsi()
        if rsi_series is not None and len(rsi_series.dropna()) > 0:
            return float(rsi_series.dropna().iloc[-1])
    except Exception as e:
        logger.debug("RSI calculation error: %s", e)
    return None


def calculate_bollinger_bands(df: pd.DataFrame,
                               period: int = BOLLINGER_PERIOD,
                               std_dev: int = BOLLINGER_STD) -> tuple:
    """Returns (lower, middle, upper) for the most recent bar."""
    try:
        bb = BollingerBands(close=df["Close"], window=period, window_dev=std_dev)
        lower = bb.bollinger_lband()
        middle = bb.bollinger_mavg()
        upper = bb.bollinger_hband()
        if lower is not None and len(lower.dropna()) > 0:
            return (
                float(lower.dropna().iloc[-1]),
                float(middle.dropna().iloc[-1]),
                float(upper.dropna().iloc[-1]),
            )
    except Exception as e:
        logger.debug("Bollinger Bands calculation error: %s", e)
    return None, None, None


def calculate_macd(df: pd.DataFrame) -> tuple:
    """Returns (macd_line, signal_line, histogram, bearish_cross)."""
    try:
        macd = MACD(close=df["Close"])
        macd_line = macd.macd()
        signal_line = macd.macd_signal()
        histogram = macd.macd_diff()
        if macd_line is not None and len(macd_line.dropna()) >= 2:
            macd_val = float(macd_line.dropna().iloc[-1])
            signal_val = float(signal_line.dropna().iloc[-1])
            hist_val = float(histogram.dropna().iloc[-1])
            macd_prev = float(macd_line.dropna().iloc[-2])
            signal_prev = float(signal_line.dropna().iloc[-2])
            bearish_cross = (macd_prev >= signal_prev) and (macd_val < signal_val)
            return macd_val, signal_val, hist_val, bearish_cross
    except Exception as e:
        logger.debug("MACD calculation error: %s", e)
    return None, None, None, False


def detect_volume_spike(df: pd.DataFrame,
                         multiplier: float = VOLUME_SPIKE_MULTIPLIER) -> tuple:
    """Returns (current_volume, avg_volume_20, is_spike)."""
    try:
        if len(df) < 21:
            return None, None, False
        current_vol = float(df["Volume"].iloc[-1])
        avg_vol = float(df["Volume"].iloc[-21:-1].mean())
        price_dropped = df["Close"].iloc[-1] < df["Close"].iloc[-2]
        is_spike = (current_vol > multiplier * avg_vol) and price_dropped
        return current_vol, avg_vol, is_spike
    except Exception as e:
        logger.debug("Volume spike detection error: %s", e)
    return None, None, False


def calculate_pct_drop_from_high(df: pd.DataFrame,
                                  lookback: int = PERCENT_DROP_LOOKBACK) -> tuple:
    """Returns (recent_high, pct_drop)."""
    try:
        window = df["High"].iloc[-lookback:] if len(df) >= lookback else df["High"]
        recent_high = float(window.max())
        current_price = float(df["Close"].iloc[-1])
        if recent_high > 0:
            pct_drop = ((current_price - recent_high) / recent_high) * 100.0
            return recent_high, pct_drop
    except Exception as e:
        logger.debug("Percent drop calculation error: %s", e)
    return None, None


# ── NEW: Earnings Catalyst Detection ────────────────────────────────────────

def check_recent_earnings(ticker: str,
                           lookback_days: int = EARNINGS_LOOKBACK_DAYS) -> tuple[bool, Optional[str], Optional[int]]:
    """
    Check if the stock had earnings within the last N trading days.
    Returns (had_recent_earnings, earnings_date_str, days_since_earnings).
    """
    try:
        stock = yf.Ticker(ticker)
        # Get earnings dates
        cal = stock.get_earnings_dates(limit=4)
        if cal is None or cal.empty:
            return False, None, None

        today = pd.Timestamp.now(tz="America/New_York").normalize()

        for idx in cal.index:
            earnings_dt = pd.Timestamp(idx)
            if earnings_dt.tzinfo is None:
                earnings_dt = earnings_dt.tz_localize("America/New_York")
            else:
                earnings_dt = earnings_dt.tz_convert("America/New_York")
            earnings_dt = earnings_dt.normalize()

            diff_days = (today - earnings_dt).days

            if 0 <= diff_days <= lookback_days:
                return True, earnings_dt.strftime("%Y-%m-%d"), diff_days

    except Exception as e:
        logger.debug("Earnings check error for %s: %s", ticker, e)

    return False, None, None


# ── NEW: Trend Analysis ─────────────────────────────────────────────────────

def calculate_trend(df: pd.DataFrame) -> dict:
    """
    Analyze the stock's trend using 50 and 200-day SMAs,
    plus higher-high / lower-low analysis.

    Returns dict with: sma_50, sma_200, above_sma_50, above_sma_200,
                        trend, golden_cross, death_cross
    """
    result = {
        "sma_50": None, "sma_200": None,
        "above_sma_50": False, "above_sma_200": False,
        "trend": "neutral", "golden_cross": False, "death_cross": False,
    }

    try:
        close = df["Close"]
        price = float(close.iloc[-1])

        # 50-day SMA
        if len(close) >= SMA_50_PERIOD:
            sma50 = float(close.rolling(window=SMA_50_PERIOD).mean().iloc[-1])
            result["sma_50"] = sma50
            result["above_sma_50"] = price > sma50

        # 200-day SMA
        if len(close) >= SMA_200_PERIOD:
            sma200 = float(close.rolling(window=SMA_200_PERIOD).mean().iloc[-1])
            result["sma_200"] = sma200
            result["above_sma_200"] = price > sma200

        # Golden/Death cross
        if result["sma_50"] is not None and result["sma_200"] is not None:
            result["golden_cross"] = result["sma_50"] > result["sma_200"]
            result["death_cross"] = result["sma_50"] < result["sma_200"]

        # Trend determination using multiple signals
        # Check recent higher-highs / lower-lows over last 60 days
        trend_window = min(60, len(df) - 1)
        if trend_window >= 20:
            recent = df.iloc[-trend_window:]
            mid = len(recent) // 2
            first_half_high = float(recent["High"].iloc[:mid].max())
            second_half_high = float(recent["High"].iloc[mid:].max())
            first_half_low = float(recent["Low"].iloc[:mid].min())
            second_half_low = float(recent["Low"].iloc[mid:].min())

            higher_highs = second_half_high > first_half_high
            higher_lows = second_half_low > first_half_low
            lower_highs = second_half_high < first_half_high
            lower_lows = second_half_low < first_half_low

            # Combine with SMA signals for robust trend detection
            bullish_signals = 0
            bearish_signals = 0

            if higher_highs:
                bullish_signals += 1
            if higher_lows:
                bullish_signals += 1
            if result.get("above_sma_50"):
                bullish_signals += 1
            if result.get("above_sma_200"):
                bullish_signals += 1
            if result.get("golden_cross"):
                bullish_signals += 1

            if lower_highs:
                bearish_signals += 1
            if lower_lows:
                bearish_signals += 1
            if not result.get("above_sma_50"):
                bearish_signals += 1
            if not result.get("above_sma_200"):
                bearish_signals += 1
            if result.get("death_cross"):
                bearish_signals += 1

            if bullish_signals >= 3:
                result["trend"] = "uptrend"
            elif bearish_signals >= 3:
                result["trend"] = "downtrend"
            else:
                result["trend"] = "neutral"

    except Exception as e:
        logger.debug("Trend calculation error: %s", e)

    return result


# ── NEW: Support / Resistance Levels ─────────────────────────────────────────

def calculate_support_resistance(df: pd.DataFrame,
                                  lookback: int = SUPPORT_LOOKBACK) -> dict:
    """
    Identify key support and resistance levels using:
    1. Swing lows (support) and swing highs (resistance)
    2. 50 and 200-day SMAs as dynamic support
    3. Round-number psychological levels

    Returns dict with: support_levels, resistance_levels,
                        nearest_support, nearest_resistance,
                        at_support, pct_above_support
    """
    result = {
        "support_levels": [],
        "resistance_levels": [],
        "nearest_support": None,
        "nearest_resistance": None,
        "at_support": False,
        "pct_above_support": None,
    }

    try:
        price = float(df["Close"].iloc[-1])
        window = df.iloc[-lookback:] if len(df) >= lookback else df

        # ── Swing lows (support) and swing highs (resistance) ───────────
        # A swing low: lower than both neighbors (using 5-bar window)
        lows = window["Low"].values
        highs = window["High"].values
        swing_supports = []
        swing_resistances = []

        for i in range(2, len(lows) - 2):
            # Swing low
            if (lows[i] < lows[i-1] and lows[i] < lows[i-2] and
                lows[i] < lows[i+1] and lows[i] < lows[i+2]):
                swing_supports.append(float(lows[i]))
            # Swing high
            if (highs[i] > highs[i-1] and highs[i] > highs[i-2] and
                highs[i] > highs[i+1] and highs[i] > highs[i+2]):
                swing_resistances.append(float(highs[i]))

        # ── Add SMA levels as dynamic support/resistance ────────────────
        close = df["Close"]
        if len(close) >= SMA_50_PERIOD:
            sma50 = float(close.rolling(window=SMA_50_PERIOD).mean().iloc[-1])
            if sma50 < price:
                swing_supports.append(sma50)
            else:
                swing_resistances.append(sma50)

        if len(close) >= SMA_200_PERIOD:
            sma200 = float(close.rolling(window=SMA_200_PERIOD).mean().iloc[-1])
            if sma200 < price:
                swing_supports.append(sma200)
            else:
                swing_resistances.append(sma200)

        # ── Cluster nearby levels (merge within 1.5% of each other) ─────
        swing_supports = _cluster_levels(sorted(swing_supports), price)
        swing_resistances = _cluster_levels(sorted(swing_resistances), price)

        # Filter: supports below price, resistances above price
        support_levels = sorted([s for s in swing_supports if s < price], reverse=True)
        resistance_levels = sorted([r for r in swing_resistances if r > price])

        result["support_levels"] = [round(s, 2) for s in support_levels[:5]]
        result["resistance_levels"] = [round(r, 2) for r in resistance_levels[:5]]

        # ── Nearest support / resistance ────────────────────────────────
        if support_levels:
            nearest = support_levels[0]
            result["nearest_support"] = round(nearest, 2)
            pct_above = ((price - nearest) / nearest) * 100
            result["pct_above_support"] = round(pct_above, 2)
            result["at_support"] = pct_above <= SUPPORT_PROXIMITY_PCT

        if resistance_levels:
            result["nearest_resistance"] = round(resistance_levels[0], 2)

    except Exception as e:
        logger.debug("Support/resistance calculation error: %s", e)

    return result


def _cluster_levels(levels: list[float], reference_price: float,
                     cluster_pct: float = 1.5) -> list[float]:
    """Merge price levels that are within cluster_pct% of each other."""
    if not levels:
        return []

    clustered = []
    cluster = [levels[0]]

    for i in range(1, len(levels)):
        if reference_price > 0:
            diff_pct = abs(levels[i] - cluster[-1]) / reference_price * 100
        else:
            diff_pct = 0

        if diff_pct <= cluster_pct:
            cluster.append(levels[i])
        else:
            # Average the cluster
            clustered.append(sum(cluster) / len(cluster))
            cluster = [levels[i]]

    if cluster:
        clustered.append(sum(cluster) / len(cluster))

    return clustered


# ── NEW: ATR + Take Profit / Stop Loss ───────────────────────────────────────

def calculate_atr(df: pd.DataFrame, period: int = 14) -> Optional[float]:
    """Calculate the Average True Range (ATR) over the given period."""
    try:
        if len(df) < period + 1:
            return None
        high = df["High"]
        low = df["Low"]
        close = df["Close"]
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = float(tr.rolling(window=period).mean().iloc[-1])
        return atr
    except Exception as e:
        logger.debug("ATR calculation error: %s", e)
    return None


# ── NEW: VWAP ────────────────────────────────────────────────────────────────

def calculate_vwap(df: pd.DataFrame) -> Optional[float]:
    """
    Calculate Volume-Weighted Average Price (VWAP).
    Uses the last 20 trading days as the rolling window.
    Institutional traders buy below VWAP and sell above.
    """
    try:
        if len(df) < 20:
            return None
        recent = df.iloc[-20:]
        typical_price = (recent["High"] + recent["Low"] + recent["Close"]) / 3
        cumulative_tp_vol = (typical_price * recent["Volume"]).sum()
        cumulative_vol = recent["Volume"].sum()
        if cumulative_vol > 0:
            return float(cumulative_tp_vol / cumulative_vol)
    except Exception as e:
        logger.debug("VWAP calculation error: %s", e)
    return None


# ── NEW: Rate of Change (ROC) ────────────────────────────────────────────────

def calculate_roc(df: pd.DataFrame, period: int = 14) -> Optional[float]:
    """
    Calculate Rate of Change (ROC) as percentage.
    ROC < -5% in short period = sharp dip (quality signal).
    ROC < -2% = moderate dip.
    """
    try:
        if len(df) < period + 1:
            return None
        current = float(df["Close"].iloc[-1])
        past = float(df["Close"].iloc[-period - 1])
        if past > 0:
            return float(((current - past) / past) * 100)
    except Exception as e:
        logger.debug("ROC calculation error: %s", e)
    return None


# ── NEW: Candle Direction ────────────────────────────────────────────────────

def detect_candle_direction(df: pd.DataFrame) -> Optional[str]:
    """
    Classify the latest candle as 'red', 'green', or 'doji'.
    Used for volume context scoring.
    """
    try:
        if len(df) < 1:
            return None
        open_price = float(df["Open"].iloc[-1])
        close_price = float(df["Close"].iloc[-1])
        body = abs(close_price - open_price)
        high_price = float(df["High"].iloc[-1])
        low_price = float(df["Low"].iloc[-1])
        full_range = high_price - low_price
        # Doji: body < 10% of full candle range
        if full_range > 0 and body / full_range < 0.1:
            return "doji"
        return "red" if close_price < open_price else "green"
    except Exception as e:
        logger.debug("Candle direction detection error: %s", e)
    return None


def calculate_tp_sl(price: float, atr: Optional[float],
                     nearest_support: Optional[float],
                     nearest_resistance: Optional[float],
                     recent_high: Optional[float],
                     support_levels: list) -> dict:
    """
    Calculate Stop Loss and Take Profit levels.

    Stop Loss logic:
      - Primary: below nearest support by 1×ATR (or 3% if no ATR)
      - If no support: price - 2×ATR (or price × 0.95)

    Take Profit logic:
      - TP1 (conservative): nearest resistance level
      - TP2 (moderate): 50% retracement from current price to recent high
      - TP3 (aggressive): recent high itself

    Risk/Reward: (TP1 - price) / (price - SL)
    """
    result = {
        "stop_loss": None,
        "take_profit_1": None,
        "take_profit_2": None,
        "take_profit_3": None,
        "risk_reward_ratio": None,
    }

    if price is None or price <= 0:
        return result

    # ── Stop Loss ───────────────────────────────────────────────────────
    if nearest_support is not None and nearest_support < price:
        # Place SL below support by 1×ATR or 1.5%
        buffer = atr if atr else price * 0.015
        result["stop_loss"] = round(nearest_support - buffer, 2)
    elif len(support_levels) >= 2:
        # Use second support level if first is too close
        second_support = support_levels[1] if support_levels[1] < price else None
        if second_support:
            buffer = atr * 0.5 if atr else price * 0.01
            result["stop_loss"] = round(second_support - buffer, 2)
    
    # Fallback: ATR-based SL
    if result["stop_loss"] is None:
        if atr:
            result["stop_loss"] = round(price - 2.0 * atr, 2)
        else:
            result["stop_loss"] = round(price * 0.95, 2)  # 5% below

    # Ensure SL is positive and below price
    if result["stop_loss"] is not None:
        result["stop_loss"] = max(result["stop_loss"], price * 0.80)  # cap at 20% loss
        if result["stop_loss"] >= price:
            result["stop_loss"] = round(price * 0.97, 2)

    # ── Take Profit ─────────────────────────────────────────────────────
    # TP1: nearest resistance
    if nearest_resistance is not None and nearest_resistance > price:
        result["take_profit_1"] = round(nearest_resistance, 2)

    # TP2: 50% retracement toward recent high
    if recent_high is not None and recent_high > price:
        halfway = price + (recent_high - price) * 0.5
        result["take_profit_2"] = round(halfway, 2)

    # TP3: recent high (full recovery)
    if recent_high is not None and recent_high > price:
        result["take_profit_3"] = round(recent_high, 2)

    # Fallbacks if no resistance/high data
    if result["take_profit_1"] is None:
        if atr:
            result["take_profit_1"] = round(price + 2.0 * atr, 2)
        else:
            result["take_profit_1"] = round(price * 1.05, 2)

    if result["take_profit_2"] is None:
        if atr:
            result["take_profit_2"] = round(price + 3.0 * atr, 2)
        else:
            result["take_profit_2"] = round(price * 1.08, 2)

    if result["take_profit_3"] is None:
        if atr:
            result["take_profit_3"] = round(price + 5.0 * atr, 2)
        else:
            result["take_profit_3"] = round(price * 1.12, 2)

    # ── Risk / Reward ratio (using TP1) ─────────────────────────────────
    sl = result["stop_loss"]
    tp1 = result["take_profit_1"]
    if sl is not None and tp1 is not None and sl < price:
        risk = price - sl
        reward = tp1 - price
        if risk > 0:
            result["risk_reward_ratio"] = round(reward / risk, 2)

    return result


# ── Main analysis function ───────────────────────────────────────────────────

def analyze_stock(ticker: str, df: pd.DataFrame,
                   check_earnings: bool = True) -> IndicatorResult:
    """
    Run all indicators on a stock's DataFrame and return an IndicatorResult.
    Includes earnings check, trend analysis, and support/resistance levels.
    """
    result = IndicatorResult(ticker=ticker)

    # Current price
    result.price = float(df["Close"].iloc[-1])

    # RSI
    result.rsi_value = calculate_rsi(df)
    if result.rsi_value is not None:
        result.rsi_oversold = result.rsi_value < RSI_OVERSOLD

    # Bollinger Bands
    result.bb_lower, result.bb_middle, result.bb_upper = calculate_bollinger_bands(df)
    if result.bb_lower is not None and result.price is not None:
        result.bb_breached = result.price < result.bb_lower

    # MACD
    result.macd_value, result.macd_signal, result.macd_histogram, result.macd_bearish_cross = calculate_macd(df)

    # Volume spike
    result.volume_current, result.volume_avg_20, result.volume_spike = detect_volume_spike(df)

    # Percent drop from high
    result.recent_high, result.pct_drop_from_high = calculate_pct_drop_from_high(df)

    # ── Earnings check ──────────────────────────────────────────────────
    if check_earnings:
        result.had_recent_earnings, result.earnings_date, result.days_since_earnings = (
            check_recent_earnings(ticker)
        )

    # ── Trend analysis ──────────────────────────────────────────────────
    trend_data = calculate_trend(df)
    result.sma_50 = trend_data["sma_50"]
    result.sma_200 = trend_data["sma_200"]
    result.above_sma_50 = trend_data["above_sma_50"]
    result.above_sma_200 = trend_data["above_sma_200"]
    result.trend = trend_data["trend"]
    result.golden_cross = trend_data["golden_cross"]
    result.death_cross = trend_data["death_cross"]

    # ── Support / Resistance ────────────────────────────────────────────
    sr_data = calculate_support_resistance(df)
    result.support_levels = sr_data["support_levels"]
    result.resistance_levels = sr_data["resistance_levels"]
    result.nearest_support = sr_data["nearest_support"]
    result.nearest_resistance = sr_data["nearest_resistance"]
    result.at_support = sr_data["at_support"]
    result.pct_above_support = sr_data["pct_above_support"]

    # ── ATR + TP/SL ─────────────────────────────────────────────────────
    result.atr = calculate_atr(df)
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

    # ── VWAP ────────────────────────────────────────────────────────────
    result.vwap = calculate_vwap(df)

    # ── Rate of Change ──────────────────────────────────────────────────
    result.roc = calculate_roc(df)

    # ── Candle Direction ────────────────────────────────────────────────
    result.candle_direction = detect_candle_direction(df)

    # ── Entropy-Based Regime Detection ─────────────────────────────────
    try:
        entropy_result = analyze_entropy(df)
        result.entropy_value = entropy_result.entropy
        result.hurst_exponent = entropy_result.hurst_exponent
        result.entropy_regime = entropy_result.regime
        result.dip_quality_score = entropy_result.dip_quality_score
    except Exception as e:
        logger.debug("Entropy analysis error for %s: %s", ticker, e)

    return result

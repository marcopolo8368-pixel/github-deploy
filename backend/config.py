"""
Simplified Daily Dip Scanner with Discord Integration
NO COMPLEX DEPENDENCIES - Just ML model scoring
"""

import os
import sys
import logging
from datetime import datetime
from typing import List, Dict
import requests
import yfinance as yf
import pandas as pd
import numpy as np

from backend.ml_scoring import MLScoringModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# Top 100 most liquid US stocks
TOP_100_TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK-B", "V", "UNH",
    "JNJ", "WMT", "JPM", "MA", "PG", "XOM", "HD", "CVX", "MRK", "ABBV",
    "LLY", "PEP", "KO", "COST", "AVGO", "TMO", "MCD", "CSCO", "ACN", "ABT",
    "CRM", "ADBE", "NFLX", "NKE", "DHR", "VZ", "CMCSA", "TXN", "NEE", "INTC",
    "AMD", "QCOM", "PM", "UNP", "HON", "RTX", "LOW", "SPGI", "INTU", "UPS",
    "BMY", "CAT", "AMAT", "BA", "AMT", "GE", "SBUX", "DE", "PLD", "BLK",
    "ADP", "MDLZ", "GILD", "MMC", "ADI", "ISRG", "CI", "LMT", "PGR", "SYK",
    "REGN", "TJX", "C", "NOW", "ZTS", "VRTX", "ETN", "BSX", "SCHW", "CB",
    "MO", "EOG", "DUK", "SO", "PYPL", "FI", "BDX", "PNC", "WM", "ITW",
    "CL", "APH", "SLB", "USB", "AON", "MMM", "MCO", "GD", "CSX", "ICE"
]


def compute_simple_features(df: pd.DataFrame) -> Dict[str, float]:
    """
    Compute simplified features directly from price data
    """
    close = df['Close'].values
    high = df['High'].values
    low = df['Low'].values
    volume = df['Volume'].values
    
    # Current price
    price = float(close[-1])
    
    # RSI (14-period)
    delta = np.diff(close)
    gains = np.where(delta > 0, delta, 0)
    losses = np.where(delta < 0, -delta, 0)
    avg_gain = np.mean(gains[-14:]) if len(gains) >= 14 else 0
    avg_loss = np.mean(losses[-14:]) if len(losses) >= 14 else 0
    rs = avg_gain / avg_loss if avg_loss > 0 else 100
    rsi = 100 - (100 / (1 + rs))
    
    # Bollinger Bands (20-period, 2 std)
    sma_20 = np.mean(close[-20:]) if len(close) >= 20 else price
    std_20 = np.std(close[-20:]) if len(close) >= 20 else 0
    bb_upper = sma_20 + (2 * std_20)
    bb_lower = sma_20 - (2 * std_20)
    bb_breached = 1 if price < bb_lower else 0
    bb_depth = ((bb_lower - price) / price) * 100 if price > 0 else 0
    
    # MACD (12, 26, 9)
    ema12 = pd.Series(close).ewm(span=12, adjust=False).mean().iloc[-1]
    ema26 = pd.Series(close).ewm(span=26, adjust=False).mean().iloc[-1]
    macd_line = ema12 - ema26
    signal_line = pd.Series(close).ewm(span=12, adjust=False).mean().ewm(span=9, adjust=False).mean().iloc[-1]
    macd_histogram = macd_line - signal_line
    
    # Volume
    volume_avg = np.mean(volume[-20:]) if len(volume) >= 20 else volume[-1]
    volume_ratio = float(volume[-1] / volume_avg) if volume_avg > 0 else 1.0
    volume_spike = 1 if volume_ratio > 2.0 else 0
    
    # % drop from high
    high_252 = np.max(high[-252:]) if len(high) >= 252 else np.max(high)
    pct_drop = ((high_252 - price) / high_252) * 100 if high_252 > 0 else 0
    
    # Support (lowest low in 60 days)
    support = np.min(low[-60:]) if len(low) >= 60 else np.min(low)
    at_support = 1 if abs(price - support) / support < 0.02 else 0
    pct_above_support = ((price - support) / support) * 100 if support > 0 else 0
    
    # SMAs
    sma_50 = np.mean(close[-50:]) if len(close) >= 50 else price
    sma_200 = np.mean(close[-200:]) if len(close) >= 200 else price
    sma50_dist = ((price - sma_50) / price) * 100 if price > 0 else 0
    sma200_dist = ((price - sma_200) / price) * 100 if price > 0 else 0
    golden_cross = 1 if sma_50 > sma_200 else 0
    above_sma_200 = 1 if price > sma_200 else 0
    
    # Momentum
    momentum_5d = ((price / close[-6]) - 1) * 100 if len(close) >= 6 else 0
    momentum_10d = ((price / close[-11]) - 1) * 100 if len(close) >= 11 else 0
    momentum_20d = ((price / close[-21]) - 1) * 100 if len(close) >= 21 else 0
    
    # Consecutive red days
    red_count = 0
    for i in range(len(close) - 1, 0, -1):
        if close[i] < close[i-1]:
            red_count += 1
        else:
            break
    
    # Bounce strength
    day_range = high[-1] - low[-1]
    bounce_strength = (price - low[-1]) / day_range if day_range > 0 else 0.5
    
    # RSI slope (change over 5 days)
    if len(close) >= 20:
        delta_old = np.diff(close[:-5])
        gains_old = np.where(delta_old > 0, delta_old, 0)
        losses_old = np.where(delta_old < 0, -delta_old, 0)
        avg_gain_old = np.mean(gains_old[-14:]) if len(gains_old) >= 14 else 0
        avg_loss_old = np.mean(losses_old[-14:]) if len(losses_old) >= 14 else 0
        rs_old = avg_gain_old / avg_loss_old if avg_loss_old > 0 else 100
        rsi_old = 100 - (100 / (1 + rs_old))
        rsi_slope = rsi - rsi_old
    else:
        rsi_slope = 0
    
    # MACD slope
    macd_slope = 0  # Simplified
    
    # Volatility regime
    if len(close) >= 60:
        vol_recent = np.std(np.diff(np.log(close[-20:])))
        vol_hist = np.std(np.diff(np.log(close[-60:])))
        vol_regime = vol_recent / vol_hist if vol_hist > 0 else 1.0
    else:
        vol_regime = 1.0
    
    # ATR
    tr = np.maximum(high[-1] - low[-1], np.abs(high[-1] - close[-2])) if len(close) > 1 else high[-1] - low[-1]
    atr = np.mean([tr]) if tr > 0 else 0  # Simplified
    atr_pct = (atr / price) * 100 if price > 0 else 0
    
    # Risk/reward
    stop_loss = price * 0.93
    take_profit = price * 1.07
    rr_ratio = (take_profit - price) / (price - stop_loss) if price > stop_loss else 1.0
    
    # ROC (Rate of Change)
    roc = ((price / close[-10]) - 1) * 100 if len(close) >= 10 else 0
    
    # Candle direction
    candle_green = 1 if close[-1] > close[-2] and len(close) > 1 else 0
    
    # Entropy (simplified - just use volatility)
    entropy = float(np.std(close[-20:])) if len(close) >= 20 else 1.0
    
    # Hurst (simplified - assume mean reverting)
    hurst = 0.4
    
    # Dip quality (simplified - based on RSI + BB)
    dip_quality = (bb_breached * 50 + (1 - rsi/100) * 50) / 100
    
    # VWAP distance (simplified)
    vwap = np.sum(close * volume) / np.sum(volume) if np.sum(volume) > 0 else price
    vwap_dist = ((price - vwap) / price) * 100 if price > 0 else 0
    
    # Placeholder for missing features
    macd_bearish_cross = 0
    
    return {
        "rsi": rsi,
        "bb_breached": bb_breached,
        "bb_depth": bb_depth,
        "macd_histogram": macd_histogram,
        "macd_bearish_cross": macd_bearish_cross,
        "volume_ratio": volume_ratio,
        "volume_spike": volume_spike,
        "pct_drop": pct_drop,
        "at_support": at_support,
        "pct_above_support": pct_above_support,
        "rr_ratio": rr_ratio,
        "atr_pct": atr_pct,
        "sma50_dist": sma50_dist,
        "sma200_dist": sma200_dist,
        "golden_cross": golden_cross,
        "above_sma_200": above_sma_200,
        "roc": roc,
        "candle_green": candle_green,
        "entropy": entropy,
        "hurst": hurst,
        "dip_quality": dip_quality,
        "momentum_5d": momentum_5d,
        "momentum_10d": momentum_10d,
        "momentum_20d": momentum_20d,
        "consecutive_red_days": red_count,
        "bounce_strength": bounce_strength,
        "rsi_slope": rsi_slope,
        "macd_slope": macd_slope,
        "vol_regime": vol_regime,
        "vwap_dist": vwap_dist,
        "price": price,
        "stop_loss": stop_loss,
        "take_profit": take_profit,
    }


class DipSignal:
    """A dip signal with ML score"""
    def __init__(self, ticker: str, price: float, ml_score: float, features: Dict):
        self.ticker = ticker
        self.price = price
        self.ml_score = ml_score
        self.features = features
        self.stop_loss = features.get('stop_loss', price * 0.93)
        self.take_profit = features.get('take_profit', price * 1.07)
    
    def to_discord_field(self) -> Dict:
        """Format as Discord embed field"""
        rsi = self.features.get('rsi', 0)
        momentum_20d = self.features.get('momentum_20d', 0)
        above_sma200 = "✅" if self.features.get('above_sma_200', 0) == 1 else "❌"
        
        return {
            "name": f"{self.ticker} @ ${self.price:.2f} | Score: {self.ml_score:.0f}%",
            "value": (
                f"📊 RSI: {rsi:.0f} | 20d Mom: {momentum_20d:+.1f}% | SMA200: {above_sma200}\n"
                f"🎯 Entry: ${self.price:.2f} | Target: ${self.take_profit:.2f} | Stop: ${self.stop_loss:.2f}"
            ),
            "inline": False
        }


def scan_stocks(model: MLScoringModel, tickers: List[str], min_score: float = 65.0) -> List[DipSignal]:
    """Scan stocks and return high-confidence dip signals"""
    signals = []
    
    logger.info(f"Scanning {len(tickers)} stocks...")
    
    for i, ticker in enumerate(tickers, 1):
        if i % 20 == 0:
            logger.info(f"Progress: {i}/{len(tickers)}")
        
        try:
            # Download recent data
            df = yf.download(ticker, period="1y", interval="1d", progress=False)
            
            if df.empty or len(df) < 100:
                continue
            
            # Flatten MultiIndex if present
            if hasattr(df.columns, 'levels'):
                df.columns = df.columns.get_level_values(0)
            
            # Compute features
            features = compute_simple_features(df)
            
            # ML score
            ml_score, win_prob = model.predict_score(features)
            
            if ml_score >= min_score:
                signal = DipSignal(ticker, features['price'], ml_score, features)
                signals.append(signal)
        
        except Exception as e:
            logger.warning(f"Error scanning {ticker}: {e}")
    
    signals.sort(key=lambda x: x.ml_score, reverse=True)
    return signals


def post_to_discord(signals: List[DipSignal], webhook_url: str):
    """Post signals to Discord"""
    if not signals:
        logger.info("No signals to post")
        return
    
    top_signals = signals[:5]
    
    embed = {
        "title": f"🎯 Daily Dip Signals — {datetime.now().strftime('%B %d, %Y')}",
        "description": f"Top {len(top_signals)} high-confidence dips (ML score ≥ 65%)",
        "color": 0x00ff00,
        "fields": [sig.to_discord_field() for sig in top_signals],
        "footer": {"text": f"ML Model: 59% CV Accuracy | Scanned {len(TOP_100_TICKERS)} stocks"},
        "timestamp": datetime.utcnow().isoformat()
    }
    
    try:
        response = requests.post(webhook_url, json={"username": "Dip Finder Bot", "embeds": [embed]})
        response.raise_for_status()
        logger.info(f"✅ Posted {len(top_signals)} signals to Discord")
    except Exception as e:
        logger.error(f"Failed to post to Discord: {e}")


def main():
    webhook_url = os.getenv("DISCORD_WEBHOOK_URL")
    if not webhook_url:
        logger.error("DISCORD_WEBHOOK_URL not set!")
        sys.exit(1)
    
    model_path = "models/xgb_dip_scorer.pkl"
    if not os.path.exists(model_path):
        logger.error(f"Model not found: {model_path}")
        sys.exit(1)
    
    logger.info(f"Loading ML model from {model_path}")
    model = MLScoringModel(model_path=model_path)
    
    if model.model is None:
        logger.error("Failed to load ML model!")
        sys.exit(1)
    
    logger.info(f"Starting daily scan at {datetime.now()}")
    signals = scan_stocks(model, TOP_100_TICKERS, min_score=65.0)
    
    logger.info(f"Found {len(signals)} signals scoring ≥ 65%")
    post_to_discord(signals, webhook_url)
    logger.info("Daily scan complete!")


if __name__ == "__main__":
    main()

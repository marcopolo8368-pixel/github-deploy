"""
Daily Dip Scanner with Discord Integration

Scans top 100 liquid stocks daily, scores with ML model,
and posts top 5 high-confidence dips to Discord.

Environment variables required:
- DISCORD_WEBHOOK_URL: Your Discord webhook URL
"""

import os
import sys
import logging
from datetime import datetime
from typing import List, Dict
import requests
import yfinance as yf

from backend.ml_scoring import MLScoringModel
from backend.indicators import build_indicators_from_window
from backend.walk_forward_backtest import extract_ml_features

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# Top 100 most liquid US stocks (S&P 100 + high volume)
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


class DipSignal:
    """A dip signal with ML score and key metrics"""
    def __init__(self, ticker: str, price: float, ml_score: float, 
                 features: Dict[str, float], stop_loss: float, take_profit: float):
        self.ticker = ticker
        self.price = price
        self.ml_score = ml_score
        self.features = features
        self.stop_loss = stop_loss
        self.take_profit = take_profit
    
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
    """
    Scan stocks and return high-confidence dip signals
    """
    signals = []
    
    logger.info(f"Scanning {len(tickers)} stocks...")
    
    for i, ticker in enumerate(tickers, 1):
        if i % 20 == 0:
            logger.info(f"Progress: {i}/{len(tickers)}")
        
        try:
            # Download recent data (need 250 days for indicators)
            df = yf.download(ticker, period="1y", interval="1d", progress=False)
            
            if df.empty or len(df) < 100:
                continue
            
            # Flatten MultiIndex columns if present
            if hasattr(df.columns, 'levels'):
                df.columns = df.columns.get_level_values(0)
            
            # Build indicators on full window
            window_df = df.iloc[-250:].copy() if len(df) >= 250 else df.copy()
            indicators = build_indicators_from_window(ticker, window_df)
            
            if indicators.price is None:
                continue
            
            # Extract features
            features = extract_ml_features(indicators, window_df)
            
            # ML score
            ml_score, win_prob = model.predict_score(features)
            
            if ml_score >= min_score:
                signal = DipSignal(
                    ticker=ticker,
                    price=indicators.price,
                    ml_score=ml_score,
                    features=features,
                    stop_loss=indicators.stop_loss or indicators.price * 0.93,
                    take_profit=indicators.take_profit_1 or indicators.price * 1.07,
                )
                signals.append(signal)
        
        except Exception as e:
            logger.warning(f"Error scanning {ticker}: {e}")
    
    # Sort by ML score descending
    signals.sort(key=lambda x: x.ml_score, reverse=True)
    
    return signals


def post_to_discord(signals: List[DipSignal], webhook_url: str):
    """
    Post signals to Discord as rich embed
    """
    if not signals:
        logger.info("No signals to post")
        return
    
    # Take top 5
    top_signals = signals[:5]
    
    # Build Discord embed
    embed = {
        "title": f"🎯 Daily Dip Signals — {datetime.now().strftime('%B %d, %Y')}",
        "description": f"Top {len(top_signals)} high-confidence dip opportunities (ML score ≥ 65%)",
        "color": 0x00ff00,  # Green
        "fields": [sig.to_discord_field() for sig in top_signals],
        "footer": {
            "text": f"ML Model: 59% CV Accuracy | Scanned {len(TOP_100_TICKERS)} stocks"
        },
        "timestamp": datetime.utcnow().isoformat()
    }
    
    payload = {
        "username": "Dip Finder Bot",
        "embeds": [embed]
    }
    
    try:
        response = requests.post(webhook_url, json=payload)
        response.raise_for_status()
        logger.info(f"✅ Posted {len(top_signals)} signals to Discord")
    except Exception as e:
        logger.error(f"Failed to post to Discord: {e}")


def main():
    """
    Main scanner execution
    """
    # Get Discord webhook from environment
    webhook_url = os.getenv("DISCORD_WEBHOOK_URL")
    if not webhook_url:
        logger.error("DISCORD_WEBHOOK_URL environment variable not set!")
        logger.info("Set it in GitHub Secrets or locally: export DISCORD_WEBHOOK_URL='your_webhook'")
        sys.exit(1)
    
    # Load ML model
    model_path = "models/xgb_dip_scorer.pkl"
    if not os.path.exists(model_path):
        logger.error(f"ML model not found at {model_path}")
        sys.exit(1)
    
    logger.info(f"Loading ML model from {model_path}")
    model = MLScoringModel(model_path=model_path)
    
    if model.model is None:
        logger.error("Failed to load ML model!")
        sys.exit(1)
    
    # Scan stocks
    logger.info(f"Starting daily scan at {datetime.now()}")
    signals = scan_stocks(model, TOP_100_TICKERS, min_score=65.0)
    
    logger.info(f"Found {len(signals)} signals scoring ≥ 65%")
    
    # Post to Discord
    post_to_discord(signals, webhook_url)
    
    logger.info("Daily scan complete!")


if __name__ == "__main__":
    main()

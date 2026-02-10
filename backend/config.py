"""
Configuration for the Dip Finder application.
"""

import os

# ── Discord Webhook (loaded from GitHub Secrets) ─────────────────────────────
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL", "")

# ── Scanning Settings ────────────────────────────────────────────────────────
SCAN_INTERVAL_MINUTES = 30
MARKET_OPEN_HOUR = 9
MARKET_OPEN_MINUTE = 30
MARKET_CLOSE_HOUR = 16
MARKET_CLOSE_MINUTE = 0

# ── Data Settings ─────────────────────────────────────────────────────────────
HISTORY_PERIOD = "1y"
HISTORY_INTERVAL = "1d"

# ── Dip Scoring Thresholds ────────────────────────────────────────────────────
RSI_OVERSOLD = 30
RSI_WEIGHT = 25
BOLLINGER_PERIOD = 20
BOLLINGER_STD = 2
BOLLINGER_WEIGHT = 20
MACD_WEIGHT = 20
VOLUME_SPIKE_MULTIPLIER = 2.0
VOLUME_WEIGHT = 15
PERCENT_DROP_WEIGHT = 20
PERCENT_DROP_LOOKBACK = 252

# ── Earnings Catalyst Filter ─────────────────────────────────────────────────
EARNINGS_LOOKBACK_DAYS = 5
EARNINGS_PENALTY = 0.4

# ── Trend Analysis ────────────────────────────────────────────────────────────
SMA_50_PERIOD = 50
SMA_200_PERIOD = 200
TREND_BONUS = 10
DOWNTREND_PENALTY = 0.6

# ── Support / Resistance ─────────────────────────────────────────────────────
SUPPORT_LOOKBACK = 60
SUPPORT_PROXIMITY_PCT = 2.0

# ── Alert Threshold ───────────────────────────────────────────────────────────
ALERT_SCORE_THRESHOLD = 70

# ── Custom Watchlist ──────────────────────────────────────────────────────────
DEFAULT_WATCHLIST = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA",
    "NVDA", "META", "NFLX", "AMD", "INTC",
]

# ── Database ──────────────────────────────────────────────────────────────────
DATABASE_URL = "sqlite:///dip_finder.db"

# ── Server ────────────────────────────────────────────────────────────────────
HOST = "127.0.0.1"
PORT = 8000

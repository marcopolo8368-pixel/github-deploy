# 🎯 AI Dip Finder — Free Daily Scanner

Automated dip scanner with **59% ML accuracy** that posts high-confidence trading signals to Discord daily.

## 🚀 Quick Start

1. **Fork this repo** (click Fork button)
2. **Create Discord webhook** (Server Settings → Integrations → Webhooks)
3. **Add webhook to GitHub Secrets:**
   - Go to Settings → Secrets and variables → Actions
   - Click "New repository secret"
   - Name: `DISCORD_WEBHOOK_URL`
   - Value: Your webhook URL
4. **Enable workflows** (Actions tab → Enable)
5. **Test it:** Actions → Daily Dip Scanner → Run workflow

## 📊 What It Does

Scans **top 100 liquid stocks** daily at **4:30 PM ET** (after market close) and posts top 5 dips to Discord:

```
🎯 Daily Dip Signals — February 10, 2026

NVDA @ $118.50 | Score: 78%
📊 RSI: 28 | 20d Mom: -5.2% | SMA200: ✅
🎯 Entry: $118.50 | Target: $124 | Stop: $110

AMD @ $145.20 | Score: 74%
...
```

## 🤖 ML Model

- **59% cross-validation accuracy** on >2% return target
- **30 technical features** (RSI, momentum, BB, MACD, entropy, volatility regime, etc.)
- **XGBoost classifier** with hyperparameter tuning
- Trained on **9,749 signals** from 50 S&P 500 stocks over 5 years

## 💰 Cost

**$0/month** — GitHub Actions gives 2,000 free minutes/month (uses ~100 min/month)

## 📖 More Info

See [DEPLOYMENT.md](DEPLOYMENT.md) for detailed setup instructions.

## ⚙️ Configuration

Edit `backend/daily_scanner.py` to customize:
- Stock universe (default: top 100)
- Minimum ML score (default: 65%)
- Number of signals posted (default: top 5)

---

**Disclaimer:** This is for educational purposes. Not financial advice. Trade at your own risk.

# Deployment Guide: Free Daily Discord Scanner

This guide shows you how to deploy your dip scanner to run automatically every day using **GitHub Actions (100% FREE)**.

---

## 📋 Prerequisites

1. GitHub account (free)
2. Discord server where you're an admin
3. Your ML model trained (`models/xgb_dip_scorer.pkl`)

---

## 🚀 Setup (5 minutes)

### Step 1: Create Discord Webhook

1. Open your Discord server
2. Go to **Server Settings** → **Integrations** → **Webhooks**
3. Click **New Webhook**
4. Name it "Dip Finder Bot"
5. Select the channel where you want signals posted (e.g., #trading-signals)
6. Click **Copy Webhook URL**
7. Save this URL — you'll need it in Step 3

### Step 2: Push Code to GitHub

```bash
cd "c:\Users\marco\Desktop\Coding projects\v4"

# Initialize git if not already done
git init
git add .
git commit -m "Add daily scanner with Discord integration"

# Create GitHub repo (do this on github.com first)
# Then push:
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
git branch -M main
git push -u origin main
```

### Step 3: Add Discord Webhook to GitHub Secrets

1. Go to your GitHub repo
2. Click **Settings** → **Secrets and variables** → **Actions**
3. Click **New repository secret**
4. Name: `DISCORD_WEBHOOK_URL`
5. Value: Paste your Discord webhook URL from Step 1
6. Click **Add secret**

### Step 4: Enable GitHub Actions

1. Go to the **Actions** tab in your repo
2. You should see "Daily Dip Scanner" workflow
3. If prompted, click **Enable workflows**

---

## ✅ Testing

### Manual Test (Before Waiting for Daily Run)

1. Go to **Actions** tab
2. Click "Daily Dip Scanner" workflow
3. Click **Run workflow** → **Run workflow**
4. Wait ~5 minutes
5. Check your Discord channel for the bot's message!

---

## ⏰ Schedule

The scanner runs automatically:
- **Time:** 9:30 PM UTC (4:30 PM ET / 1:30 PM PT)
- **Days:** Monday-Friday (market days only)
- **What it does:**
  1. Scans top 100 liquid stocks
  2. Scores each with ML model
  3. Posts top 5 dips (score ≥ 65%) to Discord

---

## 📊 Discord Message Format

```
🎯 Daily Dip Signals — February 10, 2026

Top 5 high-confidence dip opportunities (ML score ≥ 65%)

NVDA @ $118.50 | Score: 78%
📊 RSI: 28 | 20d Mom: -5.2% | SMA200: ✅
🎯 Entry: $118.50 | Target: $124.00 | Stop: $110.00

AMD @ $145.20 | Score: 74%
📊 RSI: 31 | 20d Mom: -8.1% | SMA200: ✅
🎯 Entry: $145.20 | Target: $152.00 | Stop: $136.00

...

ML Model: 59% CV Accuracy | Scanned 100 stocks
```

---

## 🔧 Customization

### Change Scan Time

Edit `.github/workflows/daily_scan.yml`:

```yaml
schedule:
  - cron: '30 21 * * 1-5'  # 9:30 PM UTC
  
# Examples:
# 9:00 PM UTC (4:00 PM ET): '0 21 * * 1-5'
# 10:00 PM UTC (5:00 PM ET): '0 22 * * 1-5'
```

### Change Minimum Score

Edit `backend/daily_scanner.py`:

```python
signals = scan_stocks(model, TOP_100_TICKERS, min_score=70.0)  # Was 65.0
```

### Change Stock Universe

Edit `backend/daily_scanner.py`:

```python
# Add/remove tickers from TOP_100_TICKERS list
```

---

## 💰 Cost

**$0.00** — GitHub Actions gives 2,000 free minutes/month.  
Your scan takes ~5 minutes = **400 free scans/month** (plenty for daily runs).

---

## 🐛 Troubleshooting

### "No signals found"
- Market might be strong (no quality dips)
- Lower `min_score` from 65 to 60 or 55

### "Workflow failed"
- Check Actions tab for error logs
- Common issues:
  - Missing `DISCORD_WEBHOOK_URL` secret
  - Missing `models/xgb_dip_scorer.pkl` in repo

### "Model file not found"
Make sure to commit and push `models/xgb_dip_scorer.pkl`:

```bash
git add models/xgb_dip_scorer.pkl
git commit -m "Add trained ML model"
git push
```

---

## 📈 Next Steps

Once this is running:

1. **Track performance:** Log signals to a spreadsheet, check win rate weekly
2. **Tune threshold:** Adjust `min_score` based on real-world results
3. **Add features:** Options flow, dark pool data (see next_level_roadmap.md)
4. **Multi-timeframe:** Build 1H chart confirmation for better entries

Enjoy your free, automated dip scanner! 🚀

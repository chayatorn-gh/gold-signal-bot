# Gold Signal Engine — GitHub Actions Edition

EMA(20/50) crossover signal bot for Gold (`GC=F`, 15m candles), with
ATR-based risk management and Telegram alerts. Runs for free on GitHub
Actions every 15 minutes — no server required.

## Setup (5 steps)

### 1. Create a new GitHub repository
Push these files to a **private** repository (recommended, since it will
hold your trading state — private repos still get 2,000 free Actions
minutes/month, which is more than enough for a 15-minute cron job).

```bash
git init
git add .
git commit -m "Initial commit: Gold Signal Engine"
git branch -M main
git remote add origin https://github.com/<your-username>/<your-repo>.git
git push -u origin main
```

### 2. Create a Telegram bot
1. Message [@BotFather](https://t.me/BotFather) on Telegram → `/newbot` → follow prompts.
2. Copy the **bot token** it gives you (looks like `123456789:ABCdefGhIJKlmNoPQRstuVwxyZ`).
3. Send any message to your new bot, then visit
   `https://api.telegram.org/bot<YOUR_TOKEN>/getUpdates` in a browser to
   find your **chat ID** (the `"chat":{"id": ...}` field).

### 3. Add GitHub Secrets
In your repo: **Settings → Secrets and variables → Actions → New repository secret**

| Secret name          | Value                          |
|-----------------------|---------------------------------|
| `TELEGRAM_BOT_TOKEN`  | your bot token from step 2      |
| `TELEGRAM_CHAT_ID`    | your chat ID from step 2        |

Never commit these values directly into `gold_signal_engine.py` — the
script reads them from environment variables, which the workflow injects
from these Secrets.

### 4. Enable the workflow
The workflow file at `.github/workflows/gold_signal.yml` is already
configured to run every 15 minutes (`cron: "*/15 * * * *"`). Once pushed,
go to the **Actions** tab in your repo — GitHub may ask you to confirm
enabling workflows the first time.

You can also trigger it manually anytime via
**Actions → Gold Signal Engine → Run workflow**.

### 5. Enable the live dashboard (GitHub Pages)
This repo includes a visual dashboard at `docs/index.html` that reads
real price/indicator data from `docs/data.json` — the same file the bot
writes on every run. To publish it as a live website:

1. Repo **Settings → Pages**
2. Under **Build and deployment → Source**, select **Deploy from a branch**
3. Branch: `main`, folder: **`/docs`** → **Save**
4. After a minute, your dashboard is live at
   `https://<your-username>.github.io/<your-repo>/`

The dashboard auto-refreshes every 60 seconds by re-fetching `data.json`,
which the bot updates every 15 minutes. It shows the same EMA/RSI/MACD/
Bollinger analysis the bot computes, plus the last signal actually sent
to Telegram — it does **not** simulate or invent any data.

> ⚠️ Since `docs/index.html` is public once Pages is enabled, don't put
> secrets in it. It only ever reads `data.json`, which contains price/
> indicator data — no tokens.

### 6. Adjust risk parameters (optional)
Open `gold_signal_engine.py` and edit the constants near the top:

```python
ACCOUNT_CAPITAL_USD: float = 1000.0
RISK_PER_TRADE_PCT: float = 0.015
ATR_MULTIPLIER_SL: float = 2.0
TP_RR_RATIO: float = 2.0
```

## How it works

- Each run is **stateless** (a fresh GitHub-hosted VM), so
  `last_processed_candle_time` (and the last Telegram alert) are saved to
  `state.json` and committed back to the repo after every run — this is
  how the anti-spam "only alert once per candle" guard, and the
  dashboard's "last alert sent" panel, survive between the 15-minute runs.
- The script checks Asia/Bangkok market hours itself before fetching data,
  so runs during the weekend/daily-pause window exit immediately without
  wasting Actions minutes (the dashboard just flips to "ตลาดปิด" using
  the last known data instead of fetching fresh).
- `docs/data.json` is rebuilt every run with fresh EMA/SMA/RSI/MACD/
  Bollinger Band values and a rolling 80-candle history for the charts —
  the dashboard (`docs/index.html`) never computes indicators itself, it
  only renders what Python already calculated.
- `[skip ci]` in the auto-commit message prevents the state/data commit
  from re-triggering other workflows.

## Two separate signal systems (by design)

- **Telegram alerts** (`gold_signal_engine.py`): event-based — fires only
  when EMA20 crosses EMA50 on a newly closed candle **and** passes the
  RSI/MACD confirmation filter (see below), with an ATR-based stop-loss/
  take-profit and position size. This is what actually pings you.
- **Dashboard score** (`docs/index.html` + `build_dashboard_analysis()`):
  continuous — a weighted buy/sell score from SMA20/50, RSI, MACD, and
  Bollinger Bands, shown for visual context. It updates every run and now
  also drives the paper-trading simulation below.

## Signal confirmation filter (fewer false EMA-crossover alerts)

A raw EMA20/50 crossover on its own can whipsaw in choppy/sideways
conditions. Before an alert is actually sent to Telegram, it's now checked
against RSI and the MACD histogram:

- **BUY** requires RSI below `RSI_BUY_MAX` (default 70 — not already
  overbought) **and** MACD histogram > 0 (bullish momentum).
- **SELL** requires RSI above `RSI_SELL_MIN` (default 30 — not already
  oversold) **and** MACD histogram < 0 (bearish momentum).

Rejected crossovers are logged (not sent) and the reason is shown on the
dashboard under the Telegram card. Disable with
`REQUIRE_SIGNAL_CONFIRMATION = False` in `gold_signal_engine.py` if you'd
rather alert on every raw crossover again.

## Paper trading simulation (educational, no real money)

`update_paper_trading()` keeps one simulated position open at a time,
driven entirely by the **continuous dashboard score** (not the Telegram
alert): it opens a BUY/SELL when the score reaches `PAPER_MIN_SIGNAL_STRENGTH`
(default 3/5), sizes SL/TP the same ATR-based way as a real signal, and
closes on a TP hit, SL hit, or the score flipping direction. Every closed
trade — entry/exit price, reason, and P/L in USD — is appended to
`docs/trade_log.json` along with running win-rate/total P/L stats; the
dashboard's "ระบบจำลองการเทรด" card renders this directly. Starting capital
and risk-per-trade reuse `ACCOUNT_CAPITAL_USD` / `RISK_PER_TRADE_PCT`.

## Server-side history (fixes "different device, no history")

Previously, the dashboard's signal-history log lived only in the visiting
browser's `localStorage`, so opening the dashboard on a different device or
browser showed nothing. Now every dashboard-score change is appended to
`docs/signal_log.json` (capped at 1000 entries) by the Python bot itself and
committed to the repo alongside `docs/data.json` — the same history shows up
everywhere the page is opened. `docs/trade_log.json` (paper trades) works
the same way.

## Local testing

```bash
pip install -r requirements.txt
export TELEGRAM_BOT_TOKEN="your_token"
export TELEGRAM_CHAT_ID="your_chat_id"
python gold_signal_engine.py            # single run -- also writes docs/data.json
python gold_signal_engine.py --loop     # optional: local infinite loop (not for CI)
```

To preview the dashboard locally after running the script once:

```bash
cd docs && python -m http.server 8000
# open http://localhost:8000 in your browser
```

## Note on LINE Notify

An earlier version of this dashboard used **LINE Notify**, but LINE
permanently shut that service down on **March 31, 2025**. It cannot be
revived with a new token — LINE's own guidance is to migrate to the
Messaging API instead. This project uses **Telegram** for all alerts
instead, which remains free and actively maintained.

## What changed in this update

You reported three problems: inaccurate/whipsaw signals, alerts arriving
late or not at all, and a dashboard that doesn't reflect reality. Here's
what was actually wrong and what changed:

### 1. Alerts silently stopped for days — and nothing told you
The root cause: if `fetch_price_data()` failed (Yahoo Finance rate-limits
or blocks intraday requests fairly often) or the script hit any unhandled
exception, it logged a warning and exited with status 0. The GitHub
Actions run still shows a green checkmark, so there was **no signal
anywhere** that the bot had actually stopped working — you'd only notice
when you checked the dashboard and saw stale data.
- `fetch_price_data()` and `send_telegram_alert()` now retry (3 attempts,
  backoff) before giving up, so a single network blip no longer costs a
  cycle or a missed alert.
- The engine now tracks consecutive failures across runs. After 3 in a
  row (~45 min), it sends you **one** Telegram message telling you it's
  down, and one more when it recovers — it does not spam you every run.

### 2. Whipsaw signals in choppy conditions
The existing RSI/MACD confirmation filter checks *direction* but not
whether a real trend exists at all. Added an **ADX trend-strength
filter**: a crossover now also needs ADX ≥ 20 (configurable via
`ADX_MIN_TREND_STRENGTH`) to be alerted, rejecting crossovers that fire
in a flat, directionless market — the classic whipsaw scenario. Disable
with `REQUIRE_ADX_FILTER = False` if you'd rather not have it. ADX also
now shows on the dashboard's condition list so you can see why a
crossover was accepted or rejected.

### 3. Dashboard didn't reflect reality
The dashboard showed "ระบบทำงานปกติ" (working normally) as long as
`data.json` loaded at all — even if that file was 6 days old. It now
compares `generated_at_utc` to the current time: if the market is open
and the data is more than 25 minutes stale, the status dot turns orange
and the banner tells you the bot may be down, instead of quietly
displaying old numbers as if they were current.

### Also worth knowing
GitHub Actions' `cron` schedule is best-effort, not guaranteed —
GitHub can delay scheduled runs by several minutes during high platform
load, and there's nothing in this repo that can fix that; it's a
platform limitation. If you need tighter timing than GitHub's cron
provides, trigger the workflow externally via `workflow_dispatch` (e.g.
from a free service like cron-job.org calling the GitHub Actions API)
instead of relying solely on `schedule`.

## Round 2: accuracy validation + trend filter

### 4. Multi-timeframe (1H) trend filter
RSI/MACD/ADX confirm direction and trend *strength*, but not whether the
15m crossover agrees with the bigger picture. A common cause of losing
whipsaw trades is a 15m BUY signal firing in the middle of a 1H
downtrend. Added `calculate_htf_trend()`: it resamples the SAME
already-fetched 15m candles up to 1H (no extra API call, so no added
rate-limit risk) and derives a trend direction from an EMA20/EMA50
crossover on that timeframe. A BUY is only confirmed when the 1H trend
is up; a SELL only when it's down. Toggle with `REQUIRE_HTF_TREND_FILTER`,
tune with `HTF_RESAMPLE_RULE` / `HTF_TREND_EMA_FAST` / `HTF_TREND_EMA_SLOW`.
Shows on the dashboard as a "เทรนด์ 1H" condition badge.

### 5. `backtest.py` -- see real numbers before trusting the parameters
All the filter thresholds (ADX ≥ 20, RSI 70/30, EMA 20/50, 1H trend) are
industry-standard defaults, not values tuned against this specific
instrument's actual behavior. `backtest.py` runs the strategy against
real historical Yahoo Finance data and prints win rate, total R, and
max drawdown for both the raw EMA crossover and the fully-filtered
version, side by side, so you can see whether the filters are actually
helping instead of taking it on faith.

```bash
pip install -r requirements.txt
python backtest.py                                   # GC=F, 15m, last 60 days (yfinance's max for 15m)
python backtest.py --interval 1h --period 2y          # longer, coarser test
python backtest.py --ticker XAUUSD=X --period 60d
```

Read the limitations in the script's docstring before trusting the
output for real capital -- yfinance's 60-day cap on 15m data means this
is a recent-behavior check, not a multi-year edge validation, and no
spread/slippage is modeled. Re-run periodically and re-tune
`ADX_MIN_TREND_STRENGTH` / `RSI_BUY_MAX` / `RSI_SELL_MIN` etc. in
`gold_signal_engine.py` based on what you see.

## ⚠️ Disclaimer

This is an educational signal-generation tool, not financial advice. It
does not place real trades. Gold's free 15-minute Yahoo Finance data can
have gaps or delays outside CME hours, and EMA-crossover systems can
whipsaw in low-volatility, sideways conditions. Paper-trade the signals
before risking real capital.

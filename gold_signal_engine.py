#!/usr/bin/env python3
"""
================================================================================
 AUTOMATED GOLD SIGNAL ENGINE WITH TELEGRAM NOTIFICATIONS
 (GitHub Actions / single-run edition)
================================================================================
A production-ready EMA-crossover signal engine for Gold (XAU), using free
Yahoo Finance data, ATR-based risk management, an ICT trading-hours guard,
and Telegram alert dispatch.

This edition is designed to be triggered ONCE per invocation (e.g. by a
GitHub Actions cron schedule every 15 minutes) rather than running an
infinite `while True` loop. State that needs to persist between runs
(the last processed candle timestamp, to avoid duplicate alerts) is saved
to a small JSON file (`state.json`) that the calling workflow is expected
to commit back to the repository between runs.

Author: Senior Python Developer / Quant Trading Engineer
Dependencies: yfinance, pandas, numpy, pytz, requests

    pip install yfinance pandas numpy pytz requests

================================================================================
"""

from __future__ import annotations

import os
import sys
import json
import logging
import traceback
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Tuple, Literal

import numpy as np
import pandas as pd
import pytz
import requests
import yfinance as yf

# ==============================================================================
# 1. CONFIGURATION -- edit these values before running
# ==============================================================================

# --- Telegram ---
# Read from environment variables first (GitHub Actions Secrets), falling
# back to a placeholder for local/manual testing. NEVER hardcode real
# secrets directly in this file if the repo is public.
TELEGRAM_BOT_TOKEN: str = os.environ.get("TELEGRAM_BOT_TOKEN", "YOUR_TELEGRAM_BOT_TOKEN_HERE")
TELEGRAM_CHAT_ID: str = os.environ.get("TELEGRAM_CHAT_ID", "YOUR_TELEGRAM_CHAT_ID_HERE")

# --- State persistence (for the anti-spam "once per candle" guard) ---
STATE_FILE_PATH: str = os.environ.get("STATE_FILE_PATH", "state.json")

# --- Dashboard JSON output (consumed by docs/index.html via GitHub Pages) ---
DASHBOARD_JSON_PATH: str = os.environ.get("DASHBOARD_JSON_PATH", "docs/data.json")
DASHBOARD_HISTORY_POINTS: int = 80   # how many recent candles to expose to the chart

# --- Data source ---
TICKER: str = "GC=F"          # Gold Futures. Alternative: "XAUUSD=X" (Gold Spot CFD proxy)
INTERVAL: str = "15m"         # Candle timeframe
PERIOD: str = "5d"            # Lookback window (yfinance max for 15m intraday is ~60d, 5d is safe/fast)

# --- Indicator settings ---
EMA_FAST_PERIOD: int = 20
EMA_SLOW_PERIOD: int = 50
ATR_PERIOD: int = 14

# --- Risk management ---
ACCOUNT_CAPITAL_USD: float = 1000.0     # total trading capital
RISK_PER_TRADE_PCT: float = 0.015       # 1.5% of capital risked per trade
ATR_MULTIPLIER_SL: float = 2.0          # SL distance = ATR * this multiplier
TP_RR_RATIO: float = 2.0                # Take-profit = SL distance * this ratio (1:2 R:R)

# --- Trading hours guard (Asia/Bangkok, ICT = UTC+7) ---
MARKET_TIMEZONE: str = "Asia/Bangkok"
DAILY_PAUSE_START_HOUR: int = 3         # 03:00 ICT daily pause start (all days)
DAILY_PAUSE_END_HOUR: int = 7           # market resumes 07:00 ICT (weekday) / Monday 07:00 (weekend)

# --- Execution ---
# NOTE: In the GitHub Actions edition there is no infinite loop -- the
# workflow's cron schedule determines how often this script runs (e.g.
# every 15 minutes). POLL_INTERVAL_SECONDS is kept only for the optional
# local `--loop` mode (see bottom of file).
POLL_INTERVAL_SECONDS: int = 60
REQUEST_TIMEOUT_SECONDS: int = 15       # network timeout for yfinance / Telegram calls

# --- Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("gold_signal_engine")


# ==============================================================================
# 2. DATA CLASSES
# ==============================================================================

@dataclass
class TradeSignal:
    """Container for a fully-computed trade signal, ready to notify."""
    signal_type: Literal["BUY", "SELL"]
    candle_time: pd.Timestamp
    entry_price: float
    stop_loss: float
    take_profit: float
    atr_value: float
    sl_distance: float
    position_size_usd: float
    risk_amount_usd: float
    rr_ratio: float


# ==============================================================================
# 3. STATE PERSISTENCE (replaces the in-memory `last_processed_candle_time`
#    variable used by the infinite-loop edition, since each GitHub Actions
#    run starts a brand-new process with no memory of the previous run)
# ==============================================================================

def load_state(path: str = STATE_FILE_PATH) -> dict:
    """Reads the full state dict (candle time + last alert) from the JSON state file."""
    try:
        if not os.path.exists(path):
            logger.info("No existing state file at %s -- treating as first run.", path)
            return {}
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        logger.warning("Failed to read state file (%s) -- starting fresh: %s", path, exc)
        return {}


def load_last_processed_candle_time(path: str = STATE_FILE_PATH) -> Optional[pd.Timestamp]:
    """Reads the last processed candle timestamp from the JSON state file."""
    data = load_state(path)
    iso_ts = data.get("last_processed_candle_time")
    if not iso_ts:
        return None
    try:
        return pd.Timestamp(iso_ts)
    except Exception:
        return None


def save_state(candle_time: Optional[pd.Timestamp] = None,
                last_signal: Optional["TradeSignal"] = None,
                path: str = STATE_FILE_PATH) -> None:
    """
    Persists state to the JSON state file, merging with what's already there
    so that saving a new candle_time doesn't wipe out a previously-saved
    last_signal (and vice versa).
    """
    try:
        existing = load_state(path)
        if candle_time is not None:
            existing["last_processed_candle_time"] = candle_time.isoformat()
        if last_signal is not None:
            existing["last_signal"] = {
                "signal_type": last_signal.signal_type,
                "candle_time": last_signal.candle_time.isoformat(),
                "entry_price": last_signal.entry_price,
                "stop_loss": last_signal.stop_loss,
                "take_profit": last_signal.take_profit,
                "position_size_usd": last_signal.position_size_usd,
                "risk_amount_usd": last_signal.risk_amount_usd,
                "rr_ratio": last_signal.rr_ratio,
            }
        existing["updated_at_utc"] = datetime.utcnow().isoformat()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(existing, f, indent=2)
        logger.info("State saved to %s", path)
    except Exception as exc:
        logger.error("Failed to write state file (%s): %s", path, exc)


# ==============================================================================
# 4. MARKET HOURS GUARD (Asia/Bangkok, 24/5 logic)
# ==============================================================================

def is_market_open(now: Optional[datetime] = None) -> bool:
    """
    Determines whether the Gold market is considered "open" for signal
    generation, using Asia/Bangkok (ICT, UTC+7) local time.

    Rules:
      - Weekend closure: from Saturday 03:00 ICT through Monday 07:00 ICT
        (mirrors the real FX/Gold market closing Fri ~04:00 ICT and
         reopening Mon ~04:00 EST -> ~07:00 ICT; we use a conservative window).
      - Daily pause: every day (including weekdays) from 03:00 to 06:59 ICT,
        which covers the illiquid rollover/maintenance window.
    """
    tz = pytz.timezone(MARKET_TIMEZONE)
    now = now.astimezone(tz) if now else datetime.now(tz)

    weekday = now.weekday()  # Monday=0 ... Sunday=6
    hour = now.hour

    # --- Weekend closure ---
    # Saturday (5) from 03:00 onward -> closed
    if weekday == 5 and hour >= DAILY_PAUSE_START_HOUR:
        return False
    # All day Sunday (6) -> closed
    if weekday == 6:
        return False
    # Monday (0) before 07:00 -> still closed (weekend carry-over)
    if weekday == 0 and hour < DAILY_PAUSE_END_HOUR:
        return False

    # --- Daily maintenance pause (weekdays Tue-Fri + Monday post-07:00 already handled) ---
    # Applies Mon(after 7am already passed check above)-Fri: 03:00 - 06:59 ICT closed daily
    if DAILY_PAUSE_START_HOUR <= hour < DAILY_PAUSE_END_HOUR:
        return False

    return True


# ==============================================================================
# 5. DATA FETCHING
# ==============================================================================

def fetch_price_data(ticker: str = TICKER, interval: str = INTERVAL,
                      period: str = PERIOD) -> Optional[pd.DataFrame]:
    """
    Fetches OHLCV candle data from Yahoo Finance via yfinance.
    Returns None (instead of raising) on any failure so the main loop can
    gracefully skip this cycle without crashing.
    """
    try:
        df = yf.download(
            tickers=ticker,
            interval=interval,
            period=period,
            progress=False,
            auto_adjust=True,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )

        if df is None or df.empty:
            logger.warning("yfinance returned empty data for %s (%s/%s).", ticker, interval, period)
            return None

        # yfinance sometimes returns MultiIndex columns (esp. single-ticker w/ new versions)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        required_cols = {"Open", "High", "Low", "Close"}
        if not required_cols.issubset(set(df.columns)):
            logger.error("Missing required OHLC columns in fetched data: %s", df.columns.tolist())
            return None

        df = df.dropna(subset=["Open", "High", "Low", "Close"])
        if df.empty:
            logger.warning("Data became empty after dropping NaN rows.")
            return None

        return df

    except Exception as exc:  # network errors, timeouts, parsing issues, etc.
        logger.error("Failed to fetch price data: %s", exc)
        logger.debug(traceback.format_exc())
        return None


# ==============================================================================
# 6. INDICATOR CALCULATIONS
# ==============================================================================

def calculate_ema(series: pd.Series, period: int) -> pd.Series:
    """Exponential Moving Average."""
    return series.ewm(span=period, adjust=False).mean()


def calculate_atr(df: pd.DataFrame, period: int = ATR_PERIOD) -> pd.Series:
    """
    Average True Range (Wilder's smoothing via EWM alpha=1/period), used to
    size stop-loss distance relative to current volatility.

    True Range = max(High-Low, |High-PrevClose|, |Low-PrevClose|)
    """
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)

    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # Wilder's smoothing ~ EWM with alpha = 1/period
    atr = true_range.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    return atr


def apply_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Attaches Fast EMA, Slow EMA, and ATR columns to the OHLCV dataframe."""
    df = df.copy()
    df["ema_fast"] = calculate_ema(df["Close"], EMA_FAST_PERIOD)
    df["ema_slow"] = calculate_ema(df["Close"], EMA_SLOW_PERIOD)
    df["atr"] = calculate_atr(df, ATR_PERIOD)
    return df


# ==============================================================================
# 6b. DASHBOARD INDICATORS (SMA / RSI / MACD / Bollinger Bands)
#     These power the visual dashboard's continuous buy/sell/hold "score",
#     which is a SEPARATE system from the ATR/EMA crossover engine above
#     that drives Telegram alerts. Keeping them separate mirrors how the
#     original dashboard mockup was designed (SMA golden-cross scoring for
#     the UI, vs. a discrete EMA-crossover event for actual trade alerts).
# ==============================================================================

def calculate_sma(series: pd.Series, period: int) -> pd.Series:
    """Simple Moving Average."""
    return series.rolling(window=period, min_periods=period).mean()


def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """
    Relative Strength Index using a simple rolling average of gains/losses
    (matches the classic "Cutler's RSI" variant): for each point, average
    gain and average loss are computed over the trailing `period` price
    changes, then RSI = 100 - 100 / (1 + avg_gain/avg_loss).
    """
    delta = series.diff()
    gains = delta.clip(lower=0).rolling(window=period, min_periods=period).mean()
    losses = (-delta.clip(upper=0)).rolling(window=period, min_periods=period).mean()
    rs = gains / losses.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.where(losses != 0, 100.0)  # if no losses at all -> RSI = 100
    return rsi


def calculate_macd(series: pd.Series, fast: int = 12, slow: int = 26,
                    signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    MACD Line = EMA(fast) - EMA(slow)
    Signal Line = EMA(signal) of the MACD line  (a REAL signal-line EMA,
    more accurate than a simplified 0.9x placeholder)
    Histogram = MACD Line - Signal Line
    """
    ema_fast = calculate_ema(series, fast)
    ema_slow = calculate_ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = calculate_ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def calculate_bollinger_bands(series: pd.Series, period: int = 20,
                               num_std: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Returns (upper_band, middle_band/SMA, lower_band)."""
    middle = calculate_sma(series, period)
    std = series.rolling(window=period, min_periods=period).std()
    upper = middle + (num_std * std)
    lower = middle - (num_std * std)
    return upper, middle, lower


def build_dashboard_analysis(df: pd.DataFrame) -> dict:
    """
    Re-implements the dashboard's original client-side scoring logic
    (SMA20/50 golden-cross, price vs SMA20, RSI zones, MACD histogram,
    Bollinger Band breakout) in Python, using REAL data, so the frontend
    becomes a pure renderer instead of an independent (fake) calculator.

    Returns a dict with the latest values, the buy/sell scores, the
    resulting signal ('buy' | 'sell' | 'hold'), a strength 0-5, and the
    human-readable condition list shown in the "เงื่อนไขสัญญาณ" card.
    """
    close = df["Close"]
    sma20 = calculate_sma(close, 20)
    sma50 = calculate_sma(close, 50)
    rsi = calculate_rsi(close, 14)
    macd_line, macd_signal, histogram = calculate_macd(close)
    bb_upper, bb_mid, bb_lower = calculate_bollinger_bands(close, 20)

    current = float(close.iloc[-1])
    ma20_val = sma20.iloc[-1]
    ma50_val = sma50.iloc[-1]
    rsi_val = rsi.iloc[-1]
    macd_val = macd_line.iloc[-1]
    hist_val = histogram.iloc[-1]
    bb_upper_val = bb_upper.iloc[-1]
    bb_lower_val = bb_lower.iloc[-1]

    buy_score = 0
    sell_score = 0
    conditions = []

    # --- SMA Golden/Death Cross ---
    if pd.notna(ma20_val) and pd.notna(ma50_val):
        if ma20_val > ma50_val:
            buy_score += 2
            conditions.append({"name": "MA20 > MA50 (Golden Cross)", "badge": "buy"})
        else:
            sell_score += 2
            conditions.append({"name": "MA20 < MA50 (Death Cross)", "badge": "sell"})

    # --- Price vs SMA20 ---
    if pd.notna(ma20_val):
        if current > ma20_val:
            buy_score += 1
            conditions.append({"name": f"ราคา > MA20 ({ma20_val:,.0f})", "badge": "buy"})
        else:
            sell_score += 1
            conditions.append({"name": f"ราคา < MA20 ({ma20_val:,.0f})", "badge": "sell"})

    # --- RSI zones ---
    if pd.notna(rsi_val):
        if rsi_val < 30:
            buy_score += 3
            conditions.append({"name": f"RSI {rsi_val:.1f} Oversold (<30)", "badge": "buy"})
        elif rsi_val > 70:
            sell_score += 3
            conditions.append({"name": f"RSI {rsi_val:.1f} Overbought (>70)", "badge": "sell"})
        elif rsi_val < 45:
            buy_score += 1
            conditions.append({"name": f"RSI {rsi_val:.1f} (Weak Bear)", "badge": "neutral"})
        elif rsi_val > 55:
            sell_score += 1
            conditions.append({"name": f"RSI {rsi_val:.1f} (Weak Bull)", "badge": "neutral"})
        else:
            conditions.append({"name": f"RSI {rsi_val:.1f} (Neutral)", "badge": "neutral"})

    # --- MACD histogram ---
    if pd.notna(hist_val):
        if hist_val > 0:
            buy_score += 2
            conditions.append({"name": f"MACD Histogram +{hist_val:.1f}", "badge": "buy"})
        else:
            sell_score += 2
            conditions.append({"name": f"MACD Histogram {hist_val:.1f}", "badge": "sell"})

    # --- Bollinger Band breakout ---
    if pd.notna(bb_upper_val) and pd.notna(bb_lower_val):
        if current < bb_lower_val:
            buy_score += 3
            conditions.append({"name": f"ราคาต่ำกว่า BB Lower ({bb_lower_val:,.0f})", "badge": "buy"})
        elif current > bb_upper_val:
            sell_score += 3
            conditions.append({"name": f"ราคาสูงกว่า BB Upper ({bb_upper_val:,.0f})", "badge": "sell"})
        else:
            conditions.append({"name": "ราคาอยู่ใน Bollinger Band", "badge": "neutral"})

    # --- Final signal + strength ---
    if buy_score > sell_score and buy_score >= 4:
        signal, strength = "buy", min(5, buy_score // 2)
    elif sell_score > buy_score and sell_score >= 4:
        signal, strength = "sell", min(5, sell_score // 2)
    else:
        signal, strength = "hold", 2

    def _safe(v):
        return None if v is None or (isinstance(v, float) and np.isnan(v)) else round(float(v), 2)

    return {
        "signal": signal,
        "strength": strength,
        "buy_score": buy_score,
        "sell_score": sell_score,
        "current_price": round(current, 2),
        "ma20": _safe(ma20_val),
        "ma50": _safe(ma50_val),
        "rsi": _safe(rsi_val),
        "macd": _safe(macd_val),
        "macd_signal": _safe(macd_signal.iloc[-1]),
        "histogram": _safe(hist_val),
        "bb_upper": _safe(bb_upper_val),
        "bb_lower": _safe(bb_lower_val),
        "conditions": conditions,
        # --- history series for the charts (aligned, nulls during warmup) ---
        "history": {
            "labels": [ts.strftime("%d/%m %H:%M") for ts in df.index[-DASHBOARD_HISTORY_POINTS:]],
            "prices": [round(float(v), 2) for v in close.tail(DASHBOARD_HISTORY_POINTS)],
            "ma20": [_safe(v) for v in sma20.tail(DASHBOARD_HISTORY_POINTS)],
            "ma50": [_safe(v) for v in sma50.tail(DASHBOARD_HISTORY_POINTS)],
            "rsi": [_safe(v) for v in rsi.tail(DASHBOARD_HISTORY_POINTS)],
            "macd": [_safe(v) for v in macd_line.tail(DASHBOARD_HISTORY_POINTS)],
            "histogram": [_safe(v) for v in histogram.tail(DASHBOARD_HISTORY_POINTS)],
        },
    }


def write_dashboard_json(payload: dict, path: str = DASHBOARD_JSON_PATH) -> None:
    """Writes the dashboard payload to disk as pretty-printed JSON."""
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        logger.info("Dashboard JSON written to %s", path)
    except Exception as exc:
        logger.error("Failed to write dashboard JSON (%s): %s", path, exc)


def read_existing_dashboard_json(path: str = DASHBOARD_JSON_PATH) -> Optional[dict]:
    """Reads back a previously-written dashboard JSON, or None if absent/invalid."""
    try:
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        logger.warning("Failed to read existing dashboard JSON (%s): %s", path, exc)
        return None


# ==============================================================================
# 7. SIGNAL DETECTION (EMA crossover on candle close)
# ==============================================================================

def detect_crossover_signal(df: pd.DataFrame) -> Optional[Literal["BUY", "SELL"]]:
    """
    Detects an EMA crossover on the LAST FULLY CLOSED candle by comparing
    the last two rows:
      - BUY  -> Fast EMA was <= Slow EMA previously, and is now > Slow EMA.
      - SELL -> Fast EMA was >= Slow EMA previously, and is now < Slow EMA.

    Requires at least EMA_SLOW_PERIOD + 2 valid rows to avoid false signals
    from unstable early EMA/ATR values.
    """
    min_rows_needed = EMA_SLOW_PERIOD + 2
    df_valid = df.dropna(subset=["ema_fast", "ema_slow", "atr"])
    if len(df_valid) < min_rows_needed:
        return None

    prev_row = df_valid.iloc[-2]
    curr_row = df_valid.iloc[-1]

    prev_fast, prev_slow = prev_row["ema_fast"], prev_row["ema_slow"]
    curr_fast, curr_slow = curr_row["ema_fast"], curr_row["ema_slow"]

    crossed_above = prev_fast <= prev_slow and curr_fast > curr_slow
    crossed_below = prev_fast >= prev_slow and curr_fast < curr_slow

    if crossed_above:
        return "BUY"
    if crossed_below:
        return "SELL"
    return None


# ==============================================================================
# 8. RISK MANAGEMENT & POSITION SIZING
# ==============================================================================

def build_trade_signal(
    signal_type: Literal["BUY", "SELL"],
    candle_time: pd.Timestamp,
    entry_price: float,
    atr_value: float,
) -> TradeSignal:
    """
    Computes SL, TP, position size, and risk amount for a detected signal,
    using ATR-based dynamic volatility management.
    """
    # --- Stop-loss distance is a multiple of current ATR (volatility-adaptive) ---
    sl_distance = atr_value * ATR_MULTIPLIER_SL

    if signal_type == "BUY":
        stop_loss = entry_price - sl_distance
        take_profit = entry_price + (sl_distance * TP_RR_RATIO)
    else:  # SELL
        stop_loss = entry_price + sl_distance
        take_profit = entry_price - (sl_distance * TP_RR_RATIO)

    # --- Position sizing: Risk Amount ($) / (% distance to SL) ---
    risk_amount_usd = ACCOUNT_CAPITAL_USD * RISK_PER_TRADE_PCT
    pct_distance_to_sl = sl_distance / entry_price  # SL distance expressed as % of entry price

    if pct_distance_to_sl <= 0:
        # Safety fallback: degenerate ATR/entry -> avoid division by zero
        position_size_usd = 0.0
    else:
        position_size_usd = risk_amount_usd / pct_distance_to_sl
        # Cap position size at maximum available account capital
        position_size_usd = min(position_size_usd, ACCOUNT_CAPITAL_USD)

    return TradeSignal(
        signal_type=signal_type,
        candle_time=candle_time,
        entry_price=round(entry_price, 2),
        stop_loss=round(stop_loss, 2),
        take_profit=round(take_profit, 2),
        atr_value=round(atr_value, 4),
        sl_distance=round(sl_distance, 2),
        position_size_usd=round(position_size_usd, 2),
        risk_amount_usd=round(risk_amount_usd, 2),
        rr_ratio=TP_RR_RATIO,
    )


# ==============================================================================
# 9. TELEGRAM NOTIFICATION DISPATCHER
# ==============================================================================

def format_telegram_message(signal: TradeSignal) -> str:
    """Builds an HTML-formatted Telegram message for a trade signal."""
    emoji = "🟢" if signal.signal_type == "BUY" else "🔴"
    candle_time_str = signal.candle_time.strftime("%Y-%m-%d %H:%M UTC") \
        if signal.candle_time.tzinfo else signal.candle_time.strftime("%Y-%m-%d %H:%M")

    message = (
        f"{emoji} <b>GOLD SIGNAL: {signal.signal_type}</b> {emoji}\n"
        f"───────────────────────\n"
        f"<b>Instrument:</b> {TICKER}\n"
        f"<b>Candle Time:</b> {candle_time_str}\n\n"
        f"<b>Entry Price:</b> ${signal.entry_price:,.2f}\n"
        f"<b>Stop Loss:</b> ${signal.stop_loss:,.2f}\n"
        f"<b>Take Profit:</b> ${signal.take_profit:,.2f}\n"
        f"<b>Risk/Reward:</b> 1:{signal.rr_ratio:.1f}\n\n"
        f"<b>ATR (14):</b> {signal.atr_value:,.4f}\n"
        f"<b>SL Distance:</b> ${signal.sl_distance:,.2f}\n\n"
        f"💰 <b>Position Size:</b> ${signal.position_size_usd:,.2f}\n"
        f"⚠️ <b>Max Risk:</b> ${signal.risk_amount_usd:,.2f} "
        f"({RISK_PER_TRADE_PCT * 100:.1f}% of ${ACCOUNT_CAPITAL_USD:,.2f})\n"
        f"───────────────────────\n"
        f"<i>Auto-generated by Gold Signal Engine (EMA{EMA_FAST_PERIOD}/EMA{EMA_SLOW_PERIOD} crossover)</i>"
    )
    return message


def send_telegram_alert(message: str) -> bool:
    """
    Sends a message to Telegram via the Bot API sendMessage endpoint.
    Returns True on success, False on failure (never raises).
    """
    if not TELEGRAM_BOT_TOKEN or "YOUR_TELEGRAM" in TELEGRAM_BOT_TOKEN:
        logger.warning("Telegram bot token not configured -- skipping alert dispatch.")
        return False

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
    }

    try:
        response = requests.post(url, json=payload, timeout=REQUEST_TIMEOUT_SECONDS)
        response.raise_for_status()
        result = response.json()
        if not result.get("ok", False):
            logger.error("Telegram API returned an error: %s", result)
            return False
        logger.info("Telegram alert sent successfully.")
        return True

    except requests.exceptions.RequestException as exc:
        logger.error("Failed to send Telegram message: %s", exc)
        return False
    except Exception as exc:  # e.g. JSON decode error
        logger.error("Unexpected error sending Telegram message: %s", exc)
        return False


# ==============================================================================
# 10. CORE PROCESSING PIPELINE (single cycle)
# ==============================================================================

def run_signal_cycle(
    last_processed_candle_time: Optional[pd.Timestamp],
) -> Tuple[Optional[pd.Timestamp], Optional[TradeSignal], Optional[pd.DataFrame]]:
    """
    Executes one full fetch -> indicator -> signal -> notify cycle.
    `last_processed_candle_time` is loaded from / saved to STATE_FILE_PATH
    by the caller, since each GitHub Actions run is a fresh process.

    Returns:
        (updated_last_processed_candle_time, signal_if_any, fetched_dataframe_or_None)
        The dataframe is returned too so the caller can reuse it for the
        dashboard JSON without fetching Yahoo Finance a second time.
    """
    df = fetch_price_data()
    if df is None:
        logger.warning("No data available this cycle; skipping.")
        return last_processed_candle_time, None, None

    df = apply_indicators(df)

    df_valid = df.dropna(subset=["ema_fast", "ema_slow", "atr"])
    if df_valid.empty:
        logger.warning("Not enough valid candles yet for indicator calculation.")
        return last_processed_candle_time, None, df

    latest_candle_time: pd.Timestamp = df_valid.index[-1]

    # --- Anti-spam guard: only process each closed candle once ---
    if last_processed_candle_time is not None and latest_candle_time <= last_processed_candle_time:
        logger.info("Candle %s already processed. No new closed candle yet.", latest_candle_time)
        return last_processed_candle_time, None, df

    signal_type = detect_crossover_signal(df)

    if signal_type is None:
        logger.info("No crossover signal on candle %s.", latest_candle_time)
        return latest_candle_time, None, df

    latest_row = df_valid.iloc[-1]
    entry_price = float(latest_row["Close"])
    atr_value = float(latest_row["atr"])

    signal = build_trade_signal(
        signal_type=signal_type,
        candle_time=latest_candle_time,
        entry_price=entry_price,
        atr_value=atr_value,
    )

    logger.info(
        "SIGNAL DETECTED: %s | Entry=%.2f SL=%.2f TP=%.2f Size=$%.2f",
        signal.signal_type, signal.entry_price, signal.stop_loss,
        signal.take_profit, signal.position_size_usd,
    )

    message = format_telegram_message(signal)
    send_telegram_alert(message)

    return latest_candle_time, signal, df


# ==============================================================================
# 11. SINGLE-RUN ENTRY POINT (GitHub Actions)
# ==============================================================================

def main() -> int:
    """
    Executes exactly ONE cycle, then exits. Designed to be invoked by an
    external scheduler (GitHub Actions cron) every 15 minutes instead of
    looping internally.

    Steps:
      1. Load `last_processed_candle_time` from STATE_FILE_PATH (previous run).
      2. Check the Asia/Bangkok market-hours guard; skip the cycle if closed.
      3. If open, run one full fetch/indicator/signal/notify cycle.
      4. Save the (possibly updated) candle time back to STATE_FILE_PATH so
         the next scheduled run knows what it already processed.
      5. Never raises: all exceptions are caught and logged so a bad run
         doesn't fail the whole workflow (still exits 0 by default).

    An optional `--loop` CLI flag is supported for local/manual testing,
    which falls back to an internal while-loop (NOT used in CI).
    """
    logger.info("=" * 60)
    logger.info("Gold Signal Engine -- single-run cycle starting.")
    logger.info("Ticker=%s Interval=%s EMA(%d/%d) ATR(%d)",
                TICKER, INTERVAL, EMA_FAST_PERIOD, EMA_SLOW_PERIOD, ATR_PERIOD)
    logger.info("Capital=$%.2f Risk/Trade=%.2f%% SL_ATR_Mult=%.1f RR=1:%.1f",
                ACCOUNT_CAPITAL_USD, RISK_PER_TRADE_PCT * 100, ATR_MULTIPLIER_SL, TP_RR_RATIO)
    logger.info("=" * 60)

    try:
        last_processed_candle_time = load_last_processed_candle_time()
        market_open = is_market_open()
        bangkok_now = datetime.now(pytz.timezone(MARKET_TIMEZONE))

        if not market_open:
            logger.info("Market closed (ICT time: %s). Skipping fetch; updating dashboard status only.",
                        bangkok_now.strftime("%Y-%m-%d %H:%M:%S %Z"))
            # Still refresh the dashboard's "market_open" flag / timestamp so the
            # UI accurately shows "closed" without needing a fresh data fetch.
            existing_payload = read_existing_dashboard_json() or {}
            existing_payload["market_open"] = False
            existing_payload["generated_at_utc"] = datetime.utcnow().isoformat()
            existing_payload["market_time_ict"] = bangkok_now.strftime("%Y-%m-%d %H:%M:%S")
            write_dashboard_json(existing_payload)
            return 0

        updated_candle_time, signal, df = run_signal_cycle(last_processed_candle_time)

        # Persist state only if it actually changed (avoids needless commits)
        if updated_candle_time is not None and updated_candle_time != last_processed_candle_time:
            save_state(candle_time=updated_candle_time, last_signal=signal)
        elif signal is not None:
            save_state(last_signal=signal)

        # --- Build & write the dashboard JSON from the SAME fetched data ---
        if df is not None and not df.empty:
            try:
                analysis = build_dashboard_analysis(df)
                state = load_state()
                payload = {
                    "generated_at_utc": datetime.utcnow().isoformat(),
                    "market_time_ict": bangkok_now.strftime("%Y-%m-%d %H:%M:%S"),
                    "market_open": True,
                    "ticker": TICKER,
                    "interval": INTERVAL,
                    "analysis": analysis,
                    "last_telegram_alert": state.get("last_signal"),
                }
                write_dashboard_json(payload)
            except Exception as exc:
                logger.error("Failed to build dashboard payload: %s", exc)
                logger.debug(traceback.format_exc())

    except Exception as exc:
        # Catch-all so a transient network/API error never fails the whole
        # GitHub Actions run with an unhandled traceback.
        logger.error("Unhandled exception during cycle: %s", exc)
        logger.debug(traceback.format_exc())

    logger.info("Cycle complete.")
    return 0


def run_local_loop() -> None:
    """
    OPTIONAL local/manual mode: mimics the original infinite `while True`
    loop for testing on your own machine (NOT used by GitHub Actions,
    which instead re-invokes main() on its own cron schedule).
    Enable with: python gold_signal_engine.py --loop
    """
    import time as _time
    logger.info("Running in local --loop mode (Ctrl+C to stop).")
    while True:
        main()
        _time.sleep(POLL_INTERVAL_SECONDS)


if __name__ == "__main__":
    if "--loop" in sys.argv:
        run_local_loop()
    else:
        sys.exit(main())

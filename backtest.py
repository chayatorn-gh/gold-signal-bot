"""
backtest.py -- Backtests the EMA-crossover strategy in gold_signal_engine.py
against real historical Yahoo Finance data, so you can see actual win-rate
and drawdown numbers instead of trusting the default parameters blindly.

It runs TWO versions side by side over the same historical data:
  1. "Baseline"       -- raw EMA20/EMA50 crossover, no filters at all.
  2. "With filters"   -- the live bot's full confirmation stack:
                         RSI + MACD + ADX + 1H trend filter.
This directly answers "did adding all these filters actually help?" with
numbers instead of guesswork.

USAGE
-----
    python backtest.py
    python backtest.py --ticker XAUUSD=X --interval 15m --period 60d
    python backtest.py --interval 1h --period 2y      # longer, coarser test

LIMITATIONS (please read before trusting the output)
-----------------------------------------------------
- yfinance caps intraday history: 15m candles are only available for the
  trailing ~60 days, 1h candles for ~2 years. This script can only ever
  validate "recent" market behavior at 15m -- it is NOT a multi-year
  edge test. If you want a longer, statistically stronger backtest at
  15m resolution, you need historical tick/OHLC data from a paid vendor
  (Dukascopy exports, TradingView exports, a broker's historical feed,
  etc.) and adapt `load_data()` to read that instead of calling yfinance.
- This assumes one open position at a time, closes on whichever of
  SL/TP is touched first within a candle (if both the candle's High and
  Low would have hit SL and TP, it conservatively assumes SL hit first --
  real intra-candle order is unknown from OHLC data alone).
- No spread/commission/slippage modeled. Real fills will be slightly
  worse than this shows, especially on a volatile instrument like gold.
- A run of 60 days of 15m data is a small sample (a few dozen trades at
  most) -- treat the win rate as a rough signal, not a statistically
  solid conclusion. Re-run periodically as more data accumulates.
"""
import argparse
import sys

import numpy as np
import pandas as pd

import gold_signal_engine as gse


def load_data(ticker: str, interval: str, period: str) -> pd.DataFrame:
    print(f"Fetching {ticker} {interval} candles, period={period} ...")
    df = gse.fetch_price_data(ticker=ticker, interval=interval, period=period)
    if df is None or df.empty:
        print("Failed to fetch historical data (check ticker/interval/period, "
              "or that this machine has network access to Yahoo Finance).")
        sys.exit(1)
    df = gse.apply_indicators(df)
    print(f"Loaded {len(df)} candles ({df.index[0]} -> {df.index[-1]})")
    return df


def run_backtest(df: pd.DataFrame, use_filters: bool) -> list:
    """
    Single-pass, vectorized-precompute backtest. One open position at a
    time (mirrors the live bot's anti-spam-per-candle behavior). All
    indicators used here are causal (rolling/ewm), so precomputing them
    over the whole series does not leak future information into past
    decisions.
    """
    rsi_series = gse.calculate_rsi(df["Close"])
    _, _, hist_series = gse.calculate_macd(df["Close"])
    htf_series = gse.calculate_htf_trend_series(df) if use_filters else None

    df_valid = df.dropna(subset=["ema_fast", "ema_slow", "atr"])
    prev_fast = df_valid["ema_fast"].shift(1)
    prev_slow = df_valid["ema_slow"].shift(1)
    crossed_up = (prev_fast <= prev_slow) & (df_valid["ema_fast"] > df_valid["ema_slow"])
    crossed_down = (prev_fast >= prev_slow) & (df_valid["ema_fast"] < df_valid["ema_slow"])

    trades = []
    open_trade = None

    for ts, row in df_valid.iterrows():
        if open_trade is not None:
            if open_trade["direction"] == "BUY":
                hit_sl = row["Low"] <= open_trade["stop_loss"]
                hit_tp = row["High"] >= open_trade["take_profit"]
            else:
                hit_sl = row["High"] >= open_trade["stop_loss"]
                hit_tp = row["Low"] <= open_trade["take_profit"]

            if hit_sl or hit_tp:
                # Conservative: if a single candle's range could have hit
                # either, assume the worse outcome (SL) hit first.
                r_multiple = -1.0 if hit_sl else gse.TP_RR_RATIO
                trades.append({
                    **open_trade,
                    "exit_time": ts,
                    "exit_price": open_trade["stop_loss"] if hit_sl else open_trade["take_profit"],
                    "r_multiple": r_multiple,
                })
                open_trade = None
            continue  # one position at a time -- don't look for new signals mid-trade

        if crossed_up.get(ts, False):
            raw_signal = "BUY"
        elif crossed_down.get(ts, False):
            raw_signal = "SELL"
        else:
            continue

        if use_filters:
            rsi_val = rsi_series.get(ts)
            hist_val = hist_series.get(ts)
            adx_val = row.get("adx")
            htf_val = htf_series.get(ts) if htf_series is not None else None
            confirmed, _ = gse.check_signal_confirmation(raw_signal, rsi_val, hist_val, adx_val, htf_val)
            if not confirmed:
                continue

        entry_price = float(row["Close"])
        atr_value = float(row["atr"])
        if pd.isna(atr_value) or atr_value <= 0:
            continue
        signal = gse.build_trade_signal(raw_signal, ts, entry_price, atr_value)
        open_trade = {
            "direction": raw_signal,
            "entry_time": ts,
            "entry_price": signal.entry_price,
            "stop_loss": signal.stop_loss,
            "take_profit": signal.take_profit,
        }

    return trades


def summarize(trades: list, label: str) -> None:
    print(f"\n=== {label} ===")
    if not trades:
        print("No trades generated over this period.")
        return

    r_values = np.array([t["r_multiple"] for t in trades])
    wins = r_values[r_values > 0]
    win_rate = len(wins) / len(r_values) * 100
    total_r = r_values.sum()

    equity = np.cumsum(r_values)
    peak = np.maximum.accumulate(equity)
    max_dd = (equity - peak).min()

    print(f"Trades:       {len(trades)}")
    print(f"Win rate:     {win_rate:.1f}%")
    print(f"Total R:      {total_r:+.2f}")
    print(f"Avg R/trade:  {total_r / len(trades):+.3f}")
    print(f"Max drawdown: {max_dd:.2f}R")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backtest the gold_signal_engine EMA-crossover strategy against historical data.")
    parser.add_argument("--ticker", default=gse.TICKER)
    parser.add_argument("--interval", default=gse.INTERVAL)
    parser.add_argument("--period", default="60d",
                         help="yfinance lookback window (max ~60d for 15m candles, ~2y for 1h)")
    args = parser.parse_args()

    df = load_data(args.ticker, args.interval, args.period)

    trades_baseline = run_backtest(df, use_filters=False)
    trades_filtered = run_backtest(df, use_filters=True)

    summarize(trades_baseline, "Baseline: raw EMA crossover (no filters)")
    summarize(trades_filtered, "With filters: RSI + MACD + ADX + 1H trend")

    print(
        "\nNote: this only covers the period fetched above -- re-run with a "
        "longer --period (or swap in a paid historical data source, see the "
        "module docstring) before trusting these numbers for real capital."
    )


if __name__ == "__main__":
    main()

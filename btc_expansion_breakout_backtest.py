
"""
btc_expansion_breakout_backtest.py

Goal
----
Quantify BTC "volatility compression -> breakout -> expansion" setups on 1H candles.

Default data source: Bybit public V5 REST API, BTCUSDT linear perpetual, 60-minute candles.
No API key is required.

What the script tests
---------------------
1) Pre-breakout compression:
   - Bollinger Band Width is in a low historical percentile.
   - Recent N-hour range is relatively tight.

2) Breakout:
   - Close breaks above the prior N-hour high.
   - Breakout volume exceeds a multiple of recent average volume.
   - Optional trend filter: close > EMA20 > EMA50.

3) Entry:
   - STANDARD: enter at breakout candle close.
   - RETEST: wait up to RETEST_BARS hours for price to retest the breakout level,
             then enter only if the candle closes back above that level.

4) Exit:
   - Stop loss as a fixed percentage from entry.
   - Profit target as a fixed R multiple.
   - Maximum holding time.
   - Optional trailing exit using prior rolling lows.

Outputs
-------
- btc_breakout_trades.csv
- btc_breakout_yearly.csv
- btc_breakout_threshold_grid.csv
- btc_breakout_summary.txt

Important
---------
This is a research/backtest script, not an execution bot.
It deliberately avoids look-ahead bias by using only information available at each bar.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests


# =========================
# CONFIG
# =========================

SYMBOL = "BTCUSDT"
CATEGORY = "linear"      # Bybit USDT perpetual
INTERVAL = "60"          # 1H
START_DATE = "2021-01-01"
END_DATE = None          # None -> now UTC

# Compression / breakout definition
LOOKBACK_HIGH = 24             # breakout above previous 24h high
RANGE_LOOKBACK = 24            # pre-breakout 24h range
MAX_PRE_RANGE_PCT = 0.030      # previous 24h high-low range <= 3.0%
BB_WINDOW = 20
BB_STD = 2.0
BBW_PERCENTILE_WINDOW = 24 * 30
MAX_BBW_PERCENTILE = 0.25      # BB width in lowest 25% of trailing 30d
VOLUME_MA = 24
MIN_VOLUME_MULT = 1.50
MIN_BREAKOUT_PCT = 0.0005      # close at least 0.05% over prior range high
USE_TREND_FILTER = True
EMA_FAST = 20
EMA_SLOW = 50

# Entry model
ENTRY_MODE = "RETEST"          # "STANDARD" or "RETEST"
RETEST_BARS = 6
RETEST_TOLERANCE = 0.0025      # low can trade 0.25% below breakout level
RETEST_CLOSE_BUFFER = 0.0000   # retest candle must close >= breakout_level*(1+buffer)

# Risk / exit
STOP_PCT = 0.008               # 0.8%
TARGET_R = 3.0                 # target = 3R
MAX_HOLD_BARS = 24             # max 24 hours
USE_TRAILING_STOP = True
TRAIL_LOOKBACK = 4
TRAIL_ACTIVATE_R = 1.5
FEE_RATE_PER_SIDE = 0.00055    # editable approximation; set to your actual rate
SLIPPAGE_PER_SIDE = 0.00010    # 1 bp each side

# Forward expansion labels for event analysis
FORWARD_WINDOWS = [6, 12, 24]
EXPANSION_THRESHOLDS = [0.02, 0.04, 0.05, 0.07]

# Parameter grid
RUN_GRID = True
GRID_VOLUME_MULT = [1.0, 1.25, 1.5, 2.0]
GRID_BBW_PERCENTILE = [0.15, 0.25, 0.35]
GRID_RANGE_PCT = [0.02, 0.03, 0.04]
GRID_STOP_PCT = [0.005, 0.008, 0.010]
GRID_TARGET_R = [2.0, 3.0, 4.0]

OUTPUT_DIR = Path("btc_breakout_results")
OUTPUT_DIR.mkdir(exist_ok=True)


# =========================
# DATA
# =========================

def ts_ms(s: str) -> int:
    dt = datetime.fromisoformat(s).replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def fetch_bybit_klines(
    symbol: str = SYMBOL,
    category: str = CATEGORY,
    interval: str = INTERVAL,
    start_date: str = START_DATE,
    end_date: Optional[str] = END_DATE,
) -> pd.DataFrame:
    """
    Bybit V5 /v5/market/kline.
    Pulls historical candles backwards in chunks and returns ascending timestamps.
    """
    url = "https://api.bybit.com/v5/market/kline"
    start_ms = ts_ms(start_date)
    if end_date:
        end_ms = ts_ms(end_date)
    else:
        end_ms = int(datetime.now(timezone.utc).timestamp() * 1000)

    all_rows = []
    cursor_end = end_ms

    while cursor_end > start_ms:
        params = {
            "category": category,
            "symbol": symbol,
            "interval": interval,
            "end": cursor_end,
            "limit": 1000,
        }
        r = requests.get(url, params=params, timeout=20)
        r.raise_for_status()
        payload = r.json()
        if payload.get("retCode") != 0:
            raise RuntimeError(payload)

        rows = payload["result"]["list"]
        if not rows:
            break

        all_rows.extend(rows)
        oldest = min(int(x[0]) for x in rows)

        if oldest <= start_ms:
            break

        cursor_end = oldest - 1
        time.sleep(0.05)

    if not all_rows:
        raise RuntimeError("No candles returned.")

    cols = ["timestamp", "open", "high", "low", "close", "volume", "turnover"]
    df = pd.DataFrame(all_rows, columns=cols)
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce").astype("int64")
    for c in cols[1:]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
    df = df[df["timestamp"] >= start_ms]
    df = df[df["timestamp"] <= end_ms]
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.set_index("datetime")

    # Ignore a still-open final candle.
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    one_hour_ms = 60 * 60 * 1000
    df = df[df["timestamp"] + one_hour_ms <= now_ms]

    return df


# =========================
# FEATURES
# =========================

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()

    x["ema20"] = x["close"].ewm(span=EMA_FAST, adjust=False).mean()
    x["ema50"] = x["close"].ewm(span=EMA_SLOW, adjust=False).mean()

    mid = x["close"].rolling(BB_WINDOW).mean()
    sd = x["close"].rolling(BB_WINDOW).std(ddof=0)
    upper = mid + BB_STD * sd
    lower = mid - BB_STD * sd
    x["bb_mid"] = mid
    x["bb_upper"] = upper
    x["bb_lower"] = lower
    x["bbw"] = (upper - lower) / mid

    # Percentile rank of current BBW within trailing window.
    def pct_rank_last(a):
        s = pd.Series(a)
        return s.rank(pct=True).iloc[-1]

    x["bbw_pct_rank"] = (
        x["bbw"]
        .rolling(BBW_PERCENTILE_WINDOW, min_periods=max(100, BBW_PERCENTILE_WINDOW // 3))
        .apply(pct_rank_last, raw=False)
    )

    # Shifted highs/lows ensure breakout level excludes current candle.
    x["prior_high"] = x["high"].shift(1).rolling(LOOKBACK_HIGH).max()
    x["prior_low"] = x["low"].shift(1).rolling(RANGE_LOOKBACK).min()
    x["pre_range_pct"] = (x["prior_high"] - x["prior_low"]) / x["prior_low"]

    x["vol_ma"] = x["volume"].shift(1).rolling(VOLUME_MA).mean()
    x["volume_mult"] = x["volume"] / x["vol_ma"]

    x["breakout_pct"] = x["close"] / x["prior_high"] - 1.0

    # Forward event labels, only for descriptive analysis.
    for h in FORWARD_WINDOWS:
        fwd_high = pd.concat([x["high"].shift(-i) for i in range(1, h + 1)], axis=1).max(axis=1)
        fwd_low = pd.concat([x["low"].shift(-i) for i in range(1, h + 1)], axis=1).min(axis=1)
        x[f"fwd_mfe_{h}h"] = fwd_high / x["close"] - 1.0
        x[f"fwd_mae_{h}h"] = fwd_low / x["close"] - 1.0

    return x


def signal_mask(
    x: pd.DataFrame,
    min_volume_mult=MIN_VOLUME_MULT,
    max_bbw_percentile=MAX_BBW_PERCENTILE,
    max_pre_range_pct=MAX_PRE_RANGE_PCT,
) -> pd.Series:
    m = (
        x["prior_high"].notna()
        & x["bbw_pct_rank"].notna()
        & (x["pre_range_pct"] <= max_pre_range_pct)
        & (x["bbw_pct_rank"] <= max_bbw_percentile)
        & (x["volume_mult"] >= min_volume_mult)
        & (x["close"] > x["prior_high"] * (1 + MIN_BREAKOUT_PCT))
    )
    if USE_TREND_FILTER:
        m &= (x["close"] > x["ema20"]) & (x["ema20"] > x["ema50"])
    return m


# =========================
# BACKTEST
# =========================

@dataclass
class Trade:
    signal_time: str
    entry_time: str
    exit_time: str
    breakout_level: float
    entry: float
    stop_initial: float
    target: float
    exit: float
    exit_reason: str
    bars_held: int
    gross_return: float
    net_return: float
    r_multiple_net: float
    mae: float
    mfe: float
    pre_range_pct: float
    bbw_pct_rank: float
    volume_mult: float
    breakout_pct: float


def choose_entry(x: pd.DataFrame, signal_i: int, entry_mode: str):
    sig = x.iloc[signal_i]
    breakout_level = float(sig["prior_high"])

    if entry_mode == "STANDARD":
        return signal_i, float(sig["close"]), breakout_level

    if entry_mode != "RETEST":
        raise ValueError("ENTRY_MODE must be STANDARD or RETEST")

    end_i = min(len(x) - 1, signal_i + RETEST_BARS)
    for j in range(signal_i + 1, end_i + 1):
        row = x.iloc[j]
        touched = row["low"] <= breakout_level * (1 + RETEST_TOLERANCE)
        held = row["close"] >= breakout_level * (1 + RETEST_CLOSE_BUFFER)
        if touched and held:
            # Conservative: enter at candle close after retest confirmation.
            return j, float(row["close"]), breakout_level

    return None


def simulate_trade(
    x: pd.DataFrame,
    signal_i: int,
    stop_pct=STOP_PCT,
    target_r=TARGET_R,
    entry_mode=ENTRY_MODE,
) -> Optional[Trade]:

    picked = choose_entry(x, signal_i, entry_mode)
    if picked is None:
        return None

    entry_i, entry_raw, breakout_level = picked

    # Apply positive slippage to long entry.
    entry = entry_raw * (1 + SLIPPAGE_PER_SIDE)
    stop_initial = entry * (1 - stop_pct)
    risk = entry - stop_initial
    target = entry + target_r * risk

    running_stop = stop_initial
    max_high = entry
    min_low = entry
    exit_price = None
    exit_reason = None
    exit_i = None

    last_i = min(len(x) - 1, entry_i + MAX_HOLD_BARS)

    for j in range(entry_i + 1, last_i + 1):
        row = x.iloc[j]
        hi = float(row["high"])
        lo = float(row["low"])
        cl = float(row["close"])

        max_high = max(max_high, hi)
        min_low = min(min_low, lo)

        # Conservative intrabar convention: if stop and target both hit in same candle,
        # assume stop occurs first.
        if lo <= running_stop:
            exit_price = running_stop * (1 - SLIPPAGE_PER_SIDE)
            exit_reason = "STOP"
            exit_i = j
            break

        if hi >= target:
            exit_price = target * (1 - SLIPPAGE_PER_SIDE)
            exit_reason = "TARGET"
            exit_i = j
            break

        if USE_TRAILING_STOP:
            unrealized_r = (max_high - entry) / risk
            if unrealized_r >= TRAIL_ACTIVATE_R and j - TRAIL_LOOKBACK + 1 >= 0:
                trail_low = float(x["low"].iloc[j - TRAIL_LOOKBACK + 1:j + 1].min())
                # Never loosen stop.
                running_stop = max(running_stop, trail_low)

    if exit_price is None:
        exit_i = last_i
        exit_price = float(x["close"].iloc[exit_i]) * (1 - SLIPPAGE_PER_SIDE)
        exit_reason = "TIME"

    gross_return = exit_price / entry - 1.0
    fees = 2 * FEE_RATE_PER_SIDE
    net_return = gross_return - fees

    mae = min_low / entry - 1.0
    mfe = max_high / entry - 1.0
    r_multiple_net = net_return / stop_pct

    sig = x.iloc[signal_i]

    return Trade(
        signal_time=str(x.index[signal_i]),
        entry_time=str(x.index[entry_i]),
        exit_time=str(x.index[exit_i]),
        breakout_level=breakout_level,
        entry=entry,
        stop_initial=stop_initial,
        target=target,
        exit=exit_price,
        exit_reason=exit_reason,
        bars_held=int(exit_i - entry_i),
        gross_return=gross_return,
        net_return=net_return,
        r_multiple_net=r_multiple_net,
        mae=mae,
        mfe=mfe,
        pre_range_pct=float(sig["pre_range_pct"]),
        bbw_pct_rank=float(sig["bbw_pct_rank"]),
        volume_mult=float(sig["volume_mult"]),
        breakout_pct=float(sig["breakout_pct"]),
    )


def backtest(
    x: pd.DataFrame,
    min_volume_mult=MIN_VOLUME_MULT,
    max_bbw_percentile=MAX_BBW_PERCENTILE,
    max_pre_range_pct=MAX_PRE_RANGE_PCT,
    stop_pct=STOP_PCT,
    target_r=TARGET_R,
    entry_mode=ENTRY_MODE,
) -> pd.DataFrame:

    sigmask = signal_mask(
        x,
        min_volume_mult=min_volume_mult,
        max_bbw_percentile=max_bbw_percentile,
        max_pre_range_pct=max_pre_range_pct,
    )
    sig_indices = np.flatnonzero(sigmask.values)

    trades = []
    blocked_until = -1

    # One live trade at a time.
    for i in sig_indices:
        if i <= blocked_until:
            continue
        t = simulate_trade(
            x,
            signal_i=i,
            stop_pct=stop_pct,
            target_r=target_r,
            entry_mode=entry_mode,
        )
        if t is not None:
            trades.append(asdict(t))
            exit_idx = x.index.get_loc(pd.Timestamp(t.exit_time))
            blocked_until = exit_idx

    return pd.DataFrame(trades)


# =========================
# REPORTING
# =========================

def summarize(trades: pd.DataFrame) -> dict:
    if trades.empty:
        return {
            "trades": 0,
            "avg_net": np.nan,
            "median_net": np.nan,
            "win_rate": np.nan,
            "profit_factor": np.nan,
            "avg_r": np.nan,
            "median_r": np.nan,
            "target_rate": np.nan,
            "stop_rate": np.nan,
            "avg_mae": np.nan,
            "avg_mfe": np.nan,
            "max_drawdown": np.nan,
            "compounded_return": np.nan,
        }

    pnl = trades["net_return"].astype(float)
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]

    pf = wins.sum() / abs(losses.sum()) if len(losses) else np.inf

    equity = (1 + pnl).cumprod()
    dd = equity / equity.cummax() - 1

    return {
        "trades": len(trades),
        "avg_net": pnl.mean(),
        "median_net": pnl.median(),
        "win_rate": (pnl > 0).mean(),
        "profit_factor": pf,
        "avg_r": trades["r_multiple_net"].mean(),
        "median_r": trades["r_multiple_net"].median(),
        "target_rate": (trades["exit_reason"] == "TARGET").mean(),
        "stop_rate": (trades["exit_reason"] == "STOP").mean(),
        "avg_mae": trades["mae"].mean(),
        "avg_mfe": trades["mfe"].mean(),
        "max_drawdown": dd.min(),
        "compounded_return": equity.iloc[-1] - 1,
    }


def yearly_table(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()

    z = trades.copy()
    z["entry_time"] = pd.to_datetime(z["entry_time"], utc=True)
    z["year"] = z["entry_time"].dt.year

    rows = []
    for year, g in z.groupby("year"):
        s = summarize(g)
        s["year"] = year
        rows.append(s)
    return pd.DataFrame(rows).set_index("year").sort_index()


def event_statistics(x: pd.DataFrame) -> pd.DataFrame:
    """
    Descriptive event stats before trade-exit rules.
    Answers: after a qualifying breakout, how often did BTC subsequently expand +2/+4/+5/+7%?
    """
    m = signal_mask(x)
    events = x[m].copy()
    rows = []

    for h in FORWARD_WINDOWS:
        row = {
            "window_hours": h,
            "events": len(events),
            "avg_mfe": events[f"fwd_mfe_{h}h"].mean(),
            "median_mfe": events[f"fwd_mfe_{h}h"].median(),
            "avg_mae": events[f"fwd_mae_{h}h"].mean(),
        }
        for th in EXPANSION_THRESHOLDS:
            row[f"hit_{int(th*100)}pct"] = (events[f"fwd_mfe_{h}h"] >= th).mean()
        rows.append(row)

    return pd.DataFrame(rows)


def run_grid(x: pd.DataFrame) -> pd.DataFrame:
    rows = []
    total = (
        len(GRID_VOLUME_MULT)
        * len(GRID_BBW_PERCENTILE)
        * len(GRID_RANGE_PCT)
        * len(GRID_STOP_PCT)
        * len(GRID_TARGET_R)
    )
    n = 0

    for vm in GRID_VOLUME_MULT:
        for bbw in GRID_BBW_PERCENTILE:
            for rp in GRID_RANGE_PCT:
                for sp in GRID_STOP_PCT:
                    for tr in GRID_TARGET_R:
                        n += 1
                        bt = backtest(
                            x,
                            min_volume_mult=vm,
                            max_bbw_percentile=bbw,
                            max_pre_range_pct=rp,
                            stop_pct=sp,
                            target_r=tr,
                            entry_mode=ENTRY_MODE,
                        )
                        s = summarize(bt)
                        rows.append({
                            "volume_mult": vm,
                            "bbw_percentile": bbw,
                            "max_pre_range_pct": rp,
                            "stop_pct": sp,
                            "target_r": tr,
                            **s,
                        })
                        if n % 20 == 0:
                            print(f"Grid {n}/{total}")

    out = pd.DataFrame(rows)

    # Avoid choosing parameter sets from tiny samples.
    out["score"] = np.where(
        out["trades"] >= 20,
        out["avg_r"] * np.sqrt(out["trades"]) + 0.15 * np.log1p(out["profit_factor"].clip(upper=20)),
        np.nan,
    )
    return out.sort_values(["score", "avg_r", "profit_factor"], ascending=False)


def fmt_pct(v):
    if pd.isna(v):
        return "n/a"
    return f"{v*100:.2f}%"


def main():
    print("Downloading Bybit BTCUSDT 1H candles...")
    df = fetch_bybit_klines()
    print(f"Candles: {len(df):,} | {df.index.min()} -> {df.index.max()}")

    x = add_features(df)

    print("\nRunning baseline backtest...")
    trades = backtest(x)
    s = summarize(trades)
    yearly = yearly_table(trades)
    events = event_statistics(x)

    trades.to_csv(OUTPUT_DIR / "btc_breakout_trades.csv", index=False)
    yearly.to_csv(OUTPUT_DIR / "btc_breakout_yearly.csv")
    events.to_csv(OUTPUT_DIR / "btc_breakout_event_stats.csv", index=False)

    lines = []
    lines.append("BTC EXPANSION / BREAKOUT BACKTEST")
    lines.append("=" * 52)
    lines.append(f"Data: {x.index.min()} -> {x.index.max()}")
    lines.append(f"Entry mode: {ENTRY_MODE}")
    lines.append(f"Lookback high: {LOOKBACK_HIGH}h")
    lines.append(f"Max pre-range: {MAX_PRE_RANGE_PCT:.2%}")
    lines.append(f"Max BBW percentile: {MAX_BBW_PERCENTILE:.0%}")
    lines.append(f"Min volume multiple: {MIN_VOLUME_MULT:.2f}x")
    lines.append(f"Stop: {STOP_PCT:.2%}")
    lines.append(f"Target: {TARGET_R:.1f}R")
    lines.append("")
    lines.append("BASELINE RESULTS")
    lines.append("-" * 52)
    lines.append(f"Trades: {s['trades']}")
    lines.append(f"Average net/trade: {fmt_pct(s['avg_net'])}")
    lines.append(f"Median net/trade: {fmt_pct(s['median_net'])}")
    lines.append(f"Win rate: {fmt_pct(s['win_rate'])}")
    lines.append(f"Profit factor: {s['profit_factor']:.3f}" if not pd.isna(s["profit_factor"]) else "Profit factor: n/a")
    lines.append(f"Average R: {s['avg_r']:.3f}" if not pd.isna(s["avg_r"]) else "Average R: n/a")
    lines.append(f"Target rate: {fmt_pct(s['target_rate'])}")
    lines.append(f"Stop rate: {fmt_pct(s['stop_rate'])}")
    lines.append(f"Average MAE: {fmt_pct(s['avg_mae'])}")
    lines.append(f"Average MFE: {fmt_pct(s['avg_mfe'])}")
    lines.append(f"Compounded return*: {fmt_pct(s['compounded_return'])}")
    lines.append(f"Max drawdown*: {fmt_pct(s['max_drawdown'])}")
    lines.append("")
    lines.append("* Compounded return assumes 100% notional allocated to each non-overlapping trade.")
    lines.append("  Use average R / PF / drawdown and a realistic risk-per-trade model for production sizing.")
    lines.append("")
    lines.append("FORWARD EXPANSION EVENT STATS")
    lines.append(events.to_string(index=False))
    lines.append("")
    lines.append("YEARLY STABILITY")
    lines.append(yearly.to_string())

    if RUN_GRID:
        print("\nRunning parameter grid...")
        grid = run_grid(x)
        grid.to_csv(OUTPUT_DIR / "btc_breakout_threshold_grid.csv", index=False)
        lines.append("")
        lines.append("TOP 20 PARAMETER SETS (minimum 20 trades for score)")
        lines.append(grid.head(20).to_string(index=False))

    report = "\n".join(lines)
    (OUTPUT_DIR / "btc_breakout_summary.txt").write_text(report, encoding="utf-8")
    print("\n" + report)
    print(f"\nSaved outputs to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()

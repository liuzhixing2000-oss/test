
"""
btc_breakout_success_failure_study.py

Research question
-----------------
Among BTCUSDT 1H breakout events, what distinguishes successful trend-expansion
breakouts from failed breakouts?

Design principles
-----------------
- Do NOT optimize the old trade parameters again.
- Build a broad event set from a simple, locked breakout definition.
- Label outcomes AFTER the event using future path information.
- Compare pre-event and event-time features only.
- Split chronologically:
    Discovery: 2021-2024
    OOS:       2025-2026
- Rank features by:
    1) effect size / monotonic separation in discovery
    2) same-direction separation in OOS
- Export event-level data so the user can inspect exact historical examples.

Data
----
Bybit BTCUSDT linear perpetual 1H public candles.
No API key required.

Outputs
-------
btc_sf_summary.txt
btc_sf_events.csv
btc_sf_feature_comparison_discovery.csv
btc_sf_feature_comparison_oos.csv
btc_sf_feature_stability.csv
btc_sf_yearly.csv
btc_sf_recent_events.csv
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd
import requests

OUT = Path("btc_sf_results")
OUT.mkdir(exist_ok=True)

SYMBOL = "BTCUSDT"
CATEGORY = "linear"
INTERVAL = "60"
START_DATE = "2021-01-01"

# Event definition: intentionally broader than the prior production candidate.
LOOKBACK_HIGH = 24
MAX_PRE_RANGE_PCT = 0.035
MIN_VOLUME_MULT = 1.25
MIN_BREAKOUT_PCT = 0.0005

# Outcome label horizon / thresholds
FORWARD_HOURS = 24
STOP_PCT = 0.01
SUCCESS_R = 3.0
SUCCESS_PCT = STOP_PCT * SUCCESS_R  # +3%
FAIL_PCT = STOP_PCT                 # -1%

DISCOVERY_END = pd.Timestamp("2025-01-01", tz="UTC")

# Feature windows
BBW_WINDOW = 20
BBW_STD = 2.0
BBW_RANK_WINDOW = 24 * 30
ATR_WINDOW = 14
VOL_MA = 24
EMA_WINDOWS = [20, 50, 100, 200]

FEATURES = [
    # context
    "pre_range_pct_24h",
    "pre_range_pct_48h",
    "ret_6h",
    "ret_12h",
    "ret_24h",
    "ret_48h",
    "distance_from_ema20",
    "distance_from_ema50",
    "distance_from_ema100",
    "distance_from_ema200",
    "ema20_slope_6h",
    "ema50_slope_12h",
    "trend_stack_score",
    # volatility
    "atr_pct",
    "atr_pct_rank_30d",
    "bbw",
    "bbw_pct_rank_30d",
    "compression_hours_bbw35",
    # breakout candle
    "volume_mult",
    "breakout_pct",
    "body_pct",
    "upper_wick_ratio",
    "lower_wick_ratio",
    "close_location",
    "range_atr",
    # momentum
    "rsi14",
    "macd_hist_norm",
    # HTF proxies derived from 1H
    "ret_4h",
    "ret_1d",
    "ema20_above_ema50",
    "ema50_above_ema200",
]


def ts_ms(s: str) -> int:
    dt = datetime.fromisoformat(s).replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def fetch_bybit_klines(start_date=START_DATE, end_date: Optional[str] = None) -> pd.DataFrame:
    url = "https://api.bybit.com/v5/market/kline"
    start_ms = ts_ms(start_date)
    end_ms = ts_ms(end_date) if end_date else int(datetime.now(timezone.utc).timestamp() * 1000)

    rows_all = []
    cursor_end = end_ms

    while cursor_end > start_ms:
        params = {
            "category": CATEGORY,
            "symbol": SYMBOL,
            "interval": INTERVAL,
            "end": cursor_end,
            "limit": 1000,
        }
        r = requests.get(url, params=params, timeout=20)
        r.raise_for_status()
        js = r.json()
        if js.get("retCode") != 0:
            raise RuntimeError(js)

        rows = js["result"]["list"]
        if not rows:
            break

        rows_all.extend(rows)
        oldest = min(int(x[0]) for x in rows)
        if oldest <= start_ms:
            break
        cursor_end = oldest - 1
        time.sleep(0.05)

    cols = ["timestamp", "open", "high", "low", "close", "volume", "turnover"]
    df = pd.DataFrame(rows_all, columns=cols)
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce").astype("int64")
    for c in cols[1:]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.drop_duplicates("timestamp").sort_values("timestamp")
    df = df[df["timestamp"] >= start_ms]
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.set_index("datetime")

    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    one_h = 3600 * 1000
    df = df[df["timestamp"] + one_h <= now_ms]
    return df


def rolling_percentile_last(s: pd.Series, window: int, min_periods: int = 100) -> pd.Series:
    def f(a):
        z = pd.Series(a)
        return z.rank(pct=True).iloc[-1]
    return s.rolling(window, min_periods=min_periods).apply(f, raw=False)


def rsi(close: pd.Series, n=14):
    d = close.diff()
    gain = d.clip(lower=0)
    loss = -d.clip(upper=0)
    ag = gain.ewm(alpha=1/n, adjust=False).mean()
    al = loss.ewm(alpha=1/n, adjust=False).mean()
    rs = ag / al.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()

    # EMAs
    for n in EMA_WINDOWS:
        x[f"ema{n}"] = x["close"].ewm(span=n, adjust=False).mean()

    # Returns
    for h in [4, 6, 12, 24, 48]:
        x[f"ret_{h}h"] = x["close"] / x["close"].shift(h) - 1
    x["ret_1d"] = x["ret_24h"]

    # ATR
    prev_close = x["close"].shift(1)
    tr = pd.concat([
        x["high"] - x["low"],
        (x["high"] - prev_close).abs(),
        (x["low"] - prev_close).abs()
    ], axis=1).max(axis=1)
    atr = tr.rolling(ATR_WINDOW).mean()
    x["atr"] = atr
    x["atr_pct"] = atr / x["close"]
    x["atr_pct_rank_30d"] = rolling_percentile_last(x["atr_pct"], 24*30, 150)

    # Bollinger
    bb_mid = x["close"].rolling(BBW_WINDOW).mean()
    bb_sd = x["close"].rolling(BBW_WINDOW).std(ddof=0)
    bb_up = bb_mid + BBW_STD * bb_sd
    bb_dn = bb_mid - BBW_STD * bb_sd
    x["bbw"] = (bb_up - bb_dn) / bb_mid
    x["bbw_pct_rank_30d"] = rolling_percentile_last(x["bbw"], BBW_RANK_WINDOW, 150)

    # Compression duration: consecutive prior hours with BBW rank <=35%.
    flag = (x["bbw_pct_rank_30d"] <= 0.35).astype(int)
    groups = (flag != flag.shift()).cumsum()
    runlen = flag.groupby(groups).cumsum()
    x["compression_hours_bbw35"] = runlen.where(flag.eq(1), 0).shift(1).fillna(0)

    # Prior ranges and breakout level, shifted to avoid look-ahead.
    x["prior_high_24h"] = x["high"].shift(1).rolling(24).max()
    x["prior_low_24h"] = x["low"].shift(1).rolling(24).min()
    x["pre_range_pct_24h"] = (x["prior_high_24h"] - x["prior_low_24h"]) / x["prior_low_24h"]

    x["prior_high_48h"] = x["high"].shift(1).rolling(48).max()
    x["prior_low_48h"] = x["low"].shift(1).rolling(48).min()
    x["pre_range_pct_48h"] = (x["prior_high_48h"] - x["prior_low_48h"]) / x["prior_low_48h"]

    # Volume / breakout candle anatomy
    x["vol_ma24"] = x["volume"].shift(1).rolling(VOL_MA).mean()
    x["volume_mult"] = x["volume"] / x["vol_ma24"]
    x["breakout_pct"] = x["close"] / x["prior_high_24h"] - 1

    candle_range = (x["high"] - x["low"]).replace(0, np.nan)
    body = (x["close"] - x["open"]).abs()
    upper_wick = x["high"] - x[["open", "close"]].max(axis=1)
    lower_wick = x[["open", "close"]].min(axis=1) - x["low"]
    x["body_pct"] = body / x["open"]
    x["upper_wick_ratio"] = upper_wick / candle_range
    x["lower_wick_ratio"] = lower_wick / candle_range
    x["close_location"] = (x["close"] - x["low"]) / candle_range
    x["range_atr"] = candle_range / x["atr"]

    # Distances / slopes / stack
    for n in [20, 50, 100, 200]:
        x[f"distance_from_ema{n}"] = x["close"] / x[f"ema{n}"] - 1
    x["ema20_slope_6h"] = x["ema20"] / x["ema20"].shift(6) - 1
    x["ema50_slope_12h"] = x["ema50"] / x["ema50"].shift(12) - 1
    x["ema20_above_ema50"] = (x["ema20"] > x["ema50"]).astype(int)
    x["ema50_above_ema200"] = (x["ema50"] > x["ema200"]).astype(int)
    x["trend_stack_score"] = (
        (x["close"] > x["ema20"]).astype(int)
        + (x["ema20"] > x["ema50"]).astype(int)
        + (x["ema50"] > x["ema100"]).astype(int)
        + (x["ema100"] > x["ema200"]).astype(int)
    )

    # RSI / MACD
    x["rsi14"] = rsi(x["close"], 14)
    ema12 = x["close"].ewm(span=12, adjust=False).mean()
    ema26 = x["close"].ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    hist = macd - signal
    x["macd_hist_norm"] = hist / x["close"]

    return x


def event_mask(x: pd.DataFrame) -> pd.Series:
    return (
        x["prior_high_24h"].notna()
        & x["vol_ma24"].notna()
        & (x["pre_range_pct_24h"] <= MAX_PRE_RANGE_PCT)
        & (x["volume_mult"] >= MIN_VOLUME_MULT)
        & (x["close"] > x["prior_high_24h"] * (1 + MIN_BREAKOUT_PCT))
    )


def label_events(x: pd.DataFrame) -> pd.DataFrame:
    mask = event_mask(x)
    idxs = np.flatnonzero(mask.values)
    rows = []

    for i in idxs:
        if i + FORWARD_HOURS >= len(x):
            continue

        entry = float(x["close"].iloc[i])
        stop = entry * (1 - FAIL_PCT)
        success = entry * (1 + SUCCESS_PCT)

        future = x.iloc[i+1:i+1+FORWARD_HOURS]
        outcome = "NEUTRAL"
        outcome_bar = None
        mfe = future["high"].max() / entry - 1
        mae = future["low"].min() / entry - 1

        # Path-aware first-touch labeling
        for k, (_, r) in enumerate(future.iterrows(), start=1):
            hit_stop = r["low"] <= stop
            hit_success = r["high"] >= success
            if hit_stop and hit_success:
                outcome = "FAILURE"  # conservative same-bar assumption
                outcome_bar = k
                break
            if hit_stop:
                outcome = "FAILURE"
                outcome_bar = k
                break
            if hit_success:
                outcome = "SUCCESS"
                outcome_bar = k
                break

        row = {
            "event_time": x.index[i],
            "entry_close": entry,
            "outcome": outcome,
            "outcome_bar": outcome_bar,
            "fwd_mfe_24h": float(mfe),
            "fwd_mae_24h": float(mae),
            "success_threshold": SUCCESS_PCT,
            "failure_threshold": -FAIL_PCT,
        }
        for f in FEATURES:
            row[f] = x.iloc[i].get(f, np.nan)

        rows.append(row)

    ev = pd.DataFrame(rows)
    if not ev.empty:
        ev["event_time"] = pd.to_datetime(ev["event_time"], utc=True)
        ev["sample"] = np.where(ev["event_time"] < DISCOVERY_END, "DISCOVERY_2021_2024", "OOS_2025_2026")
        ev["year"] = ev["event_time"].dt.year
        ev["quarter"] = ev["event_time"].dt.to_period("Q").astype(str)
    return ev


def compare_features(ev: pd.DataFrame, sample_name: str) -> pd.DataFrame:
    d = ev[(ev["sample"] == sample_name) & (ev["outcome"].isin(["SUCCESS", "FAILURE"]))].copy()
    rows = []
    if d.empty:
        return pd.DataFrame()

    succ = d[d["outcome"] == "SUCCESS"]
    fail = d[d["outcome"] == "FAILURE"]

    for f in FEATURES:
        a = pd.to_numeric(succ[f], errors="coerce").dropna()
        b = pd.to_numeric(fail[f], errors="coerce").dropna()
        if len(a) < 2 or len(b) < 2:
            continue

        mean_s = a.mean()
        mean_f = b.mean()
        pooled = math.sqrt(((len(a)-1)*a.var(ddof=1) + (len(b)-1)*b.var(ddof=1)) / max(len(a)+len(b)-2, 1))
        d_eff = (mean_s - mean_f) / pooled if pooled and np.isfinite(pooled) else np.nan

        rows.append({
            "feature": f,
            "n_success": len(a),
            "n_failure": len(b),
            "success_mean": mean_s,
            "failure_mean": mean_f,
            "difference": mean_s - mean_f,
            "cohens_d": d_eff,
            "success_median": a.median(),
            "failure_median": b.median(),
        })

    out = pd.DataFrame(rows)
    if not out.empty:
        out["abs_cohens_d"] = out["cohens_d"].abs()
        out = out.sort_values("abs_cohens_d", ascending=False)
    return out


def stability_table(discovery: pd.DataFrame, oos: pd.DataFrame) -> pd.DataFrame:
    if discovery.empty or oos.empty:
        return pd.DataFrame()

    a = discovery.set_index("feature")
    b = oos.set_index("feature")
    common = a.index.intersection(b.index)

    rows = []
    for f in common:
        d1 = a.loc[f, "cohens_d"]
        d2 = b.loc[f, "cohens_d"]
        rows.append({
            "feature": f,
            "discovery_cohens_d": d1,
            "oos_cohens_d": d2,
            "same_direction": bool(np.sign(d1) == np.sign(d2)) if pd.notna(d1) and pd.notna(d2) else False,
            "discovery_abs_d": abs(d1) if pd.notna(d1) else np.nan,
            "oos_abs_d": abs(d2) if pd.notna(d2) else np.nan,
            "stability_score": min(abs(d1), abs(d2)) if pd.notna(d1) and pd.notna(d2) and np.sign(d1)==np.sign(d2) else 0.0,
        })

    return pd.DataFrame(rows).sort_values(["stability_score", "discovery_abs_d"], ascending=False)


def yearly_summary(ev: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for y, g in ev.groupby("year"):
        labeled = g[g["outcome"].isin(["SUCCESS","FAILURE"])]
        s = (labeled["outcome"] == "SUCCESS").sum()
        f = (labeled["outcome"] == "FAILURE").sum()
        n = s + f
        rows.append({
            "year": y,
            "events_total": len(g),
            "success": s,
            "failure": f,
            "neutral": (g["outcome"] == "NEUTRAL").sum(),
            "success_rate_labeled": s / n if n else np.nan,
            "avg_mfe_24h": g["fwd_mfe_24h"].mean(),
            "avg_mae_24h": g["fwd_mae_24h"].mean(),
        })
    return pd.DataFrame(rows).set_index("year")


def main():
    print("Downloading Bybit BTCUSDT 1H candles...", flush=True)
    df = fetch_bybit_klines()
    print(f"Candles: {len(df):,} | {df.index.min()} -> {df.index.max()}", flush=True)

    print("Building features...", flush=True)
    x = add_features(df)

    print("Extracting and labeling breakout events...", flush=True)
    ev = label_events(x)
    ev.to_csv(OUT / "btc_sf_events.csv", index=False)

    discovery = compare_features(ev, "DISCOVERY_2021_2024")
    oos = compare_features(ev, "OOS_2025_2026")
    discovery.to_csv(OUT / "btc_sf_feature_comparison_discovery.csv", index=False)
    oos.to_csv(OUT / "btc_sf_feature_comparison_oos.csv", index=False)

    stability = stability_table(discovery, oos)
    stability.to_csv(OUT / "btc_sf_feature_stability.csv", index=False)

    yearly = yearly_summary(ev)
    yearly.to_csv(OUT / "btc_sf_yearly.csv")

    latest_cut = ev["event_time"].max() - pd.Timedelta(days=14) if not ev.empty else pd.Timestamp("1970-01-01", tz="UTC")
    recent = ev[ev["event_time"] >= latest_cut].copy() if not ev.empty else pd.DataFrame()
    recent.to_csv(OUT / "btc_sf_recent_events.csv", index=False)

    labeled = ev[ev["outcome"].isin(["SUCCESS","FAILURE"])]
    dlab = labeled[labeled["sample"]=="DISCOVERY_2021_2024"]
    olab = labeled[labeled["sample"]=="OOS_2025_2026"]

    def rate(z):
        return (z["outcome"]=="SUCCESS").mean() if len(z) else np.nan

    lines = []
    lines.append("BTC BREAKOUT SUCCESS vs FAILURE STUDY")
    lines.append("="*72)
    lines.append("Broad locked event definition:")
    lines.append(f"- prior 24h range <= {MAX_PRE_RANGE_PCT:.2%}")
    lines.append(f"- breakout volume >= {MIN_VOLUME_MULT:.2f}x prior 24h mean")
    lines.append(f"- close > prior 24h high by at least {MIN_BREAKOUT_PCT:.2%}")
    lines.append("")
    lines.append("Outcome labels within 24h:")
    lines.append(f"- SUCCESS: +{SUCCESS_PCT:.2%} touched before -{FAIL_PCT:.2%}")
    lines.append(f"- FAILURE: -{FAIL_PCT:.2%} touched before +{SUCCESS_PCT:.2%}")
    lines.append("- NEUTRAL: neither threshold touched within 24h")
    lines.append("")
    lines.append(f"All events: {len(ev)}")
    lines.append(f"Labeled discovery events 2021-2024: {len(dlab)} | success rate {rate(dlab):.2%}" if len(dlab) else "No discovery labeled events")
    lines.append(f"Labeled OOS events 2025-2026: {len(olab)} | success rate {rate(olab):.2%}" if len(olab) else "No OOS labeled events")
    lines.append("")
    lines.append("TOP DISCOVERY FEATURES BY |COHEN'S D|")
    lines.append(discovery.head(15).to_string(index=False) if not discovery.empty else "None")
    lines.append("")
    lines.append("TOP OOS FEATURES BY |COHEN'S D|")
    lines.append(oos.head(15).to_string(index=False) if not oos.empty else "None")
    lines.append("")
    lines.append("FEATURES WITH SAME-DIRECTION SEPARATION IN DISCOVERY AND OOS")
    stable = stability[stability["same_direction"]].head(20) if not stability.empty else pd.DataFrame()
    lines.append(stable.to_string(index=False) if not stable.empty else "None")
    lines.append("")
    lines.append("YEARLY EVENT STABILITY")
    lines.append(yearly.to_string())
    lines.append("")
    lines.append("RECENT EVENTS (last 14 days)")
    lines.append(recent[["event_time","entry_close","outcome","fwd_mfe_24h","fwd_mae_24h","volume_mult","breakout_pct","pre_range_pct_24h"]].to_string(index=False) if not recent.empty else "None")
    lines.append("")
    lines.append("Interpretation:")
    lines.append("- Discovery-only separation is not enough.")
    lines.append("- Prefer features with the SAME direction in OOS and non-trivial OOS effect size.")
    lines.append("- This script does not convert those features into a new trading rule yet.")

    report = "\n".join(lines)
    (OUT / "btc_sf_summary.txt").write_text(report, encoding="utf-8")
    print("\n" + report, flush=True)
    print(f"\nSaved outputs to {OUT.resolve()}", flush=True)


if __name__ == "__main__":
    main()

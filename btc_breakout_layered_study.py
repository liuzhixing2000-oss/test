
"""
BTC Breakout Layered Probability Study
======================================

Purpose
-------
Turn the previous success-vs-failure findings into decision-useful conditional
probabilities WITHOUT optimizing a trading strategy.

Broad event definition (locked from prior study):
- BTCUSDT 1H
- prior 24h range <= 3.5%
- breakout volume >= 1.25x prior 24h mean
- close > prior 24h high by >= 0.05%

Main questions
--------------
1. How do compression duration and HTF trend filters change forward upside odds?
2. What are the 24h hit rates for +1%, +1.5%, +2%, +3%, +4%?
3. How often does -1% occur before each upside target?
4. What MAE/MFE is typical?
5. Do the relationships survive OOS (2025-2026)?

No parameter combination is selected as a "winner" by the script.
"""

from __future__ import annotations
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests

OUT = Path("btc_layered_results")
OUT.mkdir(exist_ok=True)

SYMBOL = "BTCUSDT"
CATEGORY = "linear"
INTERVAL = "60"
START_DATE = "2021-01-01"

LOOKBACK_HIGH = 24
MAX_PRE_RANGE_PCT = 0.035
MIN_VOLUME_MULT = 1.25
MIN_BREAKOUT_PCT = 0.0005
FORWARD_HOURS = 24

TARGETS = [0.01, 0.015, 0.02, 0.03, 0.04]
STOP = 0.01

DISCOVERY_END = pd.Timestamp("2025-01-01", tz="UTC")

COMPRESSION_CUTS = [0, 12, 18, 24]
RANGE48_CUTS = [None, 0.03]
TREND_STACK_CUTS = [None, 3, 4]
EMA50_200_OPTIONS = [None, True]

def ts_ms(s):
    return int(datetime.fromisoformat(s).replace(tzinfo=timezone.utc).timestamp() * 1000)

def fetch():
    url = "https://api.bybit.com/v5/market/kline"
    start_ms = ts_ms(START_DATE)
    end_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    rows_all = []
    cursor = end_ms
    while cursor > start_ms:
        params = {"category": CATEGORY, "symbol": SYMBOL, "interval": INTERVAL,
                  "end": cursor, "limit": 1000}
        r = requests.get(url, params=params, timeout=20)
        r.raise_for_status()
        js = r.json()
        if js.get("retCode") != 0:
            raise RuntimeError(js)
        rows = js["result"]["list"]
        if not rows:
            break
        rows_all.extend(rows)
        oldest = min(int(z[0]) for z in rows)
        if oldest <= start_ms:
            break
        cursor = oldest - 1
        time.sleep(0.05)

    cols = ["timestamp","open","high","low","close","volume","turnover"]
    df = pd.DataFrame(rows_all, columns=cols)
    df["timestamp"] = pd.to_numeric(df["timestamp"]).astype("int64")
    for c in cols[1:]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.drop_duplicates("timestamp").sort_values("timestamp")
    df = df[df["timestamp"] >= start_ms]
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.set_index("datetime")
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    df = df[df["timestamp"] + 3600_000 <= now_ms]
    return df

def pct_rank(s, window=720, minp=150):
    def f(a):
        z = pd.Series(a)
        return z.rank(pct=True).iloc[-1]
    return s.rolling(window, min_periods=minp).apply(f, raw=False)

def features(df):
    x = df.copy()
    for n in [20,50,100,200]:
        x[f"ema{n}"] = x["close"].ewm(span=n, adjust=False).mean()

    mid = x["close"].rolling(20).mean()
    sd = x["close"].rolling(20).std(ddof=0)
    x["bbw"] = ((mid + 2*sd) - (mid - 2*sd)) / mid
    x["bbw_rank"] = pct_rank(x["bbw"])

    lowflag = (x["bbw_rank"] <= 0.35).astype(int)
    grp = (lowflag != lowflag.shift()).cumsum()
    run = lowflag.groupby(grp).cumsum()
    x["compression_hours"] = run.where(lowflag.eq(1), 0).shift(1).fillna(0)

    x["prior_high24"] = x["high"].shift(1).rolling(24).max()
    x["prior_low24"] = x["low"].shift(1).rolling(24).min()
    x["range24"] = (x["prior_high24"] - x["prior_low24"]) / x["prior_low24"]

    x["prior_high48"] = x["high"].shift(1).rolling(48).max()
    x["prior_low48"] = x["low"].shift(1).rolling(48).min()
    x["range48"] = (x["prior_high48"] - x["prior_low48"]) / x["prior_low48"]

    x["vol_ma24"] = x["volume"].shift(1).rolling(24).mean()
    x["volume_mult"] = x["volume"] / x["vol_ma24"]
    x["breakout_pct"] = x["close"] / x["prior_high24"] - 1

    x["trend_stack"] = (
        (x["close"] > x["ema20"]).astype(int)
        + (x["ema20"] > x["ema50"]).astype(int)
        + (x["ema50"] > x["ema100"]).astype(int)
        + (x["ema100"] > x["ema200"]).astype(int)
    )
    x["ema50_gt_200"] = x["ema50"] > x["ema200"]

    prev = x["close"].shift(1)
    tr = pd.concat([(x["high"]-x["low"]),
                    (x["high"]-prev).abs(),
                    (x["low"]-prev).abs()], axis=1).max(axis=1)
    x["atr14"] = tr.rolling(14).mean()
    x["atr_pct"] = x["atr14"] / x["close"]
    return x

def event_mask(x):
    return (
        x["prior_high24"].notna()
        & (x["range24"] <= MAX_PRE_RANGE_PCT)
        & (x["volume_mult"] >= MIN_VOLUME_MULT)
        & (x["close"] > x["prior_high24"] * (1 + MIN_BREAKOUT_PCT))
    )

def build_events(x):
    ids = np.flatnonzero(event_mask(x).values)
    rows = []
    for i in ids:
        if i + FORWARD_HOURS >= len(x):
            continue
        entry = float(x["close"].iloc[i])
        future = x.iloc[i+1:i+1+FORWARD_HOURS]

        row = {
            "event_time": x.index[i],
            "entry": entry,
            "compression_hours": float(x["compression_hours"].iloc[i]),
            "range24": float(x["range24"].iloc[i]),
            "range48": float(x["range48"].iloc[i]),
            "volume_mult": float(x["volume_mult"].iloc[i]),
            "breakout_pct": float(x["breakout_pct"].iloc[i]),
            "trend_stack": int(x["trend_stack"].iloc[i]),
            "ema50_gt_200": bool(x["ema50_gt_200"].iloc[i]),
            "atr_pct": float(x["atr_pct"].iloc[i]),
            "mfe24": float(future["high"].max()/entry - 1),
            "mae24": float(future["low"].min()/entry - 1),
        }

        # For each target: raw hit, stop hit, and first-touch target-before-stop.
        for t in TARGETS:
            tag = str(t*100).replace(".","p")
            row[f"hit_up_{tag}"] = bool((future["high"] >= entry*(1+t)).any())
            row[f"hit_down_1p0"] = bool((future["low"] <= entry*(1-STOP)).any())

            first = "NEITHER"
            first_bar = np.nan
            for k, (_, r) in enumerate(future.iterrows(), start=1):
                hs = r["low"] <= entry*(1-STOP)
                ht = r["high"] >= entry*(1+t)
                if hs and ht:
                    first = "STOP"  # conservative same-bar assumption
                    first_bar = k
                    break
                if hs:
                    first = "STOP"
                    first_bar = k
                    break
                if ht:
                    first = "TARGET"
                    first_bar = k
                    break
            row[f"first_{tag}"] = first
            row[f"first_bar_{tag}"] = first_bar

        rows.append(row)

    ev = pd.DataFrame(rows)
    ev["event_time"] = pd.to_datetime(ev["event_time"], utc=True)
    ev["sample"] = np.where(ev["event_time"] < DISCOVERY_END,
                            "DISCOVERY_2021_2024", "OOS_2025_2026")
    ev["year"] = ev["event_time"].dt.year
    return ev

def summarize_group(g, label, sample, comp, emaopt, stackcut, range48cut):
    row = {
        "sample": sample,
        "filter_label": label,
        "compression_min_h": comp,
        "ema50_gt_200_required": emaopt,
        "trend_stack_min": stackcut,
        "range48_max": range48cut,
        "events": len(g),
        "avg_mfe24": g["mfe24"].mean() if len(g) else np.nan,
        "median_mfe24": g["mfe24"].median() if len(g) else np.nan,
        "avg_mae24": g["mae24"].mean() if len(g) else np.nan,
        "median_mae24": g["mae24"].median() if len(g) else np.nan,
        "p_down_1pct": (g["mae24"] <= -0.01).mean() if len(g) else np.nan,
    }
    for t in TARGETS:
        tag = str(t*100).replace(".","p")
        row[f"p_hit_up_{tag}"] = g[f"hit_up_{tag}"].mean() if len(g) else np.nan
        row[f"p_target_before_stop_{tag}"] = (g[f"first_{tag}"]=="TARGET").mean() if len(g) else np.nan
        row[f"p_stop_before_target_{tag}"] = (g[f"first_{tag}"]=="STOP").mean() if len(g) else np.nan
    return row

def layered(ev):
    rows = []
    for sample in ["DISCOVERY_2021_2024","OOS_2025_2026"]:
        s = ev[ev["sample"]==sample]
        # Baseline
        rows.append(summarize_group(s, "BASELINE", sample, 0, None, None, None))

        for comp in COMPRESSION_CUTS[1:]:
            for emaopt in EMA50_200_OPTIONS:
                for stackcut in TREND_STACK_CUTS:
                    for r48 in RANGE48_CUTS:
                        g = s[s["compression_hours"] >= comp]
                        parts = [f"comp>={comp}h"]
                        if emaopt is True:
                            g = g[g["ema50_gt_200"]]
                            parts.append("EMA50>EMA200")
                        if stackcut is not None:
                            g = g[g["trend_stack"] >= stackcut]
                            parts.append(f"stack>={stackcut}")
                        if r48 is not None:
                            g = g[g["range48"] <= r48]
                            parts.append(f"range48<={r48:.0%}")
                        rows.append(summarize_group(
                            g, " + ".join(parts), sample, comp, emaopt, stackcut, r48
                        ))
    return pd.DataFrame(rows)

def paired_stability(layer):
    d = layer[layer["sample"]=="DISCOVERY_2021_2024"].set_index("filter_label")
    o = layer[layer["sample"]=="OOS_2025_2026"].set_index("filter_label")
    common = d.index.intersection(o.index)
    rows = []
    for lab in common:
        rd, ro = d.loc[lab], o.loc[lab]
        # duplicate labels can arise from None dimensions; retain first if needed
        if isinstance(rd, pd.DataFrame): rd = rd.iloc[0]
        if isinstance(ro, pd.DataFrame): ro = ro.iloc[0]
        row = {
            "filter_label": lab,
            "discovery_events": rd["events"],
            "oos_events": ro["events"],
            "discovery_avg_mfe24": rd["avg_mfe24"],
            "oos_avg_mfe24": ro["avg_mfe24"],
            "discovery_avg_mae24": rd["avg_mae24"],
            "oos_avg_mae24": ro["avg_mae24"],
        }
        for t in TARGETS:
            tag = str(t*100).replace(".","p")
            row[f"discovery_tb4s_{tag}"] = rd[f"p_target_before_stop_{tag}"]
            row[f"oos_tb4s_{tag}"] = ro[f"p_target_before_stop_{tag}"]
        rows.append(row)
    return pd.DataFrame(rows)

def yearly(ev):
    rows = []
    for y,g in ev.groupby("year"):
        r = {"year":y, "events":len(g), "avg_mfe24":g["mfe24"].mean(),
             "avg_mae24":g["mae24"].mean(), "p_down_1pct":(g["mae24"]<=-0.01).mean()}
        for t in TARGETS:
            tag = str(t*100).replace(".","p")
            r[f"p_hit_{tag}"] = g[f"hit_up_{tag}"].mean()
            r[f"p_target_before_stop_{tag}"] = (g[f"first_{tag}"]=="TARGET").mean()
        rows.append(r)
    return pd.DataFrame(rows).set_index("year")

def main():
    print("Downloading BTCUSDT 1H...", flush=True)
    df = fetch()
    print(f"Candles: {len(df):,} | {df.index.min()} -> {df.index.max()}", flush=True)
    x = features(df)
    ev = build_events(x)
    layer = layered(ev)
    stable = paired_stability(layer)
    yr = yearly(ev)

    ev.to_csv(OUT/"btc_layered_events.csv", index=False)
    layer.to_csv(OUT/"btc_layered_probabilities.csv", index=False)
    stable.to_csv(OUT/"btc_layered_oos_stability.csv", index=False)
    yr.to_csv(OUT/"btc_layered_yearly.csv")

    # Useful OOS shortlist: descriptive only, NOT declared optimal.
    oos = layer[(layer["sample"]=="OOS_2025_2026") & (layer["events"]>=15)].copy()
    oos = oos.sort_values(["p_target_before_stop_2p0","events"], ascending=[False,False])
    oos.head(30).to_csv(OUT/"btc_layered_oos_descriptive_shortlist.csv", index=False)

    recent = ev[ev["event_time"] >= ev["event_time"].max()-pd.Timedelta(days=14)]
    recent.to_csv(OUT/"btc_layered_recent_events.csv", index=False)

    lines = [
        "BTC BREAKOUT LAYERED PROBABILITY STUDY",
        "="*72,
        f"All events: {len(ev)}",
        f"Discovery events: {(ev['sample']=='DISCOVERY_2021_2024').sum()}",
        f"OOS events: {(ev['sample']=='OOS_2025_2026').sum()}",
        "",
        "Targets: +1%, +1.5%, +2%, +3%, +4% within 24h.",
        "For each target the study also measures TARGET-before-(-1% stop).",
        "",
        "OOS BASELINE:",
        layer[(layer['sample']=='OOS_2025_2026') & (layer['filter_label']=='BASELINE')].to_string(index=False),
        "",
        "OOS DESCRIPTIVE SHORTLIST (minimum 15 events; sorted by +2% target-before-stop):",
        oos.head(20).to_string(index=False),
        "",
        "YEARLY:",
        yr.to_string(),
        "",
        "IMPORTANT: This is a conditional-probability study, not a newly optimized production strategy.",
        "Any candidate rule must be locked and validated again on later/unseen data before use."
    ]
    report = "\n".join(lines)
    (OUT/"btc_layered_summary.txt").write_text(report, encoding="utf-8")
    print(report, flush=True)
    print(f"Saved to {OUT.resolve()}", flush=True)

if __name__ == "__main__":
    main()

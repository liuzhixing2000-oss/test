"""
btc_breakout_oos_validation.py

Locked-parameter out-of-sample validation for the BTC Expansion / Breakout model.

The locked candidate comes from the prior 324-grid search:
    volume_mult >= 2.0
    BBW percentile <= 35%
    pre-breakout 24h range <= 2%
    stop = 1.0%
    target = 4R
    entry mode = RETEST

This script DOES NOT re-optimize those parameters.

Outputs:
- btc_oos_summary.txt
- btc_oos_period_results.csv
- btc_oos_yearly.csv
- btc_oos_quarterly.csv
- btc_oos_bootstrap.csv
- btc_oos_trades.csv
- btc_latest_signal_check.csv
- btc_latest_signal_context.csv
"""

from __future__ import annotations

import math
from pathlib import Path
import numpy as np
import pandas as pd

import btc_expansion_breakout_backtest as core

OUT = Path("btc_oos_results")
OUT.mkdir(exist_ok=True)

# LOCKED PARAMETERS FROM PRIOR GRID WINNER
LOCKED_VOLUME_MULT = 2.0
LOCKED_BBW_PERCENTILE = 0.35
LOCKED_PRE_RANGE_PCT = 0.02
LOCKED_STOP_PCT = 0.010
LOCKED_TARGET_R = 4.0
LOCKED_ENTRY_MODE = "RETEST"

# Explicit split:
# 2021-2024 = historical development/in-sample era
# 2025      = first holdout
# 2026      = second holdout / most recent regime
PERIODS = [
    ("TRAIN_2021_2024", "2021-01-01", "2025-01-01"),
    ("OOS_2025",        "2025-01-01", "2026-01-01"),
    ("OOS_2026",        "2026-01-01", "2027-01-01"),
    ("OOS_2025_2026",   "2025-01-01", "2027-01-01"),
    ("FULL_2021_2026",  "2021-01-01", "2027-01-01"),
]

BOOTSTRAP_N = 10000
BOOTSTRAP_SEED = 20260820


def locked_backtest(x: pd.DataFrame) -> pd.DataFrame:
    return core.backtest(
        x,
        min_volume_mult=LOCKED_VOLUME_MULT,
        max_bbw_percentile=LOCKED_BBW_PERCENTILE,
        max_pre_range_pct=LOCKED_PRE_RANGE_PCT,
        stop_pct=LOCKED_STOP_PCT,
        target_r=LOCKED_TARGET_R,
        entry_mode=LOCKED_ENTRY_MODE,
    )


def summarize_extra(trades: pd.DataFrame) -> dict:
    s = core.summarize(trades)
    if trades.empty:
        return {
            **s,
            "sum_net": np.nan,
            "positive_expectancy": False,
            "longest_losing_streak": np.nan,
            "largest_win": np.nan,
            "largest_loss": np.nan,
        }

    pnl = trades["net_return"].astype(float).to_numpy()

    longest = 0
    cur = 0
    for r in pnl:
        if r < 0:
            cur += 1
            longest = max(longest, cur)
        else:
            cur = 0

    return {
        **s,
        "sum_net": float(np.sum(pnl)),
        "positive_expectancy": bool(np.mean(pnl) > 0),
        "longest_losing_streak": int(longest),
        "largest_win": float(np.max(pnl)),
        "largest_loss": float(np.min(pnl)),
    }


def subset_period(x: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    start_ts = pd.Timestamp(start, tz="UTC")
    end_ts = pd.Timestamp(end, tz="UTC")
    return x[(x.index >= start_ts) & (x.index < end_ts)].copy()


def bootstrap_mean_and_r(trades: pd.DataFrame, n=BOOTSTRAP_N, seed=BOOTSTRAP_SEED) -> dict:
    if trades.empty:
        return {
            "trades": 0,
            "avg_net": np.nan,
            "avg_net_ci_low": np.nan,
            "avg_net_ci_high": np.nan,
            "avg_r": np.nan,
            "avg_r_ci_low": np.nan,
            "avg_r_ci_high": np.nan,
            "prob_avg_net_gt_0": np.nan,
            "prob_avg_r_gt_0": np.nan,
        }

    rng = np.random.default_rng(seed)
    net = trades["net_return"].astype(float).to_numpy()
    rr = trades["r_multiple_net"].astype(float).to_numpy()
    ntr = len(net)

    means_net = np.empty(n)
    means_r = np.empty(n)

    # Chunk to avoid excessive memory.
    chunk = 1000
    done = 0
    while done < n:
        m = min(chunk, n - done)
        idx = rng.integers(0, ntr, size=(m, ntr))
        means_net[done:done+m] = net[idx].mean(axis=1)
        means_r[done:done+m] = rr[idx].mean(axis=1)
        done += m

    return {
        "trades": ntr,
        "avg_net": float(net.mean()),
        "avg_net_ci_low": float(np.quantile(means_net, 0.025)),
        "avg_net_ci_high": float(np.quantile(means_net, 0.975)),
        "avg_r": float(rr.mean()),
        "avg_r_ci_low": float(np.quantile(means_r, 0.025)),
        "avg_r_ci_high": float(np.quantile(means_r, 0.975)),
        "prob_avg_net_gt_0": float((means_net > 0).mean()),
        "prob_avg_r_gt_0": float((means_r > 0).mean()),
    }


def yearly_quarterly(trades: pd.DataFrame):
    if trades.empty:
        return pd.DataFrame(), pd.DataFrame()

    z = trades.copy()
    z["entry_time"] = pd.to_datetime(z["entry_time"], utc=True)
    z["year"] = z["entry_time"].dt.year
    z["quarter"] = z["entry_time"].dt.to_period("Q").astype(str)

    yr_rows = []
    for year, g in z.groupby("year"):
        s = summarize_extra(g)
        s["year"] = year
        yr_rows.append(s)

    q_rows = []
    for quarter, g in z.groupby("quarter"):
        s = summarize_extra(g)
        s["quarter"] = quarter
        q_rows.append(s)

    yearly = pd.DataFrame(yr_rows).set_index("year").sort_index()
    quarterly = pd.DataFrame(q_rows).set_index("quarter").sort_index()
    return yearly, quarterly


def latest_signal_analysis(x: pd.DataFrame):
    """
    Check whether the locked rules fired around the latest available data.
    Saves all locked qualifying signals from the last 7 days plus local candle context.
    """
    m = core.signal_mask(
        x,
        min_volume_mult=LOCKED_VOLUME_MULT,
        max_bbw_percentile=LOCKED_BBW_PERCENTILE,
        max_pre_range_pct=LOCKED_PRE_RANGE_PCT,
    )
    sigs = x[m].copy()

    latest_end = x.index.max()
    recent_start = latest_end - pd.Timedelta(days=7)
    recent = sigs[sigs.index >= recent_start].copy()

    cols = [
        "open", "high", "low", "close", "volume",
        "ema20", "ema50",
        "bb_mid", "bb_upper", "bb_lower", "bbw", "bbw_pct_rank",
        "prior_high", "prior_low", "pre_range_pct",
        "vol_ma", "volume_mult", "breakout_pct",
    ]
    keep = [c for c in cols if c in recent.columns]
    recent[keep].to_csv(OUT / "btc_latest_signal_check.csv")

    # Save the final 72 hours of candles so the exact Aug-20 move can be inspected.
    context = x[x.index >= latest_end - pd.Timedelta(hours=72)].copy()
    context_cols = [
        "open", "high", "low", "close", "volume",
        "ema20", "ema50", "bb_mid", "bb_upper", "bb_lower",
        "bbw", "bbw_pct_rank", "prior_high", "prior_low",
        "pre_range_pct", "vol_ma", "volume_mult", "breakout_pct",
    ]
    context[[c for c in context_cols if c in context.columns]].to_csv(
        OUT / "btc_latest_signal_context.csv"
    )

    # For any qualifying signal in the final 7d, simulate the actual retest trade.
    rows = []
    for ts in recent.index:
        i = x.index.get_loc(ts)
        t = core.simulate_trade(
            x,
            signal_i=i,
            stop_pct=LOCKED_STOP_PCT,
            target_r=LOCKED_TARGET_R,
            entry_mode=LOCKED_ENTRY_MODE,
        )
        row = {
            "signal_time": str(ts),
            "signal_close": float(x.iloc[i]["close"]),
            "breakout_level": float(x.iloc[i]["prior_high"]),
            "volume_mult": float(x.iloc[i]["volume_mult"]),
            "bbw_pct_rank": float(x.iloc[i]["bbw_pct_rank"]),
            "pre_range_pct": float(x.iloc[i]["pre_range_pct"]),
        }
        if t is None:
            row.update({
                "retest_entry_found": False,
                "entry_time": None,
                "entry": np.nan,
                "exit_time": None,
                "exit": np.nan,
                "exit_reason": None,
                "net_return": np.nan,
                "r_multiple_net": np.nan,
            })
        else:
            row.update({
                "retest_entry_found": True,
                "entry_time": t.entry_time,
                "entry": t.entry,
                "exit_time": t.exit_time,
                "exit": t.exit,
                "exit_reason": t.exit_reason,
                "net_return": t.net_return,
                "r_multiple_net": t.r_multiple_net,
            })
        rows.append(row)

    recent_trade_check = pd.DataFrame(rows)
    recent_trade_check.to_csv(OUT / "btc_latest_signal_trades.csv", index=False)
    return recent, recent_trade_check


def pct(x):
    return "n/a" if pd.isna(x) else f"{100*x:.3f}%"


def main():
    print("Downloading BTCUSDT 1H candles from Bybit...", flush=True)
    df = core.fetch_bybit_klines(
        start_date="2021-01-01",
        end_date=None,
    )
    print(f"Candles: {len(df):,} | {df.index.min()} -> {df.index.max()}", flush=True)

    x = core.add_features(df)

    print("Running LOCKED parameter validation (no re-optimization)...", flush=True)

    period_rows = []
    all_period_trades = []

    for name, start, end in PERIODS:
        sub = subset_period(x, start, end)
        bt = locked_backtest(sub)
        s = summarize_extra(bt)
        s.update({
            "period": name,
            "start": start,
            "end": end,
        })
        period_rows.append(s)

        if not bt.empty:
            temp = bt.copy()
            temp["validation_period"] = name
            all_period_trades.append(temp)

        print(
            f"{name}: trades={s['trades']} avg_net={pct(s['avg_net'])} "
            f"PF={s['profit_factor']:.3f}" if not pd.isna(s["profit_factor"]) else
            f"{name}: trades={s['trades']} avg_net={pct(s['avg_net'])} PF=n/a",
            flush=True
        )

    period_df = pd.DataFrame(period_rows)
    period_df.to_csv(OUT / "btc_oos_period_results.csv", index=False)

    # Use one full-series backtest for yearly/quarterly stability so trades aren't duplicated.
    full = locked_backtest(x)
    full.to_csv(OUT / "btc_oos_trades.csv", index=False)

    yearly, quarterly = yearly_quarterly(full)
    yearly.to_csv(OUT / "btc_oos_yearly.csv")
    quarterly.to_csv(OUT / "btc_oos_quarterly.csv")

    # Bootstrap the key holdout blocks and full sample.
    boot_rows = []
    for name, start, end in [
        ("OOS_2025", "2025-01-01", "2026-01-01"),
        ("OOS_2026", "2026-01-01", "2027-01-01"),
        ("OOS_2025_2026", "2025-01-01", "2027-01-01"),
        ("FULL_2021_2026", "2021-01-01", "2027-01-01"),
    ]:
        bt = locked_backtest(subset_period(x, start, end))
        b = bootstrap_mean_and_r(bt)
        b["period"] = name
        boot_rows.append(b)

    boot = pd.DataFrame(boot_rows)
    boot.to_csv(OUT / "btc_oos_bootstrap.csv", index=False)

    recent_sigs, recent_trade_check = latest_signal_analysis(x)

    lines = []
    lines.append("BTC EXPANSION / BREAKOUT — LOCKED OOS VALIDATION")
    lines.append("=" * 70)
    lines.append("LOCKED PARAMETERS (from prior grid winner; NOT re-optimized here)")
    lines.append(f"volume_mult >= {LOCKED_VOLUME_MULT:.2f}x")
    lines.append(f"BBW percentile <= {LOCKED_BBW_PERCENTILE:.0%}")
    lines.append(f"pre-breakout 24h range <= {LOCKED_PRE_RANGE_PCT:.2%}")
    lines.append(f"stop = {LOCKED_STOP_PCT:.2%}")
    lines.append(f"target = {LOCKED_TARGET_R:.1f}R")
    lines.append(f"entry mode = {LOCKED_ENTRY_MODE}")
    lines.append("")
    lines.append("PERIOD RESULTS")
    lines.append(period_df.to_string(index=False))
    lines.append("")
    lines.append("BOOTSTRAP 95% CI")
    lines.append(boot.to_string(index=False))
    lines.append("")
    lines.append("YEARLY STABILITY")
    lines.append(yearly.to_string())
    lines.append("")
    lines.append("QUARTERLY STABILITY")
    lines.append(quarterly.to_string())
    lines.append("")
    lines.append("LATEST 7-DAY QUALIFYING SIGNALS")
    if recent_trade_check.empty:
        lines.append("No locked-rule signal in the latest 7 days.")
    else:
        lines.append(recent_trade_check.to_string(index=False))
    lines.append("")
    lines.append("Interpretation rule of thumb:")
    lines.append("- Stronger evidence: OOS_2025 and OOS_2026 both positive, PF > 1, and bootstrap probability(avg_net>0) high.")
    lines.append("- Weak evidence: only full-sample is positive while one/both holdouts are negative or CI spans deeply below zero.")
    lines.append("- Latest move: btc_latest_signal_trades.csv shows whether the locked model actually generated a signal/retest entry.")

    report = "\n".join(lines)
    (OUT / "btc_oos_summary.txt").write_text(report, encoding="utf-8")

    print("\n" + report, flush=True)
    print(f"\nSaved outputs to: {OUT.resolve()}", flush=True)


if __name__ == "__main__":
    main()

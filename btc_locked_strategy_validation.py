
import time
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from pathlib import Path

OUT = Path("btc_locked_results")
OUT.mkdir(exist_ok=True)

STOP = 0.01
FEE = 0.0011
TARGETS = [0.015, 0.02]
HOURS = [6, 12, 24]
MODES = ["CLOSE", "RETEST"]

def ms(s):
    return int(datetime.fromisoformat(s).replace(tzinfo=timezone.utc).timestamp() * 1000)

def fetch():
    url = "https://api.bybit.com/v5/market/kline"
    start = ms("2021-01-01")
    end = int(datetime.now(timezone.utc).timestamp() * 1000)
    rows = []

    while end > start:
        r = requests.get(
            url,
            params={
                "category": "linear",
                "symbol": "BTCUSDT",
                "interval": "60",
                "end": end,
                "limit": 1000,
            },
            timeout=20,
        )
        r.raise_for_status()
        js = r.json()
        if js.get("retCode") != 0:
            raise RuntimeError(js)

        part = js["result"]["list"]
        if not part:
            break

        rows += part
        oldest = min(int(v[0]) for v in part)
        if oldest <= start:
            break

        end = oldest - 1
        time.sleep(0.05)

    cols = ["timestamp", "open", "high", "low", "close", "volume", "turnover"]
    df = pd.DataFrame(rows, columns=cols)
    df["timestamp"] = pd.to_numeric(df["timestamp"]).astype("int64")
    for c in cols[1:]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.drop_duplicates("timestamp").sort_values("timestamp")
    df = df[df["timestamp"] >= start]
    df["dt"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.set_index("dt")

    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    return df[df["timestamp"] + 3_600_000 <= now_ms]

def percentile_rank(series):
    def f(a):
        return pd.Series(a).rank(pct=True).iloc[-1]
    return series.rolling(720, min_periods=150).apply(f, raw=False)

def add_features(df):
    x = df.copy()

    for n in [20, 50, 100, 200]:
        x[f"ema{n}"] = x["close"].ewm(span=n, adjust=False).mean()

    mid = x["close"].rolling(20).mean()
    sd = x["close"].rolling(20).std(ddof=0)
    x["bbw_rank"] = percentile_rank(4 * sd / mid)

    flag = (x["bbw_rank"] <= 0.35).astype(int)
    grp = (flag != flag.shift()).cumsum()
    run = flag.groupby(grp).cumsum()
    x["compression_hours"] = run.where(flag.eq(1), 0).shift(1).fillna(0)

    x["prior_high24"] = x["high"].shift(1).rolling(24).max()
    x["prior_high48"] = x["high"].shift(1).rolling(48).max()
    x["prior_low48"] = x["low"].shift(1).rolling(48).min()
    x["range48"] = (x["prior_high48"] - x["prior_low48"]) / x["prior_low48"]

    x["vol_ma24"] = x["volume"].shift(1).rolling(24).mean()
    x["volume_mult"] = x["volume"] / x["vol_ma24"]

    x["trend_stack_score"] = (
        (x["close"] > x["ema20"]).astype(int)
        + (x["ema20"] > x["ema50"]).astype(int)
        + (x["ema50"] > x["ema100"]).astype(int)
        + (x["ema100"] > x["ema200"]).astype(int)
    )

    return x

def signal_mask(x):
    return (
        x["prior_high24"].notna()
        & (x["close"] > x["prior_high24"] * 1.0005)
        & (x["volume_mult"] >= 1.25)
        & (x["compression_hours"] >= 12)
        & (x["range48"] <= 0.03)
        & (x["trend_stack_score"] >= 3)
    )

def choose_entry(x, i, mode):
    if mode == "CLOSE":
        return i, float(x["close"].iloc[i])

    level = float(x["prior_high24"].iloc[i])
    for j in range(i + 1, min(i + 7, len(x))):
        row = x.iloc[j]
        touched = row["low"] <= level * 1.0025
        reclaimed = row["close"] >= level
        if touched and reclaimed:
            return j, float(row["close"])

    return None

def run_variant(x, mode, tp, hours):
    rows = []
    blocked_until = -1

    for i in np.flatnonzero(signal_mask(x).values):
        if i <= blocked_until:
            continue

        ent = choose_entry(x, i, mode)
        if ent is None:
            continue

        ei, entry = ent
        stop = entry * (1 - STOP)
        target = entry * (1 + tp)
        last = min(ei + hours, len(x) - 1)

        exit_i = None
        exit_price = None
        reason = None

        for j in range(ei + 1, last + 1):
            row = x.iloc[j]

            # Conservative same-candle convention: stop first.
            if row["low"] <= stop:
                exit_i = j
                exit_price = stop
                reason = "STOP"
                break

            if row["high"] >= target:
                exit_i = j
                exit_price = target
                reason = "TARGET"
                break

        if exit_i is None:
            exit_i = last
            exit_price = float(x["close"].iloc[last])
            reason = f"TIME_{hours}H"

        net = exit_price / entry - 1 - FEE

        rows.append(
            {
                "signal_time": x.index[i],
                "entry_time": x.index[ei],
                "exit_time": x.index[exit_i],
                "entry": entry,
                "exit": exit_price,
                "reason": reason,
                "net_return": net,
                "net_r": net / STOP,
                "mode": mode,
                "tp": tp,
                "hours": hours,
            }
        )

        blocked_until = exit_i

    return pd.DataFrame(rows)

def stats(t):
    if t.empty:
        return {
            "trades": 0,
            "avg_net": np.nan,
            "median_net": np.nan,
            "win_rate": np.nan,
            "pf": np.nan,
            "total_return": np.nan,
            "max_dd": np.nan,
            "avg_r": np.nan,
            "longest_losing_streak": np.nan,
        }

    r = t["net_return"].to_numpy()
    wins = r[r > 0].sum()
    losses = -r[r < 0].sum()

    equity = np.cumprod(1 + r)
    peak = np.maximum.accumulate(equity)
    dd = equity / peak - 1

    longest = 0
    cur = 0
    for z in r:
        if z < 0:
            cur += 1
            longest = max(longest, cur)
        else:
            cur = 0

    return {
        "trades": len(r),
        "avg_net": r.mean(),
        "median_net": np.median(r),
        "win_rate": (r > 0).mean(),
        "pf": wins / losses if losses > 0 else np.inf,
        "total_return": equity[-1] - 1,
        "max_dd": dd.min(),
        "avg_r": t["net_r"].mean(),
        "longest_losing_streak": longest,
    }

def subset(t, start, end):
    if t.empty:
        return t

    a = pd.Timestamp(start, tz="UTC")
    b = pd.Timestamp(end, tz="UTC")
    et = pd.to_datetime(t["entry_time"], utc=True)

    return t[(et >= a) & (et < b)]

def bootstrap(t, n=10000, seed=20260820):
    if t.empty:
        return np.nan, np.nan, np.nan

    a = t["net_return"].to_numpy()
    rng = np.random.default_rng(seed)
    means = np.empty(n)

    for k in range(n):
        sample = a[rng.integers(0, len(a), len(a))]
        means[k] = sample.mean()

    lo, hi = np.quantile(means, [0.025, 0.975])
    return lo, hi, (means > 0).mean()

def main():
    print("=== BTC locked strategy validation ===", flush=True)

    df = fetch()
    print(f"Candles {len(df):,}: {df.index.min()} -> {df.index.max()}", flush=True)

    x = add_features(df)

    periods = [
        ("DEV", "2021-01-01", "2025-01-01"),
        ("2025", "2025-01-01", "2026-01-01"),
        ("2026", "2026-01-01", "2027-01-01"),
        ("OOS", "2025-01-01", "2027-01-01"),
        ("LAST12M", "2025-08-20", "2026-08-21"),
        ("FULL", "2021-01-01", "2027-01-01"),
    ]

    variant_rows = []
    period_rows = []
    bootstrap_rows = []
    all_trades = []

    for mode in MODES:
        for tp in TARGETS:
            for h in HOURS:
                variant = f"{mode}_TP{tp*100:.1f}_T{h}H"
                t = run_variant(x, mode, tp, h)
                t["variant"] = variant
                all_trades.append(t)

                s = stats(t)
                s["variant"] = variant
                variant_rows.append(s)

                print(
                    f"{variant}: trades={s['trades']} "
                    f"avg_net={s['avg_net']:.4%} "
                    f"PF={s['pf']:.3f}",
                    flush=True,
                )

                for name, a, b in periods:
                    q = subset(t, a, b)
                    z = stats(q)
                    z.update({"variant": variant, "period": name})
                    period_rows.append(z)

                    if name in ["2025", "2026", "OOS", "LAST12M"]:
                        lo, hi, p = bootstrap(q)
                        bootstrap_rows.append(
                            {
                                "variant": variant,
                                "period": name,
                                "trades": len(q),
                                "avg_net": q["net_return"].mean() if len(q) else np.nan,
                                "ci_low": lo,
                                "ci_high": hi,
                                "prob_avg_gt0": p,
                            }
                        )

    V = pd.DataFrame(variant_rows)
    P = pd.DataFrame(period_rows)
    B = pd.DataFrame(bootstrap_rows)
    T = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()

    V.to_csv(OUT / "btc_locked_variant_results.csv", index=False)
    P.to_csv(OUT / "btc_locked_period_results.csv", index=False)
    B.to_csv(OUT / "btc_locked_bootstrap.csv", index=False)
    T.to_csv(OUT / "btc_locked_trades.csv", index=False)

    yearly_rows = []
    if not T.empty:
        for variant, g in T.groupby("variant"):
            temp = g.copy()
            temp["year"] = pd.to_datetime(temp["entry_time"], utc=True).dt.year

            for year, q in temp.groupby("year"):
                s = stats(q)
                s.update({"variant": variant, "year": year})
                yearly_rows.append(s)

    pd.DataFrame(yearly_rows).to_csv(OUT / "btc_locked_yearly.csv", index=False)

    recent = (
        T[
            pd.to_datetime(T["entry_time"], utc=True)
            >= x.index.max() - pd.Timedelta(days=30)
        ]
        if not T.empty
        else T
    )
    recent.to_csv(OUT / "btc_locked_recent_trades.csv", index=False)

    text = (
        "BTC LOCKED BREAKOUT VALIDATION\n"
        + "=" * 64
        + "\n"
        + "Filters: volume>=1.25x; compression>=12h; range48<=3%; "
          "trend_stack_score>=3; SL=1%; fee=0.11% round trip\n"
        + "12 predefined variants only.\n\n"
        + "FULL SAMPLE:\n"
        + V.to_string(index=False)
        + "\n\n"
        + "2025 / 2026 / OOS:\n"
        + P[P["period"].isin(["2025", "2026", "OOS"])].to_string(index=False)
        + "\n\n"
        + "BOOTSTRAP:\n"
        + B.to_string(index=False)
    )

    (OUT / "btc_locked_summary.txt").write_text(text, encoding="utf-8")
    print(text, flush=True)
    print(f"Saved to {OUT.resolve()}", flush=True)

if __name__ == "__main__":
    main()

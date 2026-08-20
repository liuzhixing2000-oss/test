
import time, math, requests, numpy as np, pandas as pd
from datetime import datetime, timezone
from pathlib import Path

OUT=Path("btc_overextension_results"); OUT.mkdir(exist_ok=True)
START="2021-01-01"; STOP=.01; TARGET=.015; HOURS=12; FEE=.0011
FEATURES=["distance_ema200","ema50_slope_24h","ema200_slope_72h","ret_30d","range_atr","volume_mult"]

def ms(s): return int(datetime.fromisoformat(s).replace(tzinfo=timezone.utc).timestamp()*1000)

def fetch():
    url="https://api.bybit.com/v5/market/kline"; start=ms(START)
    end=int(datetime.now(timezone.utc).timestamp()*1000); rows=[]
    while end>start:
        r=requests.get(url,params={"category":"linear","symbol":"BTCUSDT","interval":"60","end":end,"limit":1000},timeout=20)
        r.raise_for_status(); js=r.json()
        if js.get("retCode")!=0: raise RuntimeError(js)
        z=js["result"]["list"]
        if not z: break
        rows += z; oldest=min(int(v[0]) for v in z)
        if oldest<=start: break
        end=oldest-1; time.sleep(.05)
    c=["timestamp","open","high","low","close","volume","turnover"]
    d=pd.DataFrame(rows,columns=c); d["timestamp"]=pd.to_numeric(d["timestamp"]).astype("int64")
    for x in c[1:]: d[x]=pd.to_numeric(d[x],errors="coerce")
    d=d.drop_duplicates("timestamp").sort_values("timestamp"); d=d[d.timestamp>=start]
    d["dt"]=pd.to_datetime(d.timestamp,unit="ms",utc=True); d=d.set_index("dt")
    now=int(datetime.now(timezone.utc).timestamp()*1000)
    return d[d.timestamp+3600000<=now]

def prank(s,window=720,minp=150):
    def f(a): return pd.Series(a).rank(pct=True).iloc[-1]
    return s.rolling(window,min_periods=minp).apply(f,raw=False)

def features(d):
    x=d.copy()
    for n in [20,50,100,200]: x[f"ema{n}"]=x.close.ewm(span=n,adjust=False).mean()
    mid=x.close.rolling(20).mean(); sd=x.close.rolling(20).std(ddof=0)
    x["bbw_rank"]=prank(4*sd/mid)
    flag=(x.bbw_rank<=.35).astype(int); grp=(flag!=flag.shift()).cumsum()
    x["compression_hours"]=flag.groupby(grp).cumsum().where(flag.eq(1),0).shift().fillna(0)
    x["h24"]=x.high.shift().rolling(24).max()
    x["h48"]=x.high.shift().rolling(48).max(); x["l48"]=x.low.shift().rolling(48).min()
    x["range48"]=(x.h48-x.l48)/x.l48
    x["volume_mult"]=x.volume/x.volume.shift().rolling(24).mean()
    x["trend_stack_score"]=(x.close>x.ema20).astype(int)+(x.ema20>x.ema50).astype(int)+(x.ema50>x.ema100).astype(int)+(x.ema100>x.ema200).astype(int)
    prev=x.close.shift(); tr=pd.concat([x.high-x.low,(x.high-prev).abs(),(x.low-prev).abs()],axis=1).max(axis=1)
    atr=tr.rolling(14).mean()
    x["distance_ema200"]=x.close/x.ema200-1
    x["ema50_slope_24h"]=x.ema50/x.ema50.shift(24)-1
    x["ema200_slope_72h"]=x.ema200/x.ema200.shift(72)-1
    x["ret_30d"]=x.close/x.close.shift(720)-1
    x["range_atr"]=(x.high-x.low)/atr
    return x

def mask(x):
    return x.h24.notna()&(x.close>x.h24*1.0005)&(x.volume_mult>=1.25)&(x.compression_hours>=12)&(x.range48<=.03)&(x.trend_stack_score>=3)

def trades(x):
    out=[]; busy=-1
    for i in np.flatnonzero(mask(x).values):
        if i<=busy: continue
        entry=float(x.close.iloc[i]); sl=entry*(1-STOP); tp=entry*(1+TARGET); last=min(i+HOURS,len(x)-1)
        ex=None
        for j in range(i+1,last+1):
            if x.low.iloc[j]<=sl: ex=(j,sl,"STOP"); break
            if x.high.iloc[j]>=tp: ex=(j,tp,"TARGET"); break
        if ex is None: ex=(last,float(x.close.iloc[last]),"TIME")
        j,px,why=ex; net=px/entry-1-FEE
        r={"entry_time":x.index[i],"exit_time":x.index[j],"reason":why,"net_return":net,"year":x.index[i].year}
        for f in FEATURES:r[f]=float(x[f].iloc[i]) if pd.notna(x[f].iloc[i]) else np.nan
        out.append(r); busy=j
    return pd.DataFrame(out)

def perf(g):
    if len(g)==0:return {"trades":0,"avg_net":np.nan,"win_rate":np.nan,"pf":np.nan,"total_return":np.nan}
    r=g.net_return.to_numpy(); pos=r[r>0].sum(); neg=-r[r<0].sum()
    return {"trades":len(g),"avg_net":r.mean(),"win_rate":(r>0).mean(),"pf":pos/neg if neg else np.inf,"total_return":np.prod(1+r)-1}

def main():
    print("=== BTC overextension quantile study ===",flush=True)
    d=fetch(); print(f"Candles: {len(d):,} | {d.index.min()} -> {d.index.max()}",flush=True)
    x=features(d); t=trades(x); t.to_csv(OUT/"btc_overextension_trades.csv",index=False)

    # IMPORTANT: quartile boundaries are learned ONLY on 2021-2024 DEV.
    dev=t[t.year<=2024]
    cuts=[]; rows=[]
    periods=[("DEV_2021_2024",lambda z:z.year<=2024),("2025",lambda z:z.year==2025),("2026",lambda z:z.year==2026),("OOS_2025_2026",lambda z:z.year>=2025)]

    for f in FEATURES:
        s=pd.to_numeric(dev[f],errors="coerce").dropna()
        q=s.quantile([.25,.5,.75]).values
        q1,q2,q3=map(float,q)
        cuts.append({"feature":f,"q25":q1,"q50":q2,"q75":q3,"dev_n":len(s)})
        bins=[-np.inf,q1,q2,q3,np.inf]; labels=["Q1","Q2","Q3","Q4"]
        for pname,fn in periods:
            g=t[fn(t)].copy()
            g["quartile"]=pd.cut(g[f],bins=bins,labels=labels,include_lowest=True)
            for lab in labels:
                h=g[g.quartile==lab]
                z=perf(h); z.update({"feature":f,"period":pname,"quartile":lab})
                rows.append(z)

    C=pd.DataFrame(cuts); R=pd.DataFrame(rows)
    C.to_csv(OUT/"btc_dev_quartile_boundaries.csv",index=False)
    R.to_csv(OUT/"btc_quantile_performance.csv",index=False)

    # monotonicity / Q1+Q2 versus Q4, without selecting a new cutoff
    cmp=[]
    for f in FEATURES:
        c=C[C.feature==f].iloc[0]; bins=[-np.inf,c.q25,c.q50,c.q75,np.inf]
        for pname,fn in periods:
            g=t[fn(t)].copy()
            g["quartile"]=pd.cut(g[f],bins=bins,labels=["Q1","Q2","Q3","Q4"],include_lowest=True)
            low=g[g.quartile.isin(["Q1","Q2"])]; high=g[g.quartile=="Q4"]
            a=perf(low); b=perf(high)
            cmp.append({"feature":f,"period":pname,"low_half_trades":a["trades"],"low_half_avg":a["avg_net"],"low_half_pf":a["pf"],
                        "q4_trades":b["trades"],"q4_avg":b["avg_net"],"q4_pf":b["pf"],"avg_spread_low_minus_q4":a["avg_net"]-b["avg_net"]})
    M=pd.DataFrame(cmp); M.to_csv(OUT/"btc_low_vs_overextended_q4.csv",index=False)

    lines=["BTC OVEREXTENSION QUANTILE STUDY","="*72,
           "Strategy remains LOCKED: close entry / TP1.5% / SL1% / 12h / fee0.11%.",
           "Quartile boundaries are learned ONLY from 2021-2024 DEV and then frozen for 2025/2026.",
           "No cutoff optimization is performed.","",
           "DEV QUARTILE BOUNDARIES",C.to_string(index=False),"",
           "LOW HALF (Q1+Q2) VS MOST EXTENDED Q4",M.to_string(index=False),"",
           "FULL QUARTILE PERFORMANCE",R.to_string(index=False)]
    report="\n".join(lines)
    (OUT/"btc_overextension_summary.txt").write_text(report,encoding="utf-8")
    print(report,flush=True); print(f"Saved to {OUT.resolve()}",flush=True)

if __name__=="__main__":main()

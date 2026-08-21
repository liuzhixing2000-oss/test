
import time, requests, numpy as np, pandas as pd
from datetime import datetime, timezone
from pathlib import Path

OUT=Path("results"); OUT.mkdir(exist_ok=True)
H=[3,6,12,24,48]
TH=[.90,.95,.975,.99]
NAMES={.90:"90",.95:"95",.975:"97.5",.99:"99"}

def fetch():
    u="https://api.bybit.com/v5/market/kline"
    s=int(datetime(2021,1,1,tzinfo=timezone.utc).timestamp()*1000)
    e=int(datetime.now(timezone.utc).timestamp()*1000); a=[]
    while e>s:
        j=requests.get(u,params={"category":"linear","symbol":"BTCUSDT","interval":"60","end":e,"limit":1000},timeout=20).json()
        p=j["result"]["list"]
        if not p: break
        a+=p; old=min(int(v[0]) for v in p)
        if old<=s: break
        e=old-1; time.sleep(.05)
    c=["ts","open","high","low","close","volume","turnover"]
    x=pd.DataFrame(a,columns=c)
    for z in c:x[z]=pd.to_numeric(x[z],errors="coerce")
    x=x.drop_duplicates("ts").sort_values("ts"); x=x[x.ts>=s]
    x["dt"]=pd.to_datetime(x.ts,unit="ms",utc=True)
    return x.set_index("dt")

def main():
    print("=== BTC EMA200 first threshold-cross study ===",flush=True)
    x=fetch()
    x["ema200"]=x.close.ewm(span=200,adjust=False).mean()
    x["dist"]=x.close/x.ema200-1
    vals=x.dist.to_numpy(); pct=np.full(len(x),np.nan)
    for i in range(719,len(x)):
        h=vals[:i+1]; h=h[np.isfinite(h)]
        pct[i]=(h<=vals[i]).mean()
    x["pct"]=pct
    x["ret30"]=x.close/x.close.shift(720)-1
    x["ema200s7"]=x.ema200/x.ema200.shift(168)-1
    lr=np.log(x.close/x.close.shift())
    x["rv30"]=lr.rolling(720).std(ddof=0)*np.sqrt(720)
    x["rvmed"]=x.rv30.rolling(2160,min_periods=720).median()

    events=[]; active=False; crossed=set(); episode=0
    for i,p in enumerate(x.pct):
        if not np.isfinite(p): continue
        if active and p<.80:
            active=False; crossed=set()
        if not active and p>=.90:
            active=True; crossed=set(); episode+=1
        if active:
            for t in TH:
                if p>=t and t not in crossed:
                    crossed.add(t)
                    if i+48>=len(x): continue
                    e=x.close.iloc[i]
                    r={"episode":episode,"time":x.index[i],"year":x.index[i].year,
                       "threshold":NAMES[t],"threshold_value":t,"pct":p,"dist":x.dist.iloc[i],
                       "trend30":"UP" if x.ret30.iloc[i]>0 else "DOWN_FLAT",
                       "ema200reg":"RISING" if x.ema200s7.iloc[i]>0 else "FLAT_FALLING",
                       "volreg":"HIGH" if x.rv30.iloc[i]>x.rvmed.iloc[i] else "LOW"}
                    for h in H:
                        r[f"r{h}"]=x.close.iloc[i+h]/e-1
                    events.append(r)

    E=pd.DataFrame(events); E.to_csv(OUT/"threshold_events.csv",index=False)

    rows=[]
    for period in ["DEV","2025","2026","OOS"]:
        q=E[E.year<=2024] if period=="DEV" else E[E.year==int(period)] if period in ("2025","2026") else E[E.year>=2025]
        for t in TH:
            g=q[q.threshold_value==t]
            r={"period":period,"threshold":NAMES[t],"n":len(g)}
            for h in H:
                r[f"avg{h}"]=g[f"r{h}"].mean() if len(g) else np.nan
                r[f"median{h}"]=g[f"r{h}"].median() if len(g) else np.nan
                r[f"pneg{h}"]=(g[f"r{h}"]<0).mean() if len(g) else np.nan
            rows.append(r)
    S=pd.DataFrame(rows); S.to_csv(OUT/"threshold_summary.csv",index=False)

    yr=[]
    for y in sorted(E.year.unique()):
        for t in TH:
            g=E[(E.year==y)&(E.threshold_value==t)]
            yr.append({"year":y,"threshold":NAMES[t],"n":len(g),
                       "avg6":g.r6.mean() if len(g) else np.nan,
                       "avg12":g.r12.mean() if len(g) else np.nan,
                       "avg24":g.r24.mean() if len(g) else np.nan,
                       "avg48":g.r48.mean() if len(g) else np.nan})
    Y=pd.DataFrame(yr); Y.to_csv(OUT/"yearly_thresholds.csv",index=False)

    reg=[]
    oos=E[E.year>=2025]
    for col in ["trend30","ema200reg","volreg"]:
        for v in sorted(oos[col].dropna().unique()):
            for t in TH:
                g=oos[(oos[col]==v)&(oos.threshold_value==t)]
                reg.append({"regime":col,"value":v,"threshold":NAMES[t],"n":len(g),
                            "avg6":g.r6.mean() if len(g) else np.nan,
                            "avg12":g.r12.mean() if len(g) else np.nan,
                            "avg24":g.r24.mean() if len(g) else np.nan})
    R=pd.DataFrame(reg); R.to_csv(OUT/"oos_regime_thresholds.csv",index=False)

    txt=("OOS FIRST THRESHOLD CROSSES\n"+
         S[S.period=="OOS"].to_string(index=False)+
         "\n\nYEARLY STABILITY\n"+Y.to_string(index=False)+
         "\n\nOOS REGIME SPLITS\n"+R.to_string(index=False))
    (OUT/"summary.txt").write_text(txt)
    print(txt,flush=True)

if __name__=="__main__": main()

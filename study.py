
import time,requests,numpy as np,pandas as pd
from datetime import datetime,timezone
from pathlib import Path
O=Path("results");O.mkdir(exist_ok=True)
H=[3,6,12,24,48]; BANDS=[(.90,.95,"90-95"),(.95,.975,"95-97.5"),(.975,.99,"97.5-99"),(.99,2,"99-100")]

def fetch():
    u="https://api.bybit.com/v5/market/kline"; s=int(datetime(2021,1,1,tzinfo=timezone.utc).timestamp()*1000); e=int(datetime.now(timezone.utc).timestamp()*1000); a=[]
    while e>s:
        j=requests.get(u,params={"category":"linear","symbol":"BTCUSDT","interval":"60","end":e,"limit":1000},timeout=20).json(); p=j["result"]["list"]
        if not p: break
        a+=p; o=min(int(x[0]) for x in p)
        if o<=s: break
        e=o-1; time.sleep(.05)
    c=["ts","open","high","low","close","volume","turnover"]; d=pd.DataFrame(a,columns=c)
    for z in c:d[z]=pd.to_numeric(d[z],errors="coerce")
    d=d.drop_duplicates("ts").sort_values("ts"); d=d[d.ts>=s]; d["dt"]=pd.to_datetime(d.ts,unit="ms",utc=True)
    return d.set_index("dt")

def main():
    print("=== BTC EMA200 independent episode + regime study ===",flush=True)
    x=fetch(); x["ema200"]=x.close.ewm(span=200,adjust=False).mean(); x["dist"]=x.close/x.ema200-1
    vals=x.dist.to_numpy(); pct=np.full(len(x),np.nan)
    for i in range(719,len(x)): pct[i]=(vals[:i+1]<=vals[i]).mean()
    x["pct"]=pct; x["ret30"]=x.close/x.close.shift(720)-1; x["ema200s7"]=x.ema200/x.ema200.shift(168)-1
    lr=np.log(x.close/x.close.shift()); x["rv30"]=lr.rolling(720).std(ddof=0)*np.sqrt(720); x["rvmed"]=x.rv30.rolling(2160,min_periods=720).median()
    starts=[];active=False
    for i,p in enumerate(x.pct):
        if pd.isna(p): continue
        if not active and p>=.90: starts.append(i);active=True
        elif active and p<.80: active=False
    rows=[]
    for i in starts:
        if i+48>=len(x): continue
        p=x.pct.iloc[i]; band=next((n for lo,hi,n in BANDS if lo<=p<hi),None)
        if not band: continue
        e=x.close.iloc[i]; r={"time":x.index[i],"year":x.index[i].year,"band":band,"pct":p,"dist":x.dist.iloc[i],
                             "trend30":"UP" if x.ret30.iloc[i]>0 else "DOWN_FLAT",
                             "ema200reg":"RISING" if x.ema200s7.iloc[i]>0 else "FLAT_FALLING",
                             "volreg":"HIGH" if x.rv30.iloc[i]>x.rvmed.iloc[i] else "LOW"}
        for h in H:r[f"r{h}"]=x.close.iloc[i+h]/e-1
        rows.append(r)
    E=pd.DataFrame(rows);E.to_csv(O/"episodes.csv",index=False)
    out=[]
    for p in ["DEV","2025","2026","OOS"]:
        q=E[E.year<=2024] if p=="DEV" else E[E.year==int(p)] if p in ["2025","2026"] else E[E.year>=2025]
        for _,_,b in BANDS:
            g=q[q.band==b]; z={"period":p,"band":b,"n":len(g)}
            for h in H:z[f"avg{h}"]=g[f"r{h}"].mean() if len(g) else np.nan;z[f"pneg{h}"]=(g[f"r{h}"]<0).mean() if len(g) else np.nan
            out.append(z)
    S=pd.DataFrame(out);S.to_csv(O/"episode_summary.csv",index=False)
    reg=[]
    oos=E[E.year>=2025]
    for col in ["trend30","ema200reg","volreg"]:
        for v in oos[col].dropna().unique():
            for _,_,b in BANDS:
                g=oos[(oos[col]==v)&(oos.band==b)]; reg.append({"regime":col,"value":v,"band":b,"n":len(g),"avg6":g.r6.mean() if len(g) else np.nan,"avg12":g.r12.mean() if len(g) else np.nan,"avg24":g.r24.mean() if len(g) else np.nan})
    R=pd.DataFrame(reg);R.to_csv(O/"regime_summary.csv",index=False)
    txt="OOS EPISODES\n"+S[S.period=="OOS"].to_string(index=False)+"\n\nREGIMES\n"+R.to_string(index=False);(O/"summary.txt").write_text(txt);print(txt,flush=True)
if __name__=="__main__":main()

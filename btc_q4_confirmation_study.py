
import time, requests, numpy as np, pandas as pd
from datetime import datetime, timezone
from pathlib import Path

OUT=Path("btc_q4_confirm_results"); OUT.mkdir(exist_ok=True)
START="2021-01-01"; Q4=0.031470; STOP=.01; TARGET=.01; HOLD=12; WINDOW=6; FEE=.0011
MODES=["IMMEDIATE","FIRST_RED_CANDLE","BREAK_PREV_LOW","CLOSE_BACK_BELOW_BREAKOUT",
       "UPPER_WICK_REJECTION","MACD_HIST_WEAKEN","RSI_ROLLOVER"]

def ms(s): return int(datetime.fromisoformat(s).replace(tzinfo=timezone.utc).timestamp()*1000)

def fetch():
    url="https://api.bybit.com/v5/market/kline"; start=ms(START)
    end=int(datetime.now(timezone.utc).timestamp()*1000); rows=[]
    while end>start:
        r=requests.get(url,params={"category":"linear","symbol":"BTCUSDT","interval":"60","end":end,"limit":1000},timeout=20)
        r.raise_for_status(); js=r.json()
        if js.get("retCode")!=0: raise RuntimeError(js)
        part=js["result"]["list"]
        if not part: break
        rows+=part; oldest=min(int(v[0]) for v in part)
        if oldest<=start: break
        end=oldest-1; time.sleep(.05)
    c=["timestamp","open","high","low","close","volume","turnover"]
    d=pd.DataFrame(rows,columns=c); d["timestamp"]=pd.to_numeric(d["timestamp"]).astype("int64")
    for x in c[1:]: d[x]=pd.to_numeric(d[x],errors="coerce")
    d=d.drop_duplicates("timestamp").sort_values("timestamp"); d=d[d.timestamp>=start]
    d["dt"]=pd.to_datetime(d.timestamp,unit="ms",utc=True); d=d.set_index("dt")
    now=int(datetime.now(timezone.utc).timestamp()*1000)
    return d[d.timestamp+3600000<=now]

def prank(s):
    def f(a): return pd.Series(a).rank(pct=True).iloc[-1]
    return s.rolling(720,min_periods=150).apply(f,raw=False)

def rsi(c,n=14):
    d=c.diff(); g=d.clip(lower=0); l=-d.clip(upper=0)
    ag=g.ewm(alpha=1/n,adjust=False).mean(); al=l.ewm(alpha=1/n,adjust=False).mean()
    rs=ag/al.replace(0,np.nan); return 100-100/(1+rs)

def feat(d):
    x=d.copy()
    for n in [20,50,100,200]: x[f"ema{n}"]=x.close.ewm(span=n,adjust=False).mean()
    m=x.close.rolling(20).mean(); sd=x.close.rolling(20).std(ddof=0)
    x["bbw_rank"]=prank(4*sd/m)
    f=(x.bbw_rank<=.35).astype(int); g=(f!=f.shift()).cumsum()
    x["comp"]=f.groupby(g).cumsum().where(f.eq(1),0).shift().fillna(0)
    x["h24"]=x.high.shift().rolling(24).max()
    x["h48"]=x.high.shift().rolling(48).max(); x["l48"]=x.low.shift().rolling(48).min()
    x["r48"]=(x.h48-x.l48)/x.l48
    x["vm"]=x.volume/x.volume.shift().rolling(24).mean()
    x["trend_stack_score"]=(x.close>x.ema20).astype(int)+(x.ema20>x.ema50).astype(int)+(x.ema50>x.ema100).astype(int)+(x.ema100>x.ema200).astype(int)
    x["distance_ema200"]=x.close/x.ema200-1
    x["rsi14"]=rsi(x.close)
    e12=x.close.ewm(span=12,adjust=False).mean(); e26=x.close.ewm(span=26,adjust=False).mean()
    macd=e12-e26; sig=macd.ewm(span=9,adjust=False).mean(); x["macd_hist"]=macd-sig
    return x

def base_mask(x):
    return x.h24.notna()&(x.close>x.h24*1.0005)&(x.vm>=1.25)&(x.comp>=12)&(x.r48<=.03)&(x.trend_stack_score>=3)&(x.distance_ema200>Q4)

def period(ts):
    return "DEV_2021_2024" if ts.year<=2024 else str(ts.year)

def confirm(x,i,mode):
    if mode=="IMMEDIATE": return i
    lvl=float(x.h24.iloc[i])
    for j in range(i+1,min(i+1+WINDOW,len(x))):
        r=x.iloc[j]; p=x.iloc[j-1]
        if mode=="FIRST_RED_CANDLE" and r.close<r.open: return j
        if mode=="BREAK_PREV_LOW" and r.close<p.low: return j
        if mode=="CLOSE_BACK_BELOW_BREAKOUT" and r.close<lvl: return j
        if mode=="UPPER_WICK_REJECTION":
            rng=r.high-r.low
            if rng>0:
                upper=r.high-max(r.open,r.close); cl=(r.close-r.low)/rng
                if upper/rng>=.40 and cl<=.50: return j
        if mode=="MACD_HIST_WEAKEN" and pd.notna(r.macd_hist) and pd.notna(p.macd_hist) and r.macd_hist<p.macd_hist: return j
        if mode=="RSI_ROLLOVER" and pd.notna(r.rsi14) and pd.notna(p.rsi14) and p.rsi14>=65 and r.rsi14<p.rsi14: return j

def run_mode(x,mode):
    out=[]
    for i in np.flatnonzero(base_mask(x).values):
        j=confirm(x,i,mode)
        if j is None: continue
        entry=float(x.close.iloc[j]); sl=entry*(1+STOP); tp=entry*(1-TARGET); last=min(j+HOLD,len(x)-1)
        ex=None
        for k in range(j+1,last+1):
            if x.high.iloc[k]>=sl: ex=(k,sl,"STOP"); break
            if x.low.iloc[k]<=tp: ex=(k,tp,"TARGET"); break
        if ex is None: ex=(last,float(x.close.iloc[last]),"TIME")
        k,px,why=ex; net=entry/px-1-FEE
        out.append({"confirmation":mode,"signal_time":x.index[i],"entry_time":x.index[j],"exit_time":x.index[k],
                    "entry_delay_h":j-i,"entry":entry,"exit":px,"reason":why,"net_return":net,"net_r":net/STOP,
                    "period":period(x.index[i]),"year":x.index[i].year})
    return pd.DataFrame(out)

def perf(g):
    if len(g)==0:return dict(trades=0,avg_net=np.nan,win_rate=np.nan,pf=np.nan,total_return=np.nan,max_dd=np.nan,avg_r=np.nan,avg_entry_delay_h=np.nan)
    r=g.net_return.to_numpy(); pos=r[r>0].sum(); neg=-r[r<0].sum(); eq=np.cumprod(1+r); pk=np.maximum.accumulate(eq)
    return dict(trades=len(g),avg_net=r.mean(),win_rate=(r>0).mean(),pf=pos/neg if neg else np.inf,total_return=eq[-1]-1,max_dd=(eq/pk-1).min(),avg_r=g.net_r.mean(),avg_entry_delay_h=g.entry_delay_h.mean())

def boot(g,n=10000):
    if len(g)==0:return np.nan,np.nan,np.nan
    a=g.net_return.to_numpy(); rng=np.random.default_rng(20260821); vals=np.empty(n)
    for k in range(n): vals[k]=a[rng.integers(0,len(a),len(a))].mean()
    lo,hi=np.quantile(vals,[.025,.975]); return lo,hi,(vals>0).mean()

def main():
    print("=== BTC Q4 confirmation study ===",flush=True)
    d=fetch(); print(f"Candles: {len(d):,} | {d.index.min()} -> {d.index.max()}",flush=True); x=feat(d)
    base=int(base_mask(x).sum()); print(f"Q4 base signals: {base}",flush=True)
    alltr=[]; counts=[]
    for m in MODES:
        t=run_mode(x,m); alltr.append(t); counts.append({"confirmation":m,"base_signals":base,"confirmed_trades":len(t),"confirmation_rate":len(t)/base if base else np.nan})
    T=pd.concat(alltr,ignore_index=True) if alltr else pd.DataFrame()
    T.to_csv(OUT/"btc_q4_confirm_trades.csv",index=False); C=pd.DataFrame(counts); C.to_csv(OUT/"btc_q4_confirm_signal_counts.csv",index=False)
    prow=[]; brow=[]
    for m in MODES:
        g=T[T.confirmation==m]
        for p in ["DEV_2021_2024","2025","2026","OOS_2025_2026"]:
            q=g[g.period==p] if p!="OOS_2025_2026" else g[g.period.isin(["2025","2026"])]
            s=perf(q); s.update({"confirmation":m,"period":p}); prow.append(s)
            if p!="DEV_2021_2024":
                lo,hi,pr=boot(q); brow.append({"confirmation":m,"period":p,"trades":len(q),"avg_net":q.net_return.mean() if len(q) else np.nan,"ci_low":lo,"ci_high":hi,"prob_avg_gt0":pr})
    P=pd.DataFrame(prow); B=pd.DataFrame(brow)
    P.to_csv(OUT/"btc_q4_confirm_performance.csv",index=False); B.to_csv(OUT/"btc_q4_confirm_bootstrap.csv",index=False)
    recent=T[pd.to_datetime(T.entry_time,utc=True)>=x.index.max()-pd.Timedelta(days=120)] if len(T) else T
    recent.to_csv(OUT/"btc_q4_confirm_recent.csv",index=False)
    report="BTC Q4 OVEREXTENSION CONFIRMATION STUDY\n"+"="*70+f"\nLocked Q4 distance>{Q4:.4%}; SHORT TP1% / SL1% / max12h / fee0.11%; confirmation window={WINDOW}h\n\nCOUNTS\n"+C.to_string(index=False)+"\n\nPERFORMANCE\n"+P.to_string(index=False)+"\n\nBOOTSTRAP\n"+B.to_string(index=False)
    (OUT/"btc_q4_confirm_summary.txt").write_text(report,encoding="utf-8"); print(report,flush=True); print(f"Saved to {OUT.resolve()}",flush=True)
if __name__=="__main__": main()

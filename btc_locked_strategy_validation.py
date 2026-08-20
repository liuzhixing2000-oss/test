import time,requests,numpy as np,pandas as pd
from datetime import datetime,timezone
from pathlib import Path
OUT=Path("btc_locked_results");OUT.mkdir(exist_ok=True)
STOP=.01; FEE=.0011
TARGETS=[.015,.02]; HOURS=[6,12,24]; MODES=["CLOSE","RETEST"]

def ms(s): return int(datetime.fromisoformat(s).replace(tzinfo=timezone.utc).timestamp()*1000)
def fetch():
 u="https://api.bybit.com/v5/market/kline"; start=ms("2021-01-01"); end=int(datetime.now(timezone.utc).timestamp()*1000); a=[]
 while end>start:
  r=requests.get(u,params={"category":"linear","symbol":"BTCUSDT","interval":"60","end":end,"limit":1000},timeout=20);r.raise_for_status();q=r.json()
  if q.get("retCode")!=0: raise RuntimeError(q)
  z=q["result"]["list"]
  if not z: break
  a+=z;o=min(int(v[0]) for v in z)
  if o<=start: break
  end=o-1;time.sleep(.05)
 c=["timestamp","open","high","low","close","volume","turnover"];d=pd.DataFrame(a,columns=c);d["timestamp"]=pd.to_numeric(d["timestamp"])
 for x in c[1:]:d[x]=pd.to_numeric(d[x],errors="coerce")
 d=d.drop_duplicates("timestamp").sort_values("timestamp");d=d[d.timestamp>=start];d["dt"]=pd.to_datetime(d.timestamp,unit="ms",utc=True);return d.set_index("dt")
def prank(s):
 def f(a): return pd.Series(a).rank(pct=True).iloc[-1]
 return s.rolling(720,min_periods=150).apply(f,raw=False)
def feat(d):
 x=d.copy()
 for n in [20,50,100,200]:x[f"e{n}"]=x.close.ewm(span=n,adjust=False).mean()
 m=x.close.rolling(20).mean();sd=x.close.rolling(20).std(ddof=0);x["br"]=prank(4*sd/m)
 f=(x.br<=.35).astype(int);g=(f!=f.shift()).cumsum();x["comp"]=f.groupby(g).cumsum().where(f.eq(1),0).shift(1).fillna(0)
 x["h24"]=x.high.shift().rolling(24).max();x["h48"]=x.high.shift().rolling(48).max();x["l48"]=x.low.shift().rolling(48).min();x["r48"]=(x.h48-x.l48)/x.l48
 x["vm"]=x.volume/x.volume.shift().rolling(24).mean()
 x["stack"]=(x.close>x.e20).astype(int)+(x.e20>x.e50).astype(int)+(x.e50>x.e100).astype(int)+(x.e100>x.e200).astype(int)
 x["ema50gt200"]=x.e50>x.e200
 return x
def sig(x):return (x.close>x.h24*1.0005)&(x.vm>=1.25)&(x.comp>=12)&(x.r48<=.03)&(x.stack>=3)
def entry(x,i,mode):
 if mode=="CLOSE":return i,float(x.close.iloc[i])
 lvl=float(x.h24.iloc[i])
 for j in range(i+1,min(i+7,len(x))):
  if x.low.iloc[j]<=lvl*1.0025 and x.close.iloc[j]>=lvl:return j,float(x.close.iloc[j])
def run(x,mode,tp,hrs):
 rows=[];busy=-1
 for i in np.flatnonzero(sig(x).values):
  if i<=busy:continue
  e=entry(x,i,mode)
  if not e:continue
  ei,p=e;sl=p*(1-STOP);tg=p*(1+tp);last=min(ei+hrs,len(x)-1);ex=None
  for j in range(ei+1,last+1):
   if x.low.iloc[j]<=sl:ex=(j,sl,"STOP");break
   if x.high.iloc[j]>=tg:ex=(j,tg,"TARGET");break
  if not ex:ex=(last,float(x.close.iloc[last]),f"TIME_{hrs}H")
  j,px,why=ex; net=px/p-1-FEE
  rows.append({"signal_time":x.index[i],"entry_time":x.index[ei],"exit_time":x.index[j],"entry":p,"exit":px,"reason":why,"net_return":net,"net_r":net/STOP,"mode":mode,"tp":tp,"hours":hrs})
  busy=j
 return pd.DataFrame(rows)
def stats(t):
 if len(t)==0:return dict(trades=0,avg_net=np.nan,win=np.nan,pf=np.nan,total=np.nan,mdd=np.nan,avg_r=np.nan,losestreak=np.nan)
 r=t.net_return.to_numpy();w=r[r>0].sum();l=-r[r<0].sum();eq=np.cumprod(1+r);pk=np.maximum.accumulate(eq);cur=mx=0
 for q in r:
  cur=cur+1 if q<0 else 0;mx=max(mx,cur)
 return dict(trades=len(r),avg_net=r.mean(),win=(r>0).mean(),pf=w/l if l else np.inf,total=eq[-1]-1,mdd=(eq/pk-1).min(),avg_r=t.net_r.mean(),losestreak=mx)
def boot(t,n=10000):
 if not len(t):return np.nan,np.nan,np.nan
 a=t.net_return.to_numpy();rng=np.random.default_rng(20260820);v=np.array([a[rng.integers(0,len(a),len(a))].mean() for _ in range(n)])
 return *np.quantile(v,[.025,.975]),(v>0).mean()
def main():
 print("=== BTC locked strategy validation ===",flush=True);d=fetch();print(f"Candles {len(d):,}: {d.index.min()} -> {d.index.max()}",flush=True);x=feat(d)
 periods=[("DEV","2021","2025"),("2025","2025","2026"),("2026","2026","2027"),("OOS","2025","2027"),("LAST12M","2025-08-20","2026-08-21"),("FULL","2021","2027")]
 vr=[];pr=[];br=[];all=[]
 for mode in MODES:
  for tp in TARGETS:
   for h in HOURS:
    v=f"{mode}_TP{tp*100:.1f}_T{h}H";t=run(x,mode,tp,h);t["variant"]=v;all.append(t);s=stats(t);s["variant"]=v;vr.append(s)
    for name,a,b in periods:
     q=t[(t.entry_time>=pd.Timestamp(a,tz="UTC"))&(t.entry_time<pd.Timestamp(b,tz="UTC"))];z=stats(q);z.update(variant=v,period=name);pr.append(z)
     if name in ["2025","2026","OOS","LAST12M"]:
      lo,hi,p=boot(q);br.append({"variant":v,"period":name,"trades":len(q),"avg_net":q.net_return.mean() if len(q) else np.nan,"ci_low":lo,"ci_high":hi,"prob_avg_gt0":p})
 V=pd.DataFrame(vr);P=pd.DataFrame(pr);B=pd.DataFrame(br);T=pd.concat(all,ignore_index=True)
 V.to_csv(OUT/"btc_locked_variant_results.csv",index=False);P.to_csv(OUT/"btc_locked_period_results.csv",index=False);B.to_csv(OUT/"btc_locked_bootstrap.csv",index=False);T.to_csv(OUT/"btc_locked_trades.csv",index=False)
 y=[] 
 for v,g in T.groupby("variant"):
  z=g.copy();z["year"]=z.entry_time.dt.year
  for yr,q in z.groupby("year"):s=stats(q);s.update(variant=v,year=yr);y.append(s)
 pd.DataFrame(y).to_csv(OUT/"btc_locked_yearly.csv",index=False)
 recent=T[T.entry_time>=x.index.max()-pd.Timedelta(days=30)];recent.to_csv(OUT/"btc_locked_recent_trades.csv",index=False)
 txt="BTC LOCKED BREAKOUT VALIDATION\n"+"="*60+"\nFilters: volume>=1.25x; compression>=12h; range48<=3%; trend stack>=3; SL=1%; fee=0.11% round trip\n12 predefined variants only.\n\nFULL:\n"+V.to_string(index=False)+"\n\n2025/2026/OOS:\n"+P[P.period.isin(["2025","2026","OOS"])].to_string(index=False)+"\n\nBOOTSTRAP:\n"+B.to_string(index=False)
 (OUT/"btc_locked_summary.txt").write_text(txt);print(txt,flush=True)
if __name__=="__main__":main()

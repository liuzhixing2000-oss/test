import time,requests,numpy as np,pandas as pd
from datetime import datetime,timezone
from pathlib import Path
O=Path("results");O.mkdir(exist_ok=True);H=[3,6,12,24,48]
B=[0,.5,.7,.8,.9,.95,.975,.99,1.000001];L=["0-50","50-70","70-80","80-90","90-95","95-97.5","97.5-99","99-100"]
def fetch():
 u="https://api.bybit.com/v5/market/kline";s=int(datetime(2021,1,1,tzinfo=timezone.utc).timestamp()*1000);e=int(datetime.now(timezone.utc).timestamp()*1000);a=[]
 while e>s:
  j=requests.get(u,params={"category":"linear","symbol":"BTCUSDT","interval":"60","end":e,"limit":1000},timeout=20).json();p=j["result"]["list"]
  if not p:break
  a+=p;o=min(int(x[0]) for x in p)
  if o<=s:break
  e=o-1;time.sleep(.05)
 c=["ts","open","high","low","close","volume","turnover"];d=pd.DataFrame(a,columns=c)
 for z in c:d[z]=pd.to_numeric(d[z])
 d=d.drop_duplicates("ts").sort_values("ts");d=d[d.ts>=s];d["dt"]=pd.to_datetime(d.ts,unit="ms",utc=True);return d.set_index("dt")
def main():
 print("=== BTC EMA200 overextension response curve ===",flush=True);x=fetch();x["ema200"]=x.close.ewm(span=200,adjust=False).mean();x["dist"]=x.close/x.ema200-1
 # causal percentile: rank current distance against history available at that hour
 vals=x.dist.to_numpy();pct=np.full(len(x),np.nan)
 for i in range(719,len(x)):pct[i]=(vals[:i+1]<=vals[i]).mean()
 x["pct"]=pct;x["band"]=pd.cut(x.pct,B,labels=L,include_lowest=True,right=False)
 for h in H:x[f"r{h}"]=x.close.shift(-h)/x.close-1
 x["period"]=np.where(x.index.year<=2024,"DEV_2021_2024",x.index.year.astype(str))
 x=x.dropna(subset=["band","r48"]);x.to_csv(O/"observations.csv")
 rows=[]
 for p in ["DEV_2021_2024","2025","2026","OOS_2025_2026"]:
  q=x[x.period.isin(["2025","2026"])] if p=="OOS_2025_2026" else x[x.period==p]
  for b in L:
   g=q[q.band.astype(str)==b];r={"period":p,"band":b,"n":len(g),"mean_dist":g.dist.mean() if len(g) else np.nan}
   for h in H:r[f"avg_{h}h"]=g[f"r{h}"].mean() if len(g) else np.nan;r[f"p_negative_{h}h"]=(g[f"r{h}"]<0).mean() if len(g) else np.nan
   rows.append(r)
 S=pd.DataFrame(rows);S.to_csv(O/"response_curve.csv",index=False)
 Y=[]
 for y in sorted(x.index.year.unique()):
  q=x[x.index.year==y]
  for b in ["90-95","95-97.5","97.5-99","99-100"]:
   g=q[q.band.astype(str)==b];Y.append({"year":y,"band":b,"n":len(g),"avg_6h":g.r6.mean() if len(g) else np.nan,"avg_12h":g.r12.mean() if len(g) else np.nan,"avg_24h":g.r24.mean() if len(g) else np.nan})
 Y=pd.DataFrame(Y);Y.to_csv(O/"yearly_top_tail.csv",index=False)
 txt="OOS RESPONSE CURVE\n"+S[S.period=="OOS_2025_2026"].to_string(index=False)+"\n\nYEARLY TOP TAIL\n"+Y.to_string(index=False);(O/"summary.txt").write_text(txt);print(txt,flush=True)
if __name__=="__main__":main()

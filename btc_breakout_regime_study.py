import time, math, requests, numpy as np, pandas as pd
from datetime import datetime, timezone
from pathlib import Path
OUT=Path('btc_regime_results'); OUT.mkdir(exist_ok=True)
STOP=.01; TARGET=.015; TIME_STOP=12; FEE=.0011
FEATURES=['atr_pct','atr_rank_30d','rv_24h','rv_7d','rv_30d','rv_ratio_7d_30d','bbw','bbw_rank_30d','compression_hours','ret_24h','ret_72h','ret_7d','ret_30d','distance_ema20','distance_ema50','distance_ema200','ema20_slope_6h','ema50_slope_24h','ema200_slope_72h','trend_stack_score','volume_mult','breakout_pct','body_pct','upper_wick_ratio','lower_wick_ratio','close_location','range_atr','rsi14','macd_hist_norm']
def ms(s): return int(datetime.fromisoformat(s).replace(tzinfo=timezone.utc).timestamp()*1000)
def fetch():
 u='https://api.bybit.com/v5/market/kline'; start=ms('2021-01-01'); end=int(datetime.now(timezone.utc).timestamp()*1000); rows=[]
 while end>start:
  r=requests.get(u,params={'category':'linear','symbol':'BTCUSDT','interval':'60','end':end,'limit':1000},timeout=20); r.raise_for_status(); js=r.json()
  if js.get('retCode')!=0: raise RuntimeError(js)
  p=js['result']['list']
  if not p: break
  rows+=p; oldest=min(int(v[0]) for v in p)
  if oldest<=start: break
  end=oldest-1; time.sleep(.05)
 c=['timestamp','open','high','low','close','volume','turnover']; d=pd.DataFrame(rows,columns=c); d['timestamp']=pd.to_numeric(d['timestamp']).astype('int64')
 for x in c[1:]: d[x]=pd.to_numeric(d[x],errors='coerce')
 d=d.drop_duplicates('timestamp').sort_values('timestamp'); d=d[d.timestamp>=start]; d['dt']=pd.to_datetime(d.timestamp,unit='ms',utc=True); d=d.set_index('dt')
 now=int(datetime.now(timezone.utc).timestamp()*1000); return d[d.timestamp+3600000<=now]
def prank(s):
 def f(a): return pd.Series(a).rank(pct=True).iloc[-1]
 return s.rolling(720,min_periods=150).apply(f,raw=False)
def rsi(c,n=14):
 d=c.diff(); g=d.clip(lower=0); l=-d.clip(upper=0); ag=g.ewm(alpha=1/n,adjust=False).mean(); al=l.ewm(alpha=1/n,adjust=False).mean(); rs=ag/al.replace(0,np.nan); return 100-100/(1+rs)
def feat(d):
 x=d.copy();
 for n in [20,50,100,200]: x[f'ema{n}']=x.close.ewm(span=n,adjust=False).mean()
 for h,nm in [(24,'ret_24h'),(72,'ret_72h'),(168,'ret_7d'),(720,'ret_30d')]: x[nm]=x.close/x.close.shift(h)-1
 lr=np.log(x.close/x.close.shift(1)); x['rv_24h']=lr.rolling(24).std(ddof=0)*np.sqrt(24); x['rv_7d']=lr.rolling(168).std(ddof=0)*np.sqrt(168); x['rv_30d']=lr.rolling(720).std(ddof=0)*np.sqrt(720); x['rv_ratio_7d_30d']=x.rv_7d/x.rv_30d
 prev=x.close.shift(); tr=pd.concat([x.high-x.low,(x.high-prev).abs(),(x.low-prev).abs()],axis=1).max(axis=1); atr=tr.rolling(14).mean(); x['atr_pct']=atr/x.close; x['atr_rank_30d']=prank(x.atr_pct)
 mid=x.close.rolling(20).mean(); sd=x.close.rolling(20).std(ddof=0); x['bbw']=4*sd/mid; x['bbw_rank_30d']=prank(x.bbw)
 f=(x.bbw_rank_30d<=.35).astype(int); g=(f!=f.shift()).cumsum(); x['compression_hours']=f.groupby(g).cumsum().where(f.eq(1),0).shift(1).fillna(0)
 x['prior_high24']=x.high.shift().rolling(24).max(); x['prior_high48']=x.high.shift().rolling(48).max(); x['prior_low48']=x.low.shift().rolling(48).min(); x['range48']=(x.prior_high48-x.prior_low48)/x.prior_low48
 x['vol_ma24']=x.volume.shift().rolling(24).mean(); x['volume_mult']=x.volume/x.vol_ma24; x['breakout_pct']=x.close/x.prior_high24-1
 x['trend_stack_score']=(x.close>x.ema20).astype(int)+(x.ema20>x.ema50).astype(int)+(x.ema50>x.ema100).astype(int)+(x.ema100>x.ema200).astype(int)
 x['distance_ema20']=x.close/x.ema20-1; x['distance_ema50']=x.close/x.ema50-1; x['distance_ema200']=x.close/x.ema200-1; x['ema20_slope_6h']=x.ema20/x.ema20.shift(6)-1; x['ema50_slope_24h']=x.ema50/x.ema50.shift(24)-1; x['ema200_slope_72h']=x.ema200/x.ema200.shift(72)-1
 cr=(x.high-x.low).replace(0,np.nan); body=(x.close-x.open).abs(); up=x.high-x[['open','close']].max(axis=1); lo=x[['open','close']].min(axis=1)-x.low; x['body_pct']=body/x.open; x['upper_wick_ratio']=up/cr; x['lower_wick_ratio']=lo/cr; x['close_location']=(x.close-x.low)/cr; x['range_atr']=cr/atr
 x['rsi14']=rsi(x.close); e12=x.close.ewm(span=12,adjust=False).mean(); e26=x.close.ewm(span=26,adjust=False).mean(); m=e12-e26; s=m.ewm(span=9,adjust=False).mean(); x['macd_hist_norm']=(m-s)/x.close
 return x
def sig(x): return x.prior_high24.notna()&(x.close>x.prior_high24*1.0005)&(x.volume_mult>=1.25)&(x.compression_hours>=12)&(x.range48<=.03)&(x.trend_stack_score>=3)
def sim(x):
 rows=[]; busy=-1
 for i in np.flatnonzero(sig(x).values):
  if i<=busy: continue
  entry=float(x.close.iloc[i]); sl=entry*(1-STOP); tg=entry*(1+TARGET); last=min(i+TIME_STOP,len(x)-1); ex=None
  for j in range(i+1,last+1):
   r=x.iloc[j]
   if r.low<=sl: ex=(j,sl,'STOP'); break
   if r.high>=tg: ex=(j,tg,'TARGET'); break
  if ex is None: ex=(last,float(x.close.iloc[last]),'TIME_12H')
  j,px,why=ex; net=px/entry-1-FEE; rec={'signal_time':x.index[i],'entry_time':x.index[i],'exit_time':x.index[j],'entry':entry,'exit':px,'reason':why,'net_return':net,'net_r':net/STOP,'outcome':'WIN' if net>0 else 'LOSS','year':x.index[i].year,'quarter':f"{x.index[i].year}Q{((x.index[i].month-1)//3)+1}"}
  for f in FEATURES: rec[f]=x.iloc[i].get(f,np.nan)
  rows.append(rec); busy=j
 return pd.DataFrame(rows)
def cd(a,b):
 a=pd.to_numeric(pd.Series(a),errors='coerce').dropna(); b=pd.to_numeric(pd.Series(b),errors='coerce').dropna()
 if len(a)<2 or len(b)<2:return np.nan
 p=math.sqrt(((len(a)-1)*a.var(ddof=1)+(len(b)-1)*b.var(ddof=1))/max(len(a)+len(b)-2,1)); return (a.mean()-b.mean())/p if p and np.isfinite(p) else np.nan
def main():
 print('=== BTC breakout regime study ===',flush=True); d=fetch(); print(f'Candles: {len(d):,} | {d.index.min()} -> {d.index.max()}',flush=True); x=feat(d); t=sim(x); t.to_csv(OUT/'btc_regime_trades.csv',index=False)
 out=[]
 for pname,mask in [('DEV_2021_2024',t.year<=2024),('2025',t.year==2025),('2026',t.year==2026),('OOS_2025_2026',t.year>=2025)]:
  g=t[mask]; w=g[g.outcome=='WIN']; l=g[g.outcome=='LOSS']
  for f in FEATURES:
   a=w[f]; b=l[f]; out.append({'period':pname,'feature':f,'n_win':a.notna().sum(),'n_loss':b.notna().sum(),'win_mean':pd.to_numeric(a,errors='coerce').mean(),'loss_mean':pd.to_numeric(b,errors='coerce').mean(),'difference_win_minus_loss':pd.to_numeric(a,errors='coerce').mean()-pd.to_numeric(b,errors='coerce').mean(),'cohens_d':cd(a,b)})
 od=pd.DataFrame(out); od.to_csv(OUT/'btc_regime_feature_by_outcome.csv',index=False)
 shifts=[]
 for f in FEATURES:
  a=t.loc[t.year==2025,f]; b=t.loc[t.year==2026,f]; d0=cd(a,b); shifts.append({'feature':f,'mean_2025':pd.to_numeric(a,errors='coerce').mean(),'mean_2026':pd.to_numeric(b,errors='coerce').mean(),'difference_2025_minus_2026':pd.to_numeric(a,errors='coerce').mean()-pd.to_numeric(b,errors='coerce').mean(),'cohens_d_2025_vs_2026':d0,'abs_d':abs(d0) if pd.notna(d0) else np.nan})
 sh=pd.DataFrame(shifts).sort_values('abs_d',ascending=False); sh.to_csv(OUT/'btc_regime_2025_vs_2026_shift.csv',index=False)
 stab=[]; dev=od[od.period=='DEV_2021_2024'].set_index('feature'); y25=od[od.period=='2025'].set_index('feature'); y26=od[od.period=='2026'].set_index('feature')
 for f in dev.index.intersection(y25.index).intersection(y26.index):
  ds=[dev.loc[f,'cohens_d'],y25.loc[f,'cohens_d'],y26.loc[f,'cohens_d']]; same=all(pd.notna(z) for z in ds) and np.sign(ds[0])==np.sign(ds[1])==np.sign(ds[2]); stab.append({'feature':f,'dev_d':ds[0],'d_2025':ds[1],'d_2026':ds[2],'same_direction_all':same,'min_abs_d':min(map(abs,ds)) if same else 0})
 st=pd.DataFrame(stab).sort_values(['same_direction_all','min_abs_d'],ascending=[False,False]); st.to_csv(OUT/'btc_regime_feature_stability.csv',index=False)
 yr=[]
 for y,g in t.groupby('year'):
  r=g.net_return; wins=r[r>0].sum(); losses=-r[r<0].sum(); yr.append({'year':y,'trades':len(g),'avg_net':r.mean(),'win_rate':(r>0).mean(),'pf':wins/losses if losses>0 else np.inf,'total_return':np.prod(1+r)-1})
 ydf=pd.DataFrame(yr); ydf.to_csv(OUT/'btc_regime_year_summary.csv',index=False)
 recent=t[pd.to_datetime(t.entry_time,utc=True)>=x.index.max()-pd.Timedelta(days=60)]; recent.to_csv(OUT/'btc_regime_recent_trades.csv',index=False)
 txt='BTC BREAKOUT REGIME STUDY\n'+'='*72+'\nLocked strategy: CLOSE / TP1.5% / SL1% / 12h / fee0.11%\n\nYEAR SUMMARY\n'+ydf.to_string(index=False)+'\n\nTOP 2025->2026 REGIME SHIFTS\n'+sh.head(15).to_string(index=False)+'\n\nSTABLE WIN/LOSS FEATURES\n'+st[st.same_direction_all].head(20).to_string(index=False)+'\n\n2025 OUTCOME FEATURES\n'+od[od.period=='2025'].assign(abs_d=lambda d:d.cohens_d.abs()).sort_values('abs_d',ascending=False).head(15).to_string(index=False)+'\n\n2026 OUTCOME FEATURES\n'+od[od.period=='2026'].assign(abs_d=lambda d:d.cohens_d.abs()).sort_values('abs_d',ascending=False).head(15).to_string(index=False)
 (OUT/'btc_regime_summary.txt').write_text(txt); print(txt,flush=True); print(f'Saved to {OUT.resolve()}',flush=True)
if __name__=='__main__': main()

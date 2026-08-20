import time, requests, numpy as np, pandas as pd
from datetime import datetime, timezone
from pathlib import Path

OUT=Path('btc_q4_short_results'); OUT.mkdir(exist_ok=True)
START='2021-01-01'; FEE=.0011; STOP=.01
TARGETS=[.01,.015]; HOURS=[12,24]; HORIZONS=[6,12,24,48]
Q4_DISTANCE=.031470; Q4_EMA50=.004034; Q4_EMA200=.022135

def ms(s): return int(datetime.fromisoformat(s).replace(tzinfo=timezone.utc).timestamp()*1000)

def fetch():
    url='https://api.bybit.com/v5/market/kline'; start=ms(START); end=int(datetime.now(timezone.utc).timestamp()*1000); rows=[]
    while end>start:
        r=requests.get(url,params={'category':'linear','symbol':'BTCUSDT','interval':'60','end':end,'limit':1000},timeout=20); r.raise_for_status(); js=r.json()
        if js.get('retCode')!=0: raise RuntimeError(js)
        part=js['result']['list']
        if not part: break
        rows += part; oldest=min(int(v[0]) for v in part)
        if oldest<=start: break
        end=oldest-1; time.sleep(.05)
    c=['timestamp','open','high','low','close','volume','turnover']; d=pd.DataFrame(rows,columns=c); d['timestamp']=pd.to_numeric(d['timestamp']).astype('int64')
    for x in c[1:]: d[x]=pd.to_numeric(d[x],errors='coerce')
    d=d.drop_duplicates('timestamp').sort_values('timestamp'); d=d[d.timestamp>=start]; d['dt']=pd.to_datetime(d.timestamp,unit='ms',utc=True); d=d.set_index('dt')
    now=int(datetime.now(timezone.utc).timestamp()*1000); return d[d.timestamp+3600000<=now]

def prank(s,window=720,minp=150):
    def f(a): return pd.Series(a).rank(pct=True).iloc[-1]
    return s.rolling(window,min_periods=minp).apply(f,raw=False)

def feat(d):
    x=d.copy()
    for n in [20,50,100,200]: x[f'ema{n}']=x.close.ewm(span=n,adjust=False).mean()
    mid=x.close.rolling(20).mean(); sd=x.close.rolling(20).std(ddof=0); x['bbw_rank']=prank(4*sd/mid)
    f=(x.bbw_rank<=.35).astype(int); g=(f!=f.shift()).cumsum(); x['compression_hours']=f.groupby(g).cumsum().where(f.eq(1),0).shift().fillna(0)
    x['h24']=x.high.shift().rolling(24).max(); x['h48']=x.high.shift().rolling(48).max(); x['l48']=x.low.shift().rolling(48).min(); x['range48']=(x.h48-x.l48)/x.l48
    x['volume_mult']=x.volume/x.volume.shift().rolling(24).mean(); x['trend_stack_score']=(x.close>x.ema20).astype(int)+(x.ema20>x.ema50).astype(int)+(x.ema50>x.ema100).astype(int)+(x.ema100>x.ema200).astype(int)
    x['distance_ema200']=x.close/x.ema200-1; x['ema50_slope_24h']=x.ema50/x.ema50.shift(24)-1; x['ema200_slope_72h']=x.ema200/x.ema200.shift(72)-1
    return x

def base(x): return x.h24.notna()&(x.close>x.h24*1.0005)&(x.volume_mult>=1.25)&(x.compression_hours>=12)&(x.range48<=.03)&(x.trend_stack_score>=3)

def masks(x):
    b=base(x)
    return {'Q4_DISTANCE':b&(x.distance_ema200>Q4_DISTANCE),'Q4_EMA50':b&(x.ema50_slope_24h>Q4_EMA50),'Q4_EMA200':b&(x.ema200_slope_72h>Q4_EMA200),'Q4_DISTANCE_AND_EMA50':b&(x.distance_ema200>Q4_DISTANCE)&(x.ema50_slope_24h>Q4_EMA50)}

def per(ts):
    if ts.year<=2024:return 'DEV_2021_2024'
    if ts.year==2025:return '2025'
    if ts.year==2026:return '2026'
    return str(ts.year)

def forward(x):
    rows=[]
    for typ,m in masks(x).items():
        for i in np.flatnonzero(m.values):
            if i+48>=len(x): continue
            e=float(x.close.iloc[i]); r={'signal_type':typ,'signal_time':x.index[i],'entry':e,'period':per(x.index[i])}
            for h in HORIZONS:
                f=x.iloc[i+1:i+1+h]; r[f'ret_{h}h']=float(x.close.iloc[i+h]/e-1); r[f'mfe_{h}h']=float(f.high.max()/e-1); r[f'mae_{h}h']=float(f.low.min()/e-1)
                first='NEITHER'
                for _,z in f.iterrows():
                    up=z.high>=e*1.01; dn=z.low<=e*.99
                    if up and dn: first='UP_FIRST'; break
                    if up: first='UP_FIRST'; break
                    if dn: first='DOWN_FIRST'; break
                r[f'first_{h}h']=first
            rows.append(r)
    return pd.DataFrame(rows)

def short(x,typ,m,tp,h):
    rows=[]; busy=-1
    for i in np.flatnonzero(m.values):
        if i<=busy: continue
        e=float(x.close.iloc[i]); sl=e*(1+STOP); tg=e*(1-tp); last=min(i+h,len(x)-1); ex=None
        for j in range(i+1,last+1):
            if x.high.iloc[j]>=sl: ex=(j,sl,'STOP'); break
            if x.low.iloc[j]<=tg: ex=(j,tg,'TARGET'); break
        if ex is None: ex=(last,float(x.close.iloc[last]),f'TIME_{h}H')
        j,px,why=ex; net=e/px-1-FEE; rows.append({'signal_type':typ,'entry_time':x.index[i],'exit_time':x.index[j],'reason':why,'tp':tp,'hours':h,'net_return':net,'net_r':net/STOP,'period':per(x.index[i])}); busy=j
    return pd.DataFrame(rows)

def perf(g):
    if g.empty:return {'trades':0,'avg_net':np.nan,'win_rate':np.nan,'pf':np.nan,'total_return':np.nan,'avg_r':np.nan}
    r=g.net_return.to_numpy(); p=r[r>0].sum(); n=-r[r<0].sum(); return {'trades':len(r),'avg_net':r.mean(),'win_rate':(r>0).mean(),'pf':p/n if n else np.inf,'total_return':np.prod(1+r)-1,'avg_r':g.net_r.mean()}

def boot(g,n=10000):
    if g.empty:return (np.nan,np.nan,np.nan)
    a=g.net_return.to_numpy(); rng=np.random.default_rng(20260821); vals=np.array([a[rng.integers(0,len(a),len(a))].mean() for _ in range(n)]); lo,hi=np.quantile(vals,[.025,.975]); return lo,hi,(vals>0).mean()

def main():
    print('=== BTC Q4 overextension -> SHORT validation ===',flush=True); d=fetch(); print(f'Candles: {len(d):,} | {d.index.min()} -> {d.index.max()}',flush=True); x=feat(d)
    ev=forward(x); ev.to_csv(OUT/'btc_q4_forward_paths.csv',index=False)
    ps=[]
    for typ,g in ev.groupby('signal_type'):
        for period in ['DEV_2021_2024','2025','2026','OOS_2025_2026']:
            p=g[g.period.isin(['2025','2026'])] if period=='OOS_2025_2026' else g[g.period==period]; row={'signal_type':typ,'period':period,'events':len(p)}
            for h in HORIZONS:
                row[f'avg_ret_{h}h']=p[f'ret_{h}h'].mean(); row[f'avg_mfe_{h}h']=p[f'mfe_{h}h'].mean(); row[f'avg_mae_{h}h']=p[f'mae_{h}h'].mean(); row[f'down_first_{h}h']=(p[f'first_{h}h']=='DOWN_FIRST').mean() if len(p) else np.nan; row[f'up_first_{h}h']=(p[f'first_{h}h']=='UP_FIRST').mean() if len(p) else np.nan
            ps.append(row)
    P=pd.DataFrame(ps); P.to_csv(OUT/'btc_q4_forward_summary.csv',index=False)
    all=[]; rows=[]; bs=[]
    for typ,m in masks(x).items():
        for tp in TARGETS:
            for h in HOURS:
                t=short(x,typ,m,tp,h); v=f'{typ}_SHORT_TP{tp*100:.1f}_T{h}H'; t['variant']=v; all.append(t)
                for period in ['DEV_2021_2024','2025','2026','OOS_2025_2026']:
                    q=t[t.period.isin(['2025','2026'])] if period=='OOS_2025_2026' else t[t.period==period]; s=perf(q); s.update({'variant':v,'period':period}); rows.append(s)
                    if period in ['2025','2026','OOS_2025_2026']:
                        lo,hi,p=boot(q); bs.append({'variant':v,'period':period,'trades':len(q),'avg_net':q.net_return.mean() if len(q) else np.nan,'ci_low':lo,'ci_high':hi,'prob_avg_gt0':p})
    T=pd.concat(all,ignore_index=True) if all else pd.DataFrame(); T.to_csv(OUT/'btc_q4_short_trades.csv',index=False); R=pd.DataFrame(rows); R.to_csv(OUT/'btc_q4_short_performance.csv',index=False); B=pd.DataFrame(bs); B.to_csv(OUT/'btc_q4_short_bootstrap.csv',index=False)
    txt='BTC Q4 OVEREXTENSION -> SHORT VALIDATION\n'+'='*70+f'\nLocked boundaries: distance>{Q4_DISTANCE:.4%}; ema50 slope>{Q4_EMA50:.4%}; ema200 slope>{Q4_EMA200:.4%}\n\nFORWARD PATHS\n'+P.to_string(index=False)+'\n\nSHORT PERFORMANCE\n'+R.to_string(index=False)+'\n\nBOOTSTRAP\n'+B.to_string(index=False)
    (OUT/'btc_q4_short_summary.txt').write_text(txt); print(txt,flush=True); print(f'Saved to {OUT.resolve()}',flush=True)
if __name__=='__main__': main()

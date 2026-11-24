"""Repeat recovery archive replay at wider assumed publication delays."""
from pathlib import Path
import argparse
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
HERE=Path(__file__).resolve().parent
A=argparse.ArgumentParser()
A.add_argument('--out',type=Path,default=HERE/'reproduced/extended_lags')
D=A.parse_args().out
D.mkdir(parents=True,exist_ok=True)
I=HERE/'inputs/recovery_reconciled_reports.csv'
H='hurricane_helene_2024'; K=['event_id','county_fips']; TS='report_timestamp_utc'
C=['cell_sites_out_due_to_'+s for s in ['power','transport','damage']]
KEY=K+['target_utc']; LAGS=[0,6,11,12,18,24,30,35,36,48,59,60,72]
r=pd.read_csv(I,dtype={'county_fips':str})
for c in [TS,'landfall_utc']:r[c]=pd.to_datetime(r[c],utc=True,format='mixed')
r['pct']=100*r.cell_sites_out/r.cell_sites_served
rows=[]
for (ev,c),g in r.sort_values(K+[TS]).groupby(K):
    land=g.landfall_utc.iloc[0]
    for _,z in g[g[TS].between(land,land+pd.Timedelta(days=11))].iterrows():
        t=z[TS]; issue=t.normalize(); day=(issue-land).days
        horizon=(t-issue).total_seconds()/3600
        if not 0<=day<10 or not 8<=horizon<=22:continue
        for lag in LAGS:
            eligible=g[g[TS].between(issue-pd.Timedelta(hours=72),issue-pd.Timedelta(hours=lag))]
            if eligible.empty:continue
            p=eligible.iloc[-1]
            if not (p[C].notna().all() and p[C].le(p.cell_sites_out).all()):continue
            dt=(p[TS]-eligible.iloc[-2][TS]).total_seconds()/3600 if len(eligible)>1 else np.nan
            trend=(p.pct-eligible.iloc[-2].pct)*24/dt if 6<=dt<=60 else np.nan
            rows.append(dict(event_id=ev,county_fips=c,target_utc=t,issue_utc=issue,prior_utc=p[TS],
                lag=lag,age=(issue-p[TS]).total_seconds()/3600,horizon=horizon,day=day,prior=p.pct,
                target=z.pct,trend=trend,stable=p.cell_sites_served==z.cell_sites_served,
                power=100*p[C[0]]/p.cell_sites_served,transport=100*p[C[1]]/p.cell_sites_served,damage=100*p[C[2]]/p.cell_sites_served))
v=pd.DataFrame(rows)
v.to_csv(D/'extended lag eligible pairs.csv',index=False)
scopes={'all eligible first ten days':v,'days 3–9 with equal prior and target totals':v[v.day.between(3,9)&v.stable]}
core=[6,12,24,36,48]
def subsets(z,lag):
    q=z[z.lag.eq(lag)].set_index(KEY,drop=False)
    assert q.index.is_unique
    return q[q.event_id.ne(H)],q[q.event_id.eq(H)]
def fit(a,b,ext):
    w=a.event_id.map(1/a.event_id.value_counts()).to_numpy();w=w/w.mean()
    def prep(q):
        f=q[['prior','trend','age',*ext]].copy()
        for c in ['prior','age',*ext]:f[c]=np.log1p(f[c])
        f['trend_missing']=f.trend.isna().astype(int)
        return f
    x,y=prep(a),prep(b)
    m=x.median().fillna(0)
    x=x.fillna(m).to_numpy();y=y.fillna(m).to_numpy()
    mu=np.average(x,axis=0,weights=w)
    std=np.sqrt(np.average((x-mu)**2,axis=0,weights=w));std[std<1e-10]=1
    with np.errstate(all='ignore'):
        mod=Ridge(alpha=.10*len(a)).fit((x-mu)/std,np.log1p(a.target)-np.log1p(a.prior),sample_weight=w)
        return np.clip(np.expm1(np.log1p(b.prior)+mod.predict((y-mu)/std)),0,100)
result=[]; pred=[]
for scope,z in scopes.items():
    data={lag:subsets(z,lag) for lag in LAGS}
    both=[data[l] for l in core]
    commontrain,commontest=both[0][0].index,both[0][1].index
    for a,b in both[1:]:
        commontrain=commontrain.intersection(a.index)
        commontest=commontest.intersection(b.index)
    for lag,(a,b) in data.items():
        sets={'available at each lag':(a,b)}
        if lag in core:
            sets['same historical and Helene keys at 6–48 hours']=(a.loc[commontrain],b.loc[commontest])
        for kind,(x,y) in sets.items():
            info=dict(scope=scope,comparison=kind,lag_h=lag,train_n=len(x),train_storms=x.event_id.nunique(),test_n=len(y),
                      test_counties=y.county_fips.nunique(),median_prior_age_h=y.age.median(),median_target_lead_h=y.horizon.median())
            if not len(y):
                result.append(dict(**info,model='none',mae_pp=np.nan));continue
            mods={'persistence':y.prior.to_numpy()}
            if x.event_id.nunique()>=2:
                mods.update(history=fit(x,y,[]),transport=fit(x,y,['transport']),all_causes=fit(x,y,['power','transport','damage']))
            for m,q in mods.items():
                result.append(dict(**info,model=m,mae_pp=np.mean(abs(q-y.target))))
                pred.append(y.reset_index(drop=True)[KEY+['prior_utc','target','age']].assign(scope=scope,comparison=kind,lag_h=lag,model=m,predicted=np.asarray(q)))
q=pd.DataFrame(result)
q.to_csv(D/'extended lag scores.csv',index=False)
pd.concat(pred).to_csv(D/'extended lag predictions.csv',index=False)
a=q.query('scope=="all eligible first ten days" and comparison=="available at each lag" and lag_h==6')
assert np.isclose(a.query('model=="persistence"').mae_pp.iloc[0],13.768347150554378)
assert np.isclose(a.query('model=="transport"').mae_pp.iloc[0],26.72453988326774)
for key, df in q[q.lag_h.isin([0,6,12,24,36,48,60])].groupby(['scope','comparison'],sort=False):
 print(key); print(df.pivot(index=['lag_h','train_n','train_storms','test_n','median_prior_age_h'],columns='model',values='mae_pp').round(3).to_string())

expect=pd.read_csv(HERE/'expected/extended_lag_scores.csv')
idx=['scope','comparison','lag_h','model']
q=q.set_index(idx).sort_index()
expect=expect.set_index(idx).sort_index()
assert expect.index.equals(q.index)
assert np.allclose(expect.mae_pp,q.mae_pp,equal_nan=True)
print('Verified all extended delay comparisons.')

"""Rebuild paper bench scores from bundled derived inputs. No source tree or network."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge

HERE=Path(__file__).resolve().parent
IN=HERE/'inputs'
A=argparse.ArgumentParser()
A.add_argument('--out',type=Path,default=HERE/'reproduced')
args=A.parse_args(); OUT=args.out; OUT.mkdir(parents=True,exist_ok=True)
H='hurricane_helene_2024'
L=pd.Timestamp('2024-09-27T03:10Z')
K=['event_id','county_fips']; F=K[1]; TS='report_timestamp_utc'
W=['wind_wsp34_county_mean_p_lower','rain_risk_rank']
TERRA=['flood_static_slope_pct','flood_static_low_slope_share']
PCT=lambda d: 100*d.cell_sites_out/d.cell_sites_served
read=lambda f: pd.read_csv(IN/f,dtype={F:str})
def time(x):
    return pd.to_datetime(x,format='mixed',utc=True)
def save(d,f):
    d.to_csv(OUT/f,index=False)
def ridge(a,b,cols,y,transforms=(),has_trend=False):
    w=1/a.groupby('event_id').event_id.transform('size')
    w=w/w.mean()
    x=a[cols].astype(float).copy(); v=b[cols].astype(float).copy()
    for c in transforms:
        x[c]=np.log1p(x[c]);v[c]=np.log1p(v[c])
    if has_trend:
        for z in (x,v):z['trend_missing']=z.prior_slope_pp_day.isna().astype(int)
    m=x.median().fillna(0)
    x=x.fillna(m).to_numpy();v=v.fillna(m).to_numpy()
    mn=np.average(x,axis=0,weights=w)
    sd=np.sqrt(np.average((x-mn)**2,axis=0,weights=w));sd[sd<1e-10]=1
    with np.errstate(all='ignore'):
        result=Ridge(alpha=.1*len(a)).fit((x-mn)/sd,y,sample_weight=w).predict((v-mn)/sd)
    assert np.isfinite(result).all()
    return result
def predict_peak(a,b,cols):
    return np.clip(np.expm1(ridge(a,b,cols,np.log1p(PCT(a)))),0,100)
def scores(a,obs='observed_peak_pct',pred='forecast_peak_pct'):
    y=a[obs];z=a[pred]
    top=lambda c:set(a.sort_values([c,F],ascending=[False,True])[F].head(5))
    return {'n':len(a),'mae_pp':abs(y-z).mean(),'bias_pp':(z-y).mean(),
        'rho':spearmanr(y,z).statistic if y.nunique()>1 and z.nunique()>1 else np.nan,'top5_overlap':len(top(obs)&top(pred))}
d=read('helene_fcc_daily.csv')
d[TS]=time(d[TS]); d['peak_pct']=PCT(d)
d['hours']=(d[TS]-L).dt.total_seconds()/3600
assert len(d)==420 and d[TS].nunique()==24 and not d.duplicated([F,TS]).any()
def handoff(lo=0,hi=168,stable=False,omit=None):
    q=d[d.hours.between(lo,hi)&~d[TS].eq(omit)].copy()
    if lo: q=q[q.hours.gt(lo)].copy()
    if stable:
        g=q.groupby(F).cell_sites_served.agg(['nunique','size'])
        q=q[q[F].isin(g[(g['nunique']==1)&(g['size']==7)].index)]
    n=q.groupby(F).size().rename('n_full_qualifying')
    return q.sort_values(['peak_pct',TS],ascending=[False,True]).drop_duplicates(F).merge(n,on=F)[
        [F,'peak_pct',TS,'source_url','n_full_qualifying']].rename(columns={'peak_pct':'observed_peak_pct',TS:'full_peak_time','source_url':'full_peak_source'})
training=read('peak_training.csv')
ht=read('peak_helene_weather.csv')
assert (len(training),training.event_id.nunique(),H in set(training.event_id))==(589,11,False)
forecast=ht[[F]].assign(forecast_peak_pct=predict_peak(training,ht,W))
def peakjoin(obs,fc=forecast):
    return obs.merge(fc,on=F,validate='one_to_one')
full=peakjoin(handoff()).sort_values(['observed_peak_pct',F],ascending=[False,True])
full['absolute_error_pp']=(full.observed_peak_pct-full.forecast_peak_pct).abs()
save(full[['county_fips','observed_peak_pct','forecast_peak_pct','full_peak_time','full_peak_source',
           'n_full_qualifying','absolute_error_pp']],'Figure 1 data.csv')
noian=training[~training.event_id.eq('hurricane_ian_2022')]
s=[dict(comparison=k,**scores(t)) for k,t in [
    ('primary',full),('first 48h',peakjoin(handoff(hi=48))),('stable seven reports',peakjoin(handoff(stable=True))),
    ('omit Sep 29',peakjoin(handoff(omit=pd.Timestamp('2024-09-29T13:00Z')))),
    ('no Ian',peakjoin(handoff(),ht[[F]].assign(forecast_peak_pct=predict_peak(noian,ht,W))))]]
terrain=read('terrain_county_derived.csv')
def add_terrain(a):
    z=a.merge(terrain,on=F,validate='many_to_one').dropna(subset=TERRA).copy()
    return z.assign(slope_x_ero=z[TERRA[0]]*z[W[1]])
at=add_terrain(training);bt=add_terrain(ht)
for tag,cols in [('matched terrain weather only',W),('matched terrain plus slope',W+TERRA+['slope_x_ero'])]:
    zz=ht[[F]].merge(bt[[F]].assign(forecast_peak_pct=predict_peak(at,bt,cols)),on=F)
    s.append(dict(comparison=tag,train_n=len(at),**scores(peakjoin(handoff(),zz))))
sparse=read('helene_sparse_peak_labels.csv')
sparse['observed_peak_pct']=PCT(sparse)
sparse=peakjoin(sparse).merge(full[[F,'observed_peak_pct']],on=F,suffixes=('','_full'),validate='one_to_one')
s.append(dict(comparison='original sparse test labels',**scores(sparse),
              changed_peaks=int((sparse.observed_peak_pct_full>sparse.observed_peak_pct+1e-9).sum()),
              largest_peak_change_pp=(sparse.observed_peak_pct_full-sparse.observed_peak_pct).max()))
save(pd.DataFrame(s),'peak_metrics.csv')
rows=[]
tm=read('timing_matched_features.csv')
for arm,z in tm.groupby('arm'):
    at=z[z.year.lt(2024)]
    bt=z[z.event_id.eq(H)]
    assert (len(at),at.event_id.nunique(),len(bt))==(490,8,21)
    for tag,cols in [('original two features',W),('add moderate/high rain coverage',W+['moderate_rain_area_share'])]:
        q=peakjoin(handoff(lo=12),bt[[F]].assign(forecast_peak_pct=predict_peak(at,bt,cols)))
        rows.append(dict(arm=arm,model=tag,train_n=len(at),**scores(q)))
save(pd.DataFrame(rows),'timing_metrics.csv')
trigger=read('nc_trigger_frozen_predictions.csv')
trigger_metrics=[]
for (scope,arm),z in trigger.groupby(['scope','arm']):
    q=peakjoin(handoff(lo=12),z[[F,'score_pct']].rename(columns={'score_pct':'forecast_peak_pct'}))
    trigger_metrics.append(dict(comparison=scope,arm=arm,**scores(q)))
save(pd.DataFrame(trigger_metrics),'trigger_metrics.csv')

C=['cell_sites_out_due_to_'+s for s in ['power','transport','damage']]
B='cell_sites_on_backup_power'
r=read('recovery_reconciled_reports.csv')
r[TS]=time(r[TS]);r['landfall_utc']=time(r.landfall_utc)
r['pct']=PCT(r)
r=r.sort_values(K+[TS])
assert not r.duplicated(K+[TS]).any()
p=[]
for (ev,fips),g in r.groupby(K):
    land=g.landfall_utc.iloc[0]
    for target in g[g[TS].between(land,land+pd.Timedelta(days=25))].itertuples(index=False):
        timestamp=getattr(target,TS); issue=timestamp.normalize()
        if issue<land or not 8<=(timestamp-issue).total_seconds()/3600<=22: continue
        for lag in [0,6,12,24]:
            pri=g[g[TS].between(issue-pd.Timedelta(hours=72),issue-pd.Timedelta(hours=lag))]
            if pri.empty: continue
            prior=pri.iloc[-1]
            elapsed=(prior[TS]-pri.iloc[-2][TS]).total_seconds()/3600 if len(pri)>1 else np.nan
            slope=(prior.pct-pri.iloc[-2].pct)*24/elapsed if 6<=elapsed<=60 else np.nan
            v=dict(event_id=ev,county_fips=fips,issue_utc=issue,target_utc=timestamp,prior_utc=prior[TS],availability_lag_h=lag,
                   day_after=(issue-land).days,prior_age_h=(issue-prior[TS]).total_seconds()/3600,
                   target_pct=target.pct,prior_pct=prior.pct,prior_slope_pp_day=slope,
                   same_denominator=prior.cell_sites_served==target.cell_sites_served,
                   cause_ok=prior[C].le(prior.cell_sites_out).all(),
                   backup_ok=prior[B]<=prior.cell_sites_served-prior.cell_sites_out)
            for k,n in zip(C+[B],['power','transport','damage','backup']):
                v['prior_pct_'+n]=100*prior[k]/prior.cell_sites_served
            p.append(v)
P=pd.DataFrame(p)
base=['prior_pct','prior_slope_pp_day','prior_age_h']
causes=['prior_pct_'+c for c in ['power','transport','damage']]
back=['prior_pct_backup']
def fit_recovery(a,b,models):
    y=np.log1p(a.target_pct)-np.log1p(a.prior_pct)
    res={'persistence':b.prior_pct.to_numpy()}
    for k,v in {'learned_fcc':[],**models}.items():
        q=ridge(a,b,base+v,y,['prior_pct','prior_age_h']+v,True)
        res[k]=np.clip(np.expm1(np.log1p(b.prior_pct)+q),0,100)
    return res
def eval_recovery(frame,holdout,extra):
    a=frame[~frame.event_id.eq(holdout)];b=frame[frame.event_id.eq(holdout)]
    z=b[[F,'issue_utc','target_utc','target_pct']].copy()
    for k,v in fit_recovery(a,b,extra).items(): z[k]=v
    return a,z
def summarize(a,b):
    return [{'model':k,'train_n':len(a),'train_storms':a.event_id.nunique(),
        'test_n':len(b),'mae_pp':abs(b[k]-b.target_pct).mean()}
        for k in b.columns if k not in [F,'issue_utc','target_utc','target_pct']]
extra={'power_only':causes[:1],'transport_only':causes[1:2],'damage_only':causes[2:],'causes':causes}
main=P[P.availability_lag_h.eq(6)&P.day_after.lt(10)&P.cause_ok]
a,z=eval_recovery(main,H,extra)
assert (len(a),a.event_id.nunique(),len(z),z[F].nunique())==(1203,8,210,21)
rec=summarize(a,z)
labels=['FCC carry-forward','Learned FCC','+ Power-related outages','+ Telecom transport outages','+ Damage-related outages','+ All reported causes']
f2=pd.DataFrame(rec).rename(columns={'test_n':'county_target_pairs'})
f2['display_label']=labels
save(f2[['model','county_target_pairs','mae_pp','display_label']],'Figure 2 error data.csv')
zz=z.groupby('target_utc',as_index=False).agg(counties=(F,'nunique'),observed_pct=('target_pct','mean'),persistence_pct=('persistence','mean'))
zz['date']=zz.target_utc.dt.strftime('%b %d')
save(zz,'Figure 2 trajectory data.csv')
table=[dict(task='peak',comparison=r['comparison'],n=r['n'],mae_pp=r['mae_pp']) for r in s[:3]]
table += [dict(task='next_report',comparison=r['model'],n=r['test_n'],mae_pp=r['mae_pp']) for r in rec if r['model'] in ['persistence','learned_fcc','transport_only','causes']]
save(pd.DataFrame(table),'Table 2.csv')
rrows=[]
stable={}
for lag in [0,6,12,24]:
    q=P[P.availability_lag_h.eq(lag)&P.day_after.lt(10)]
    for tag,filt,models in [('all causes',q.cause_ok,extra),('backup',q.backup_ok,{'backup':back}),
        ('causes and backup',q.cause_ok&q.backup_ok,{'causes':causes,'backup':back,'causes_backup':causes+back})]:
        a,b=eval_recovery(q[filt],H,models)
        rrows.extend(dict(scope=tag,lag_h=lag,**row) for row in summarize(a,b))
    q=P[P.availability_lag_h.eq(lag)&P.day_after.between(3,9)&P.same_denominator&P.cause_ok]
    a,b=eval_recovery(q,H,{'transport_only':causes[1:2]})
    b=b.set_index([F,'target_utc'],drop=False)
    stable[lag]=(a,b)
    rrows.extend(dict(scope='days 3–9 prior and target totals agree',lag_h=lag,**row) for row in summarize(a,b))
shared=stable[6][1].index.intersection(stable[24][1].index)
assert len(shared)==140
for lag in [6,24]:
    a,b=stable[lag]
    rrows.extend(dict(scope='days 3–9 shared 6h/24h targets',lag_h=lag,**row) for row in summarize(a,b.loc[shared]))
save(pd.DataFrame(rrows),'recovery_sensitivity.csv')
earlier=main[~main.event_id.eq(H)]
loso=[]
for e in earlier.event_id.unique():
    a,b=eval_recovery(earlier,e,extra)
    loso.extend(dict(storm=e,**row) for row in summarize(a,b))
save(pd.DataFrame(loso).groupby('model',as_index=False).agg(storms=('storm','size'),storm_weighted_mae_pp=('mae_pp','mean')),'earlier_storm_transfer.csv')

def verify(name,keys,columns,expected=None):
    actual=pd.read_csv(OUT/name,dtype={F:str})
    reference=pd.read_csv(HERE/'expected'/(expected or name),dtype={F:str})
    x=actual.merge(reference,on=keys,validate='one_to_one',suffixes=('_actual','_reference'))
    assert len(x)==len(actual)==len(reference),(name,len(x),len(actual),len(reference))
    for k in columns:
        np.testing.assert_allclose(x[k+'_actual'],x[k+'_reference'],atol=1e-8,rtol=1e-8,err_msg=name+' '+k)
verify('Figure 1 data.csv',[F],['observed_peak_pct','forecast_peak_pct','n_full_qualifying','absolute_error_pp'])
verify('Figure 2 error data.csv',['model'],['county_target_pairs','mae_pp'])
verify('Figure 2 trajectory data.csv',['target_utc'],['counties','observed_pct','persistence_pct'])
ref=pd.read_csv(HERE/'expected/full_helene_outcome_sensitivity.csv')
mainref=ref[(ref.comparison=='all_earlier')&(ref.model=='wind rain')].iloc[0]
np.testing.assert_allclose([s[0]['mae_pp'],s[0]['rho'],s[0]['top5_overlap']],[mainref.mae_full,mainref.rho_full,mainref.top5_full],atol=1e-8,rtol=1e-8)
for row in rows:
    r=ref[(ref.comparison=='all_arms_matched')&(ref.arm==row['arm'])&(ref.model==row['model'])].iloc[0]
    np.testing.assert_allclose([row['mae_pp'],row['rho'],row['top5_overlap']],[r.mae_full,r.rho_full,r.top5_full],atol=1e-8,rtol=1e-8)
for row in trigger_metrics:
    r=ref[(ref.comparison==row['comparison'])&(ref.arm==row['arm'])&(ref.experiment=='forecast NC-approach trigger')].iloc[0]
    np.testing.assert_allclose([row['mae_pp'],row['rho'],row['top5_overlap']],[r.mae_full,r.rho_full,r.top5_full],atol=1e-8,rtol=1e-8)
print('Frozen Figure 1–2 numerical verification and matched timing checks passed.')
print(pd.DataFrame(table).to_string(index=False))

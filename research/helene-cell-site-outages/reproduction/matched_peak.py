"""Rebuild one-report timing checks from the source-linked eligible report ledger."""
from pathlib import Path
import argparse
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
D=Path(__file__).resolve().parent
p=argparse.ArgumentParser()
p.add_argument('--out',type=Path,default=D/'reproduced/matched_peak')
O=p.parse_args().out
O.mkdir(parents=True,exist_ok=True)
K=['event_id','county_fips'];F=K[1];T='report_timestamp_utc';TARGET='target_pct'
H='hurricane_helene_2024'
W=['wind_wsp34_county_mean_p_lower','rain_risk_rank']
read=lambda p:pd.read_csv(p,dtype={F:str})
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

def fit(a,b,cols,mode):
    pr=np.clip(np.expm1(ridge(a,b,cols,np.log1p(a[TARGET]))),0,100)
    return pr.reshape(-1,1)
train=read(D/'inputs/peak_training_with_coverage.csv')
allobs=read(D/'inputs/peak_eligible_report_inventory.csv')
allobs[T]=pd.to_datetime(allobs[T],format='mixed',utc=True)
train[T]=pd.to_datetime(train[T],format='mixed',utc=True)
assert len(train)==589 and train.event_id.nunique()==11
assert not allobs.duplicated(K+[T]).any()
assert (np.abs(allobs[TARGET]-100*allobs.cell_sites_out/allobs.cell_sites_served)<1e-8).all()
peak=lambda g:g.sort_values([TARGET,T],ascending=[False,True]).drop_duplicates(K)
test=peak(allobs[allobs.event_id.eq(H)])[K+[TARGET,T]].merge(read(D/'inputs/peak_helene_weather.csv')[K+W],on=K,validate='one_to_one')
assert len(test)==21
rows=[]; preds=[]; selectors=[]
def record(rule,arm,tr,te,pred,label,snapshot_hours=None):
    y=te[TARGET].to_numpy(); d=pred-y
    rho=spearmanr(y,pred).statistic if np.unique(y).size>1 and np.unique(pred).size>1 else np.nan
    top5=np.nan if np.unique(pred).size==1 else len(set(np.argsort(-pred,kind='stable')[:5])&set(np.argsort(-y,kind='stable')[:5]))
    rows.append(dict(rule=rule,fit=arm,evaluation_label=label,train_n=len(tr),train_storms=tr.event_id.nunique(),train_storm_ids='; '.join(sorted(tr.event_id.unique())),test_n=len(te),helene_report_hour=snapshot_hours,observed_median=np.median(y),prediction_median=np.median(pred),bias=d.mean(),mae=np.abs(d).mean(),underestimated=int((d<0).sum()),rho=rho,top5_overlap=top5))
    preds.append(te[K+[TARGET]].rename(columns={TARGET:'observed_pct'}).assign(rule=rule,fit=arm,evaluation_label=label,predicted_pct=pred))

full=fit(train,test,W,'ridge')[:,0]
record('original','full frozen peak ridge',train,test,full,'full week reported maximum')
w=train.event_id.map(1/train.event_id.value_counts())
order=np.argsort(train[TARGET].to_numpy(),kind='stable')
weighted_median=train[TARGET].iloc[order[np.searchsorted(w.iloc[order].cumsum().to_numpy(), w.sum()/2)]]
for name,val in {'pooled training median':train[TARGET].median(),'storm-balanced training median':weighted_median,'pooled training mean':train[TARGET].mean(),'storm-balanced training mean':np.average(train[TARGET],weights=w)}.items():
    record('original',name,train,test,np.repeat(val,len(test)),'full week reported maximum')
def readjust(z,obs):
    return z.drop(columns=[TARGET,T],errors='ignore').merge(obs[K+[TARGET,T,'hours_after_landfall']],on=K,validate='one_to_one')
# Fixed, publication-independent UTC lead rules, chosen without examining outage
# values: primary day-two +36h ±12h, alternate day-one +12h ±6h.
for center,tol,use in [(36,12,'primary'),(12,6,'alternate')]:
    rule=f'{use}: +{center}h ±{tol}h'
    z=allobs[allobs.hours_after_landfall.between(center-tol,center+tol)].copy()
    z['abs_lag_offset']=abs(z.hours_after_landfall-center)
    nearest=z.sort_values(['abs_lag_offset',T]).drop_duplicates(K)
    window_peak=peak(z)
    act=readjust(test,nearest)
    assert len(act)==len(test)
    act_peak=readjust(test,window_peak)
    snapshot_hour='; '.join(f'{h:.6f}' for h in sorted(act.hours_after_landfall.unique()))
    tr_snap=readjust(train,nearest)
    tr_win=readjust(train,window_peak)
    tr_peak=train.merge(tr_snap[K],on=K,validate='one_to_one')
    assert len(tr_snap)==len(tr_win)==len(tr_peak)
    keys0=list(zip(test.event_id,test[F]))
    assert list(zip(act.event_id,act[F]))==keys0==list(zip(act_peak.event_id,act_peak[F]))
    # Record training support separately even if insufficient for a comparison.
    selectors.append(nearest.assign(rule=rule,selected='nearest'))
    selectors.append(window_peak.assign(rule=rule,selected='within-window maximum'))
    if tr_snap.event_id.nunique()<2:
        print(rule,'not fit:',len(tr_snap),'eligible rows from',tr_snap.event_id.nunique(),'storms')
        continue
    studies=[('full frozen peak ridge',train,full),('matched-cohort full week peak ridge',tr_peak,fit(tr_peak,test,W,'ridge')[:,0]),('matched-cohort window peak ridge',tr_win,fit(tr_win,test,W,'ridge')[:,0]),('matched-cohort nearest-snapshot ridge',tr_snap,fit(tr_snap,test,W,'ridge')[:,0])]
    for arm,tr,pr in studies:
        record(rule,arm,tr,test,pr,'full week reported maximum')
        record(rule,arm,tr,act_peak,pr,'matched window reported maximum',snapshot_hour)
        record(rule,arm,tr,act,pr,'matched nearest snapshot',snapshot_hour)
pd.DataFrame(rows).to_csv(O/'peak_coverage_and_matched_snapshot_results.csv',index=False)
pd.concat(preds).to_csv(O/'peak_coverage_and_matched_snapshot_predictions.csv',index=False)
pd.concat(selectors).to_csv(O/'peak_matched_selected_snapshots.csv',index=False)

expected=read(D/'expected/matched_peak_metrics.csv').sort_values(['rule','fit','evaluation_label'])
actual=pd.DataFrame(rows).sort_values(['rule','fit','evaluation_label'])
assert len(actual)==len(expected)==29
for k in ['mae','bias','observed_median','prediction_median','train_n','test_n']:
    assert np.allclose(actual[k],expected[k],equal_nan=True),k
ref=read(D/'expected/matched_peak_predictions.csv')
idx=['rule','fit','evaluation_label','county_fips']
diff=ref.merge(pd.concat(preds),on=idx,validate='one_to_one',suffixes=('_reference','_portable'))
assert len(diff)==29*21
assert np.allclose(diff.predicted_pct_reference,diff.predicted_pct_portable,atol=1e-9,rtol=0)
assert np.allclose(diff.observed_pct_reference,diff.observed_pct_portable,atol=1e-9,rtol=0)
# File serialization and numerical backends can break exact ties at the 1e-14 scale.
# Verify the ordering with effectively equal observations receiving average ranks.
for _,g in diff.groupby(idx[:3]):
    u=round(g.observed_pct_reference,9)
    x=round(g.predicted_pct_reference,9); y=round(g.predicted_pct_portable,9)
    if x.nunique()>1:
        assert np.isclose(spearmanr(u,x).statistic,spearmanr(u,y).statistic)
print('Verified 29 matched peak comparisons and all county predictions. Near-tie ranks verified to nine decimal places.')

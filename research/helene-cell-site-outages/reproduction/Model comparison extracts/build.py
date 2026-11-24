"""Freeze exact score artifacts and reconcile display-level values for the supplement."""
from pathlib import Path
import hashlib,json
import pandas as pd,numpy as np
D=Path(__file__).resolve().parent; N=D.parents[2]; T=N/'Decision Model Suite/ML Comparison'
files={
    'peak nested scores.csv':T/'Peak/Tuning/event_scores.csv',
    'peak nested storm means.csv':T/'Peak/Tuning/macro_scores.csv',
    'peak nested time audit.csv':T/'Peak/Tuning/nested_fold_audit.csv',
    'recovery nested scores.csv':T/'Recovery/Tuning/helene_scores.csv',
    'recovery nested storm means.csv':T/'Recovery/Tuning/historical_macro.csv',
    'recovery nested time audit.csv':T/'Recovery/Tuning/inner_chronology.csv',
    'statewide coordination replay.csv':N/'Extreme Decision Model/helene_nc_decision_replay.csv',
    'retrospective closure scores.csv':N/'Decision Model Suite/Mechanisms/blocked_scores.csv',
}
def load():
    freeze=[]
    for name, src in files.items():
        blob=src.read_bytes();(D/name).write_bytes(blob)
        freeze.append(dict(file=name,source=str(src.relative_to(N)),sha256=hashlib.sha256(blob).hexdigest()))
    (D/'manifest.json').write_text(json.dumps(freeze,indent=2)+'\n')
def verify():
    for rec in json.loads((D/'manifest.json').read_text()):
        assert hashlib.sha256((D/rec['file']).read_bytes()).hexdigest()==rec['sha256']
    p=pd.read_csv(D/'peak nested scores.csv')
    h=p[(p.event_id=='hurricane_helene_2024')&(p.panel=='matched quantitative weather')&(p.features=='added rain and wetness')&(p.task=='magnitude')]
    vals=h.pivot(index='model',columns='mode',values='mae_pp')
    assert h.n.eq(21).all()
    assert np.allclose(vals.loc['XGBoost'],[62.93,64.01],atol=.005)
    assert np.allclose(vals.loc['Extra Trees'],[63.36,62.53],atol=.005)
    z=pd.read_csv(D/'recovery nested scores.csv')
    z=z.query('scope=="all paired"')
    assert z.n.eq(210).all()
    a=z.query('task=="share"')
    forest=a[a.family.str.contains('half carry half random forest',case=False)].query('lead_days==1 and variant=="tuned"')
    boost=a[a.family.str.contains('half carry half XGBoost',case=False)].query('lead_days==2 and variant=="tuned"')
    assert np.allclose(forest.mae,[12.72],atol=.005) and np.allclose(boost.mae,[14.67],atol=.005)
    q=pd.read_csv(D/'statewide coordination replay.csv')
    assert len(q)==100 and q.ero_review.sum()==28 and (q.ero_review&q.target_pct.ge(25)).sum()==20
    assert (q.ero_review&q.target_pct.isna()).sum()==7
    print('Score and scope checks passed.')
if __name__=='__main__':
    import sys
    if '--freeze' in sys.argv: load()
    verify()

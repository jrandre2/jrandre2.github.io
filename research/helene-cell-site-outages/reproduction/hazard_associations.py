"""Regenerate Figure 3 correlations from frozen county summaries, not raw geography."""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata

D = Path(__file__).resolve().parent
d = pd.read_csv(D / 'inputs/hazard_county_derived.csv', dtype={'county_fips':str})
assert d.county_fips.is_unique and len(d) == 21
assert d.county_fips.str.startswith('37').all()
np.testing.assert_allclose(d.early_peak_pct, d[['sep28','sep30','oct01']].max(axis=1))
np.testing.assert_allclose(d.initial, d.oct01)
expected = pd.read_csv(D / 'expected/Figure 3 data.csv')
rows = []
for panel, y, x, controls in [
    ('a','early_peak_pct','flood_area','slide_road'),
    ('a','early_peak_pct','flood_bsl','slide_road'),
    ('a','early_peak_pct','flood_bsl','slide_road;hrrr_mean_mm'),
    ('b','day09','slide_road','initial;flood_bsl'),
    ('b','day09','slide_road','initial;flood_bsl;east;north'),
]:
    xy = d[[y,x,*controls.split(';')]].dropna()
    assert len(xy) == 21
    ranks = rankdata(xy.to_numpy(), axis=0)
    z = np.column_stack([np.ones(len(ranks)),ranks[:,2:]])
    r = ranks[:,:2] - z @ np.linalg.lstsq(z,ranks[:,:2],rcond=None)[0]
    rows.append([panel,y,x,controls,len(xy),np.corrcoef(r.T)[0,1]])
out = pd.DataFrame(rows,columns=expected.columns)
pd.testing.assert_frame_equal(out,expected,atol=1e-12,rtol=0)
path = D/'reproduced/Figure 3 data.csv'
path.parent.mkdir(exist_ok=True)
out.to_csv(path,index=False)
print('Reproduced all five Figure 3 values from 21 county records.')

from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

def read_bin(provider, instrument, field, calendar):
    path = provider / 'features' / instrument.lower() / f'{field}.day.bin'
    out = pd.Series(np.nan, index=calendar)
    if path.exists():
        a = np.fromfile(path, dtype='<f4')
        if len(a):
            start = int(a[0]); n = min(len(a)-1, len(calendar)-start)
            if start < 0: raise ValueError('Negative binary offset')
            out.iloc[start:start+n] = a[1:n+1]
    return out

def frozen_panel(provider=None):
    """Read original bytes without changing the provider or adjustment convention."""
    provider = Path(provider or ROOT/'data/frozen_cn_data')
    calendar = pd.DatetimeIndex(pd.read_csv(provider/'calendars/day.txt', header=None)[0])
    intervals = pd.read_csv(provider/'instruments/csi300.txt', sep='\t', header=None,
                            names=['instrument','start','end'])
    rows=[]
    for code, spans in intervals.groupby('instrument'):
        f = pd.DataFrame({k:read_bin(provider,code,k,calendar) for k in
                          ['open','high','low','close','volume','amount']})
        f['member']=False
        for span in spans.itertuples(): f.loc[span.start:span.end,'member']=True
        f['instrument']=code.upper(); f.index.name='date'; rows.append(f.reset_index())
    panel=pd.concat(rows,ignore_index=True).set_index(['date','instrument']).sort_index()
    return panel, calendar

def load_wind(path, calendar):
    folder=Path(path)
    receipt=json.loads((folder/'field_contract.json').read_text(encoding='utf-8'))
    if not receipt.get('reviewed') or receipt.get('price_adjustment')!='none':
        raise ValueError('Review Wind field_contract.json and raw-price convention first')
    if not receipt.get('field_config_sha256'):raise ValueError('Missing field-map provenance; rerun current downloader and review contract')
    parts=[pd.read_csv(p) for p in folder.glob('*.csv') if p.name!='audit.csv']
    if not parts: raise ValueError('No Wind CSV files')
    df=pd.concat(parts,ignore_index=True); df['date']=pd.to_datetime(df['date'])
    df['instrument']=df['instrument'].str.upper()
    if df.duplicated(['date','instrument']).any(): raise ValueError('Duplicate Wind keys')
    df=df.set_index(['date','instrument']).sort_index()
    counts=df.get('trade_count',pd.Series(dtype=float)).dropna()
    if len(counts) and ((counts<0).any() or not np.allclose(counts,counts.round())):
        raise ValueError('trade_count must contain genuine nonnegative integer counts')
    return df,receipt

def audit(panel, out):
    report={'rows':len(panel),'dates':int(panel.index.get_level_values('date').nunique()),
            'securities':int(panel.index.get_level_values('instrument').nunique()),
            'missing_fraction':panel.isna().mean().to_dict(),
            'trade_count_available':bool('trade_count' in panel and panel.trade_count.notna().any()),
            'small_active_flow_available':all(c in panel for c in ['small_buy','small_sell']),
            'source':'original frozen community Qlib snapshot; research-adjusted prices'}
    Path(out).write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    return report

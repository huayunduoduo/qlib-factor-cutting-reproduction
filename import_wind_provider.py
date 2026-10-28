"""Build a separate Qlib provider only from reviewed raw Wind fields and PIT membership."""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from research.data import ROOT,load_wind

def main():
    p=argparse.ArgumentParser();p.add_argument('--raw',default='data/wind_raw')
    p.add_argument('--calendar',required=True,help='Official exported trading calendar: one date per line')
    p.add_argument('--universe',required=True,help='CSV: instrument,start,end; actual historical CSI300 intervals')
    p.add_argument('--out',default='data/wind_provider');a=p.parse_args()
    raw=ROOT/a.raw;contract=json.loads((raw/'field_contract.json').read_text())
    for flag in ['reviewed','price_fields_verified','adjustment_factor_verified','historical_universe_verified']:
        if not contract.get(flag):raise ValueError('Review contract flag: '+flag)
    cal=pd.DatetimeIndex(pd.to_datetime(pd.read_csv(a.calendar,header=None)[0])).sort_values()
    if cal.has_duplicates:raise ValueError('Duplicate calendar')
    f,_=load_wind(raw,cal);univ=pd.read_csv(a.universe,parse_dates=['start','end'])
    if (univ.end<univ.start).any():raise ValueError('Invalid membership interval')
    required={'open','high','low','close','volume','amount','adj_factor'}
    if not required.issubset(f):raise ValueError('Missing reviewed price/adjustment fields: '+str(required-set(f)))
    universe_codes=set(univ.instrument.str.upper())
    missing=universe_codes-set(f.index.get_level_values('instrument'))
    if missing:raise ValueError('Missing historical securities: '+str(sorted(missing)))
    out=ROOT/a.out
    if out.exists() and any(out.iterdir()):raise ValueError('Choose a new empty provider directory to preserve prior versions')
    (out/'calendars').mkdir(parents=True,exist_ok=True);(out/'instruments').mkdir(exist_ok=True)
    pd.Series(cal.strftime('%Y-%m-%d')).to_csv(out/'calendars/day.txt',index=False,header=False)
    univ.instrument=univ.instrument.str.upper()
    univ[['instrument','start','end']].to_csv(out/'instruments/csi300.txt',sep='\t',index=False,header=False,date_format='%Y-%m-%d')
    allrows=[]
    for code,g in f.groupby(level='instrument'):
        g=g.droplevel('instrument').reindex(cal)
        factor=g.adj_factor;valid=(g.close>0)&(factor>0)
        if not valid.any():raise ValueError('No valid adjusted prices: '+code)
        anchor=(g.close*factor).loc[valid].iloc[0];pf=factor/anchor
        values={c:g[c]*pf for c in ['open','high','low','close']}
        rawvol=g.volume*float(contract['volume_to_shares']);rawamt=g.amount*float(contract['amount_to_yuan'])
        values.update(volume=rawvol/pf,amount=rawamt,vwap=rawamt/rawvol.where(rawvol>0)*pf,factor=pf)
        target=out/'features'/code.lower();target.mkdir(parents=True,exist_ok=True)
        for name,s in values.items():
            np.r_[np.float32(0),s.to_numpy(dtype='float32')].astype('<f4').tofile(target/f'{name}.day.bin')
        allrows.append([code,cal[valid][0].date(),cal[valid][-1].date()])
    pd.DataFrame(allrows).to_csv(out/'instruments/all.txt',sep='\t',header=False,index=False)
    (out/'wind_provenance.json').write_text(json.dumps({'contract':contract,'source':str(raw),'universe':str(a.universe),
       'calendar':str(a.calendar),'adjustment':'raw price * adj_factor / first valid adjusted close; volume inverse factor',
       'limitations':'Requires separately verified dividend convention, delisting treatment and exchange-specific trading status.'},ensure_ascii=False,indent=2),encoding='utf-8')
    print('Created separate provider:',out)

if __name__=='__main__':main()

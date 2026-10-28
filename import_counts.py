"""Normalize supplied Wind counts without inventing missing values."""
import argparse,json,hashlib
from pathlib import Path
import pandas as pd

def main():
 p=argparse.ArgumentParser();p.add_argument('source');p.add_argument('--out',default='wind_normalized');a=p.parse_args()
 out=Path(a.out);out.mkdir(exist_ok=True,parents=True);parts=[];sources=[]
 for f in sorted(Path(a.source).iterdir()):
  if '笔数' not in f.name and '成交笔数' not in f.name:continue
  if f.suffix=='.xlsx':
   raw=pd.read_excel(f,header=None);codes=raw.iloc[1,1:].astype(str).str.extract(r'(\d{6}\.(?:SH|SZ))',expand=False);body=raw.iloc[2:].copy()
  elif f.suffix=='.csv' and '清单' not in f.name:
   raw=pd.read_csv(f,header=None,encoding='gb18030',low_memory=False);codes=raw.iloc[4,1:].astype(str);body=raw.iloc[5:].copy()
  else:continue
  dates=pd.to_datetime(body.iloc[:,0],errors='coerce');x=body.iloc[:,1:].apply(pd.to_numeric,errors='coerce');x.columns=codes.to_numpy();x.index=dates
  x=x.loc[x.index.notna(),x.columns.str.match(r'^\d{6}\.(SH|SZ)$',na=False)]
  x.columns=[s.split('.')[1]+s.split('.')[0] for s in x.columns]
  z=x.rename_axis(index='date',columns='instrument').stack().rename('trade_count').reset_index();z=z[(z.date>='2009-01-01')&(z.date<='2025-12-31')];z['source']=f.name;parts.append(z)
  sources.append({'file':f.name,'sha256':hashlib.sha256(f.read_bytes()).hexdigest(),'observations':len(z)})
 allrows=pd.concat(parts,ignore_index=True);keys=['date','instrument']
 if (allrows.trade_count<0).any() or (allrows.trade_count%1!=0).any():raise ValueError('Invalid counts')
 nunique=allrows.groupby(keys).trade_count.nunique();bad=nunique[nunique>1]
 allrows.merge(bad.rename('nunique'),on=keys).to_csv(out/'count_conflicts.csv',index=False)
 clean=allrows.drop_duplicates(keys).set_index(keys);clean=clean.loc[~clean.index.isin(bad.index)].reset_index();clean.to_csv(out/'trade_count.csv.gz',index=False)
 stats=clean.groupby(clean.date.dt.year).agg(observations=('trade_count','size'),securities=('instrument','nunique'),positive=('trade_count',lambda s:int((s>0).sum())),zero=('trade_count',lambda s:int((s==0).sum())))
 stats.to_csv(out/'count_coverage_by_year.csv')
 (out/'audit.json').write_text(json.dumps({'sources':sources,'conflicting_keys':len(bad),'retained':len(clean),'missing_policy':'NA preserved, conflicting keys excluded; zero is not a positive count','factor_status':'No verified raw total amount supplied: ideal reversal pending. All-order active buy is not small-order buy/sell.'},ensure_ascii=False,indent=2))
 print(stats.to_string());print('conflicts',len(bad))
if __name__=='__main__':main()

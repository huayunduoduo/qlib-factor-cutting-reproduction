"""Reprice saved scores with prior-day fixed candidates and failed-buy cash."""
import json,hashlib,argparse
import pandas as pd
from research.data import ROOT,frozen_panel,read_bin
from research.evaluation import portfolio,performance
from evaluate_replication import drawdown_dates

def main(folders):
    provider=ROOT/'data/research_2009_2025';panel,cal=frozen_panel(provider)
    bench=read_bin(provider,'SH000300','close',cal).pct_change(fill_method=None)
    for folder in folders:
        out=ROOT/folder;s=pd.read_csv(out/'summary.csv');key='factor' if 'factor' in s else 'experiment'
        annual=[];costs=[]
        for i,row in s.iterrows():
            name=row[key];p=out/f'{name}_predictions.csv.gz'
            pred=pd.read_csv(p,parse_dates=['date']).set_index(['date','instrument']) if p.exists() else pd.read_parquet(out/f'{name}_predictions.parquet')
            dates=cal[(cal>=pred.index.get_level_values('date').min())&(cal<='2025-12-31')]
            for slip in [0.,.001,.002]:
                bt,h=portfolio(panel,pred,dates,slippage=slip);bt['benchmark_return']=bench.reindex(bt.index)
                perf=performance(bt);costs.append({key:name,'slippage_per_side':slip,**perf})
                if slip==0:
                    bt.to_csv(out/f'{name}_portfolio.csv');h.to_csv(out/f'{name}_holdings.csv',index=False)
                    for k,v in {**perf,**drawdown_dates(bt)}.items():s.loc[i,k]=v
                    s.loc[i,'days_with_stale_positions']=int((bt.stale_holdings>0).sum())
                    for year,g in bt.groupby(bt.index.year):
                        z=g.copy();z.nav=(1+z.net_return).cumprod();annual.append({key:name,'year':year,**performance(z),**drawdown_dates(z)})
            print('REPRICED',folder,name,flush=True)
        s.to_csv(out/'summary.csv',index=False);pd.DataFrame(annual).to_csv(out/'yearly_portfolio.csv',index=False);pd.DataFrame(costs).to_csv(out/'cost_sensitivity.csv',index=False)
        receipt={'execution':'previous-day top50 fixed before observing fill-day tradability; rejected buys remain cash, locked positions remain held','evaluation_source_sha256':hashlib.sha256((ROOT/'research/evaluation.py').read_bytes()).hexdigest(),'prediction_files_unchanged':True}
        (out/'execution_revision.json').write_text(json.dumps(receipt,indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('folders',nargs='+');a=p.parse_args();main(a.folders)

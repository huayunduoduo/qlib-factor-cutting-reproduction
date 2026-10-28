"""Add actual annual returns and drawdown dates without rerunning models."""
import pandas as pd
from research.data import ROOT
from evaluate_replication import drawdown_dates
for folder in ['results_2015_2025','results_with_reversal']:
    out=ROOT/folder
    if not (out/'summary.csv').exists():continue
    summary=pd.read_csv(out/'summary.csv');yearly=pd.read_csv(out/'yearly_portfolio.csv')
    for i,row in summary.iterrows():
        bt=pd.read_csv(out/f'{row.experiment}_portfolio.csv',parse_dates=['date']).set_index('date')
        for k,v in drawdown_dates(bt).items():summary.loc[i,k]=v
        summary.loc[i,'days_with_stale_positions']=int((bt.stale_holdings>0).sum())
        for year,g in bt.groupby(bt.index.year):
            mask=(yearly.experiment==row.experiment)&(yearly.year==year)
            z=g.copy();z.nav=(1+z.net_return).cumprod()
            for k,v in drawdown_dates(z).items():yearly.loc[mask,k]=v
    summary.to_csv(out/'summary.csv',index=False);yearly.to_csv(out/'yearly_portfolio.csv',index=False)
print('Enriched actual yearly returns and drawdown dates')

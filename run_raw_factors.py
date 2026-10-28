import json
import pandas as pd,numpy as np
from research.data import ROOT,frozen_panel,load_wind,read_bin
from research.features import build_features
from research.evaluation import daily_ic,portfolio,performance,paired_bootstrap
from evaluate_replication import drawdown_dates

def main():
    out=ROOT/'results_raw_factors';out.mkdir(exist_ok=True)
    provider=ROOT/'data/research_2009_2025';p,c=frozen_panel(provider)
    w,_=load_wind(ROOT/'verified_wind_join',c);p=p.join(w.trade_count)
    p['avg_trade_amount']=p.amount*1000/p.trade_count.where(p.trade_count>0)
    f=build_features(p,c).loc['2015-01-01':'2025-12-31'];f=f[f.member]
    f.to_parquet(out/'factor_panel.parquet')
    b=read_bin(provider,'SH000300','close',c).pct_change(fill_method=None)
    summary=[];years=[];ic_years=[];quantiles=[];coverage=[]
    dates=c[(c>='2015-01-01')&(c<='2025-12-31')]
    months=set(pd.Series(c,index=c).groupby(c.to_period('M')).max().to_list())
    for factor in ['cut_amplitude','cut_reversal']:
        pred=f[[factor,'label']].rename(columns={factor:'score'}).dropna(subset=['score'])
        # Fixed economic sign, chosen from report, never from this run's OOS return.
        pred.score=-pred.score
        pred.to_parquet(out/f'{factor}_predictions.parquet')
        ic=daily_ic(pred);ic.to_csv(out/f'{factor}_ic.csv')
        mic=ic.loc[ic.index.isin(months)]
        for y,g in ic.groupby(ic.index.year):ic_years.append({'factor':factor,'year':y,'signed_RankIC':g.RankIC.mean(),'original_raw_RankIC':-g.RankIC.mean(),'days':len(g),'mean_n':g.n.mean()})
        bt,h=portfolio(p,pred,dates);bt['benchmark_return']=b.reindex(bt.index)
        bt.to_csv(out/f'{factor}_portfolio.csv');h.to_csv(out/f'{factor}_holdings.csv',index=False)
        summary.append({'factor':factor,'signed_RankIC':ic.RankIC.mean(),'original_raw_RankIC':-ic.RankIC.mean(),'monthly_signed_RankIC':mic.RankIC.mean(),**performance(bt),**drawdown_dates(bt)})
        for y,g in bt.groupby(bt.index.year):
            z=g.copy();z.nav=(1+z.net_return).cumprod();years.append({'factor':factor,'year':y,**performance(z),**drawdown_dates(z)})
        for date,g in pred.dropna().groupby(level='date'):
            if date not in months or len(g)<20:continue
            bins=pd.qcut(g.score.rank(method='first'),10,labels=False)
            for q,h in g.groupby(bins):quantiles.append({'factor':factor,'date':date,'signed_decile':int(q)+1,'mean_forward_return':h.label.mean(),'n':len(h)})
    pd.DataFrame(summary).to_csv(out/'summary.csv',index=False)
    pd.DataFrame(years).to_csv(out/'yearly_portfolio.csv',index=False)
    pd.DataFrame(ic_years).to_csv(out/'yearly_ic.csv',index=False)
    pd.DataFrame(quantiles).to_csv(out/'monthly_decile_forward_returns.csv',index=False)
    (out/'metadata.json').write_text(json.dumps({'sign':'minus original raw factor, fixed from paper','portfolio':'monthly Top50, 95% target exposure, buy 5bp sell 15bp','universe':'CSI300 historical constituent intervals','price_limit':'symmetric 9.5% proxy, no exact ST/board rules','neutralization':False,'group_return':'forward 20d labels, not an executable long-short NAV','cutoff':'2025-12-31'},indent=2))
    print(pd.DataFrame(summary).to_string(index=False))
if __name__=='__main__':main()

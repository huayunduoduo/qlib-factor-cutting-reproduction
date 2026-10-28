"""Evaluate all completed annual neural predictions, raw factors, and matched controls.

Daily overlapping 20-day IC is separate from monthly implemented portfolios.
Quantile forward labels are not a tradable long-short NAV.
"""
import json,hashlib
from pathlib import Path
import numpy as np,pandas as pd
from research.data import ROOT,frozen_panel,load_wind,read_bin
from research.features import build_features
from research.evaluation import daily_ic,paired_bootstrap,portfolio,performance
from research.combine import symmetric_combine
from research.preprocess import residualize

def drawdown_dates(bt):
    v=bt.nav;peak=np.maximum.accumulate(np.r_[1.,v.to_numpy()])[1:];dd=v.to_numpy()/peak-1;k=int(np.argmin(dd));before=v.iloc[:k+1]
    if before.max()<1: start='initial capital';pv=1.
    else:start=str(before.idxmax().date());pv=float(before.max())
    recovered=v.iloc[k+1:];recovered=recovered[recovered>=pv]
    return {'drawdown_start':start,'drawdown_trough':str(v.index[k].date()),'drawdown_recovery':str(recovered.index[0].date()) if len(recovered) else 'unrecovered','total_return':float(v.iloc[-1]-1),'first_date':str(v.index.min().date()),'last_date':str(v.index.max().date())}

def main():
    out=ROOT/'results_replication';out.mkdir(exist_ok=True)
    provider=ROOT/'data/research_2009_2025';panel,cal=frozen_panel(provider)
    w,contract=load_wind(ROOT/'verified_wind_join',cal)
    panel=panel.join(w.trade_count);panel['avg_trade_amount']=panel.amount*1000/panel.trade_count.where(panel.trade_count>0)
    f=build_features(panel,cal).loc['2015-01-01':'2025-12-31'];f=f[f.member]
    scores=pd.DataFrame({'raw_amplitude':-f.cut_amplitude,'raw_reversal':-f.cut_reversal,'label':f.label})
    metadata=[]
    for factor in ['amplitude','reversal']:
        for model in ['gru','dbd']:
            parts=[]
            for year in range(2015,2026):
                p=ROOT/'results_neural_grid'/factor/f'{model}_{year}_predictions.parquet'
                if not p.exists():raise RuntimeError(f'Missing full annual run: {p}')
                z=pd.read_parquet(p);assert not z.index.has_duplicates
                assert (z.fit_end<z.index.get_level_values('date')).all()
                parts.append(z.score)
                metadata.append(json.loads(p.with_name(f'{model}_{year}.json').read_text()))
            s=pd.concat(parts);assert not s.index.has_duplicates
            scores[f'{model}_{factor}']=s
    pd.DataFrame(metadata).to_csv(out/'training_runs.csv',index=False)
    scores['dbd_two_factor']=symmetric_combine(scores[['dbd_amplitude','dbd_reversal']])
    # Standalone coverage and common-support analysis are both retained.
    standalone=[]
    for name in scores.columns.drop('label'):
        ic=daily_ic(scores[[name,'label']].rename(columns={name:'score'}))
        ic.to_csv(out/f'{name}_available_ic.csv');standalone.append({'experiment':name,'RankIC':ic.RankIC.mean(),'days':len(ic),'mean_securities':ic.n.mean()})
    pd.DataFrame(standalone).to_csv(out/'available_ic_summary.csv',index=False)
    for factor in ['amplitude','reversal']:
        scores[f'dbd_{factor}_residual_vs_raw_gru']=residualize(scores[f'dbd_{factor}'],scores[[f'raw_{factor}',f'gru_{factor}']])
    # Add baseline only as a matched prediction comparator, not trained stacking.
    b=pd.read_csv(ROOT/'results_2015_2025/E0_baseline_predictions.csv.gz',parse_dates=['date']).set_index(['date','instrument'])
    scores['alpha158_lgb']=b.score
    cols=list(scores.columns.drop('label'))
    common=scores.dropna(subset=cols).copy()
    common.to_parquet(out/'matched_predictions.parquet')
    summary=[];annual=[];icannual=[];ics={};quantiles=[];costs=[]
    bench=read_bin(provider,'SH000300','close',cal).pct_change(fill_method=None)
    months=set(pd.Series(cal,index=cal).groupby(cal.to_period('M')).max().to_list())
    evalcal=cal[(cal>='2015-01-01')&(cal<='2025-12-31')]
    for name in cols:
        pred=common[[name,'label']].rename(columns={name:'score'})
        ic=daily_ic(pred);ics[name]=ic;ic.to_csv(out/f'{name}_ic.csv')
        for yr,g in ic.groupby(ic.index.year):icannual.append({'experiment':name,'year':yr,'RankIC':g.RankIC.mean(),'IC':g.IC.mean(),'days':len(g),'mean_securities':g.n.mean()})
        monthly=ic.loc[ic.index.isin(months)];monthly.to_csv(out/f'{name}_monthly_ic.csv')
        for date,g in pred.dropna().groupby(level='date'):
            if date not in months or len(g)<20:continue
            bins=pd.qcut(g.score.rank(method='first'),10,labels=False)
            for q,h in g.groupby(bins):quantiles.append({'experiment':name,'date':date,'decile_low_to_high':int(q)+1,'mean_forward_20d_return':h.label.mean(),'n':len(h)})
        for slip in [0.,.001,.002]:
            bt,hold=portfolio(panel,pred,evalcal,slippage=slip)
            bt['benchmark_return']=bench.reindex(bt.index)
            perf=performance(bt);costs.append({'experiment':name,'slippage_per_side':slip,**perf})
            if slip==0:
                bt.to_csv(out/f'{name}_portfolio.csv');hold.to_csv(out/f'{name}_holdings.csv',index=False)
                summary.append({'experiment':name,'daily_RankIC':ic.RankIC.mean(),'monthly_RankIC':monthly.RankIC.mean(),'monthly_RankICIR_unannualized':monthly.RankIC.mean()/monthly.RankIC.std(),'IC_end':str(ic.index.max().date()),**perf,**drawdown_dates(bt)})
                for year,g in bt.groupby(bt.index.year):
                    g=g.copy();g['nav']=(1+g.net_return).cumprod()
                    annual.append({'experiment':name,'year':year,'year_return':float(g.nav.iloc[-1]-1),**performance(g),**drawdown_dates(g)})
    pd.DataFrame(summary).to_csv(out/'summary.csv',index=False)
    pd.DataFrame(annual).to_csv(out/'yearly_portfolio.csv',index=False)
    pd.DataFrame(icannual).to_csv(out/'yearly_ic.csv',index=False)
    pd.DataFrame(quantiles).to_csv(out/'monthly_decile_forward_returns.csv',index=False)
    pd.DataFrame(costs).to_csv(out/'cost_sensitivity.csv',index=False)
    tests={}
    for factor in ['amplitude','reversal']:
        for control in ['raw','gru']:
            tests[f'dbd_{factor}_minus_{control}']=paired_bootstrap(ics[f'{control}_{factor}'].RankIC,ics[f'dbd_{factor}'].RankIC,rep=2000)
    tests['two_factor_minus_alpha158']=paired_bootstrap(ics['alpha158_lgb'].RankIC,ics['dbd_two_factor'].RankIC,rep=2000)
    (out/'paired_tests.json').write_text(json.dumps(tests,indent=2))
    # These are genuine OOS upstream features for later stacking; cannot fill 2009-2014.
    z=common[['dbd_two_factor']].rename(columns={'dbd_two_factor':'dbd_score'}).copy()
    z['fit_end']=[pd.Timestamp(d.year-1,12,31) for d in z.index.get_level_values('date')]
    z.reset_index().to_csv(out/'two_factor_oos_scores.csv',index=False)
    meta={'status':'partial paper replication on CSI300 historical membership, not full-market or exact PIT replication','common_rows':len(common),'first_date':str(common.index.get_level_values('date').min()),'last_date':str(common.index.get_level_values('date').max()),'raw_sign':'negative raw factors for long-high portfolio; original paper raw RankIC has opposite sign','model_sign':'maximize positive IC; no OOS sign selection','cutoff':'2025-12-31, future 2026 labels never used','three_factor_combine':'not run: missing small-order active buys and sells','transaction_limits':'9.5% symmetric proxy, not official board/ST daily price limits','exposure':.95,'costs':{'buy':.0005,'sell':.0015},'quantile_output':'monthly 20-day forward returns, not executable long-short NAV','neutralization':'not performed: historical industry and market cap missing','source_hashes':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'research').glob('*.py')}}
    (out/'metadata.json').write_text(json.dumps(meta,indent=2));print(pd.DataFrame(summary).to_string(index=False))
if __name__=='__main__':main()

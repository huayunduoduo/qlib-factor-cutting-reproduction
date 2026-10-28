"""Honest two-factor DBD OOS stacking: 2021-2025 only.

Upstream OOS scores start in 2015; a full 5+1 lower-layer window first permits
testing in 2021. No training-in-sample neural scores fill earlier years.
"""
import json
import numpy as np,pandas as pd
import lightgbm as lgb
from research.data import ROOT,frozen_panel,read_bin
from research.features import alpha158,build_features
from research.preprocess import Transform,residualize
from research.splits import walk_forward
from research.evaluation import daily_ic,paired_bootstrap,portfolio,performance
from evaluate_replication import drawdown_dates

def main():
    out=ROOT/'results_stacking_2021_2025';out.mkdir(exist_ok=True)
    provider=ROOT/'data/research_2009_2025';panel,cal=frozen_panel(provider)
    base=alpha158(provider,'2015-01-01','2025-12-31');cols=list(base)
    scores=pd.read_csv(ROOT/'results_replication/two_factor_oos_scores.csv',parse_dates=['date','fit_end']).set_index(['date','instrument'])
    assert (scores.fit_end<scores.index.get_level_values('date')).all()
    f=build_features(panel,cal)[['label','label_end','member']].reindex(base.index).join(base).join(scores.dbd_score)
    f=f.loc[f.member & f.dbd_score.notna()].copy()
    # Observable same-date cross-sectional residual, not a target regression.
    f['dbd_residual']=residualize(f.dbd_score,f[cols])
    f=f[f.dbd_residual.notna()]
    variants={'A_alpha158':cols,'B_plus_two_dbd':cols+['dbd_score'],'C_plus_residual_two_dbd':cols+['dbd_residual']}
    preds={k:[] for k in variants};folds=[]
    weeks=set(pd.Series(cal,index=cal).groupby(cal.to_period('W-FRI')).max().to_list())
    for fold in walk_forward(f,'2021-01-01','2025-12-31'):
        tr=f.loc[fold['train']];tr=tr.loc[tr.index.get_level_values('date').isin(weeks)];va=f.loc[fold['valid']];te=f.loc[fold['test']]
        assert set(tr.index.get_level_values('date').year)==set(range(fold['train_start'].year,fold['valid_start'].year))
        record={k:str(v) for k,v in fold.items() if k not in ['train','valid','test']};record.update(train_max_label_end=str(tr.label_end.max()),valid_max_label_end=str(va.label_end.max()),train_n=len(tr),valid_n=len(va),test_n=len(te));folds.append(record)
        transform=Transform().fit(tr[cols+['dbd_score','dbd_residual']]);xt=transform.transform(tr[cols+['dbd_score','dbd_residual']]);xv=transform.transform(va[cols+['dbd_score','dbd_residual']]);xe=transform.transform(te[cols+['dbd_score','dbd_residual']])
        yr=fold['test_start'].year
        pd.DataFrame({'median':transform.median,'scale':transform.scale}).to_csv(out/f'preprocess_{yr}.csv')
        yt=tr.label.groupby(level='date').transform(lambda s:(s-s.mean())/(s.std()+1e-8));yv=va.label.groupby(level='date').transform(lambda s:(s-s.mean())/(s.std()+1e-8))
        for name,columns in variants.items():
            m=lgb.LGBMRegressor(n_estimators=200,learning_rate=.05,num_leaves=15,n_jobs=2,random_state=2026,verbosity=-1)
            m.fit(xt[columns],yt,eval_set=[(xv[columns],yv)],callbacks=[lgb.early_stopping(20,verbose=False)])
            m.booster_.save_model(str(out/f'{name}_{yr}.txt'));preds[name].append(pd.DataFrame({'score':m.predict(xe[columns]),'label':te.label},index=te.index))
        print('STACK FOLD',yr,flush=True)
    pd.DataFrame(folds).to_csv(out/'folds.csv',index=False)
    summary=[];yearly=[];icy=[];ics={};benchmark=read_bin(provider,'SH000300','close',cal).pct_change(fill_method=None)
    for name,parts in preds.items():
        pred=pd.concat(parts).sort_index();assert not pred.index.has_duplicates;pred.to_csv(out/f'{name}_predictions.csv.gz')
        ic=daily_ic(pred);ics[name]=ic;ic.to_csv(out/f'{name}_ic.csv')
        bt,h=portfolio(panel,pred,cal[(cal>='2021-01-01')&(cal<='2025-12-31')]);bt['benchmark_return']=benchmark.reindex(bt.index);bt.to_csv(out/f'{name}_portfolio.csv');h.to_csv(out/f'{name}_holdings.csv',index=False)
        summary.append({'experiment':name,'RankIC':ic.RankIC.mean(),**performance(bt),**drawdown_dates(bt)})
        for year,g in bt.groupby(bt.index.year):
            g=g.copy();g.nav=(1+g.net_return).cumprod();yearly.append({'experiment':name,'year':year,**performance(g),**drawdown_dates(g)})
        for year,g in ic.groupby(ic.index.year):icy.append({'experiment':name,'year':year,'RankIC':g.RankIC.mean(),'days':len(g)})
    pd.DataFrame(summary).to_csv(out/'summary.csv',index=False);pd.DataFrame(yearly).to_csv(out/'yearly_portfolio.csv',index=False);pd.DataFrame(icy).to_csv(out/'yearly_ic.csv',index=False)
    tests={k:paired_bootstrap(ics['A_alpha158'].RankIC,v.RankIC,rep=2000) for k,v in ics.items() if k!='A_alpha158'}
    (out/'paired_tests.json').write_text(json.dumps(tests,indent=2))
    (out/'metadata.json').write_text(json.dumps({'range':'2021-2025','reason':'first honest upstream OOS year 2015, full downstream 5 train + 1 validation','upstream':'two-factor symmetric orthogonal equal-weight DBD; not three-factor DBD-Combine','residual':'same-date label-free regression on Alpha158, all variants common support','seed':2026,'max_trees':200,'early_stop':20},indent=2))
    print(pd.DataFrame(summary).to_string(index=False))
if __name__=='__main__':main()

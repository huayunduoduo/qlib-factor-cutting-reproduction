"""Unified incremental study. Run from this directory; legacy results are untouched."""
import argparse,json,hashlib,platform,gzip
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from research.data import ROOT,frozen_panel,load_wind,audit,read_bin
from research.features import build_features,alpha158
from research.preprocess import Transform,residualize,neutralize
from research.splits import walk_forward
from research.regime import CausalHMM,market_features
from research.evaluation import daily_ic,paired_bootstrap,portfolio,performance

def main():
    from threadpoolctl import threadpool_limits
    threadpool_limits(2)
    p=argparse.ArgumentParser();p.add_argument('--out',default='results_research')
    p.add_argument('--provider',default='data/research_2009_2025')
    p.add_argument('--features',choices=['alpha158','small'],default='alpha158')
    p.add_argument('--model',choices=['ridge','lightgbm'],default='ridge')
    p.add_argument('--normalization',choices=['robust','rank_gauss'],default='robust')
    p.add_argument('--start',default='2015-01-01');p.add_argument('--end',default='2025-12-31')
    p.add_argument('--horizon',type=int,default=20);p.add_argument('--step-months',type=int,default=12)
    p.add_argument('--wind');p.add_argument('--neutralize',action='store_true')
    p.add_argument('--active-rule',choices=['return','close']);p.add_argument('--regime',action='store_true')
    p.add_argument('--neural-scores',help='Honest out-of-fold CSV from train_neural.py; never fitted scores')
    a=p.parse_args()
    if a.horizon<1 or a.step_months<1:raise ValueError('positive horizon and step required')
    if a.regime and a.model!='ridge':raise ValueError('Use --model ridge with --regime for matched Ridge expert comparisons')
    out=ROOT/a.out;out.mkdir(parents=True,exist_ok=True)
    panel,calendar=frozen_panel(ROOT/a.provider);contract={}
    if pd.Timestamp(a.end)>calendar.max():
        raise ValueError(f'Requested end {a.end} exceeds provider calendar {calendar.max().date()}; supply complete prices/calendar/historical membership, or explicitly shorten --end. Never silently truncate.')
    if pd.Timestamp(a.start)-pd.DateOffset(years=6)<calendar.min():
        raise ValueError('Provider cannot cover the full 5-year training plus 1-year validation window')
    if a.wind:
        w,contract=load_wind(ROOT/a.wind,calendar)
        if contract.get('trade_count_verified') and 'trade_count' in w:
            raw=w.amount*float(contract['amount_to_yuan'])
            w['avg_trade_amount']=raw/w.trade_count.where(w.trade_count>0)
        else: w=w.drop(columns=['trade_count'],errors='ignore')
        extra=[c for c in ['trade_count','avg_trade_amount','small_buy','small_sell','market_cap','industry'] if c in w]
        if a.active_rule and not contract.get('small_active_flow_verified'):raise ValueError('Active flow field semantics not reviewed')
        if a.neutralize and not contract.get('pit_exposures_verified'):raise ValueError('Historical PIT exposures not reviewed')
        panel=panel.join(w[extra])
    if a.neutralize and not a.wind:raise ValueError('Neutralization requires PIT Wind exposures')
    audit_info=audit(panel,out/'data_audit.json')
    audit_info['provider']=a.provider
    if a.provider!='data/frozen_cn_data':audit_info['source']='separately imported provider; inspect wind_provenance.json'
    (out/'data_audit.json').write_text(json.dumps(audit_info,ensure_ascii=False,indent=2),encoding='utf-8')
    frame=build_features(panel,calendar,a.horizon,a.active_rule)
    auxiliary=['label','label_end','member']; cuts=[c for c in frame if c.startswith('cut_')]
    if a.features=='alpha158':
        needed_start=max(calendar.min(),pd.Timestamp(a.start)-pd.DateOffset(years=6))
        base=alpha158(ROOT/a.provider,str(needed_start.date()),a.end)
        frame=frame[auxiliary+cuts].reindex(base.index).join(base);basecols=list(base);del base
    else:basecols=[c for c in frame if c not in auxiliary+cuts]
    learned=[]
    if a.neural_scores:
        s=pd.read_csv(ROOT/a.neural_scores,parse_dates=['date','fit_end'])
        if not (s.fit_end<s.date).all():raise ValueError('Neural feature provenance leaks future training information')
        if s.duplicated(['date','instrument']).any():raise ValueError('Duplicate neural scores')
        learned=[c for c in ['gru_score','dbd_score'] if c in s]
        frame=frame.join(s.set_index(['date','instrument'])[learned])
    frame=frame.replace([np.inf,-np.inf],np.nan)
    # Use identical intersection for all ablations, including baseline.
    common=frame.member & frame[cuts+learned].notna().all(axis=1)
    frame=frame.loc[common].copy()
    if not len(frame):raise ValueError('No common valid factor samples')
    if a.neutralize:
        exp=panel.reindex(frame.index)[['market_cap','industry']]
        frame[basecols+cuts+learned]=neutralize(frame[basecols+cuts+learned],exp)
        frame=frame.dropna(subset=basecols+cuts+learned)
    specs={'E0_baseline':basecols,'E1_raw_cut':basecols+cuts,'E2_residual_cut':basecols+[f'resid_{c}' for c in cuts]}
    if learned:specs['E3_neural_oof']=basecols+cuts+learned
    for c in cuts:frame['resid_'+c]=residualize(frame[c],frame[basecols])
    # Market features come from historical constituents at each date.
    market=market_features(panel,calendar) if a.regime else None
    predictions={k:[] for k in specs};folds=[]
    if a.regime:predictions['E4_regime']=[]
    for fold in walk_forward(frame,a.start,a.end,step_months=a.step_months):
        tr=frame.loc[fold['train']];va=frame.loc[fold['valid']];te=frame.loc[fold['test']]
        # Weekly training subsampling; validation/test stay on daily common support.
        dates=tr.index.get_level_values('date');weekly=pd.Series(calendar,index=calendar).groupby(calendar.to_period('W-FRI')).max()
        tr=tr.loc[dates.isin(weekly)]
        if min(len(tr),len(va),len(te))<100:raise ValueError('Insufficient train/valid/test samples')
        record={k:str(v) for k,v in fold.items() if k not in ['train','valid','test']}
        record.update(train_first_date=str(tr.index.get_level_values('date').min()),train_last_date=str(tr.index.get_level_values('date').max()),valid_first_date=str(va.index.get_level_values('date').min()),test_last_date=str(te.index.get_level_values('date').max()),train_n=len(tr),valid_n=len(va),test_n=len(te),train_max_label_end=str(tr.label_end.max()),valid_max_label_end=str(va.label_end.max()))
        folds.append(record);print('FOLD',record,flush=True)
        feature_columns=[c for c in frame.columns if c not in auxiliary]
        transform=Transform(a.normalization).fit(tr[feature_columns])
        xt=transform.transform(tr[feature_columns]);xv=transform.transform(va[feature_columns]);xe=transform.transform(te[feature_columns])
        pd.DataFrame({'median':transform.median,'scale':transform.scale}).to_csv(out/f'preprocess_{fold["test_start"].year}.csv')
        yt=tr.label.groupby(level='date').transform(lambda s:(s-s.mean())/(s.std()+1e-8))
        yv=va.label.groupby(level='date').transform(lambda s:(s-s.mean())/(s.std()+1e-8))
        for name,columns in specs.items():
            if a.model=='lightgbm':
                import lightgbm as lgb
                model=lgb.LGBMRegressor(n_estimators=200,learning_rate=.05,num_leaves=15,n_jobs=2,random_state=2026,verbosity=-1)
                model.fit(xt[columns],yt,eval_set=[(xv[columns],yv)],callbacks=[lgb.early_stopping(20,verbose=False)])
                model.booster_.save_model(str(out/f'{name}_{fold["test_start"].year}.txt'))
            else:model=Ridge(alpha=100).fit(xt[columns],yt)
            score=pd.Series(model.predict(xe[columns]),index=te.index,name='score')
            prediction=score.to_frame().join(te[['label']]);predictions[name].append(prediction)
            prediction.to_parquet(out/f'{name}_{fold["test_start"].year}_predictions.parquet')
            if a.regime and name==list(specs)[-1]:
                mh=market.loc[(market.index>=fold['train_start'])&(market.index<fold['valid_start'])]
                hmm=CausalHMM().fit(mh)
                forward=market.loc[(market.index>=mh.index.min())&(market.index<=fold['test_end'])]
                probs=pd.DataFrame(hmm.filter(forward),index=forward.index)
                aligned=probs.reindex(tr.index.get_level_values('date')).set_axis(tr.index)
                experts=[]
                for state in range(3):
                    expert=Ridge(alpha=100).fit(xt[columns],yt,sample_weight=aligned[state].fillna(1/3))
                    experts.append(expert.predict(xe[columns]))
                tp=probs.reindex(te.index.get_level_values('date')).fillna(1/3).to_numpy()
                gated=(np.column_stack(experts)*tp).sum(1)
                gated_frame=pd.DataFrame({'score':gated,'label':te.label},index=te.index)
                for state in range(3):gated_frame[f'p{state}']=tp[:,state]
                predictions['E4_regime'].append(gated_frame)
                probs.to_csv(out/f'regime_prob_{fold["test_start"].year}.csv')
        pd.DataFrame(folds).to_csv(out/'folds.csv',index=False)
    summary=[];ics={};yearly=[];costs=[];yearly_portfolio=[]
    benchmark=read_bin(ROOT/a.provider,'SH000300','close',calendar).pct_change(fill_method=None)
    for name,parts in predictions.items():
        pred=pd.concat(parts).sort_index()
        if pred.index.has_duplicates:raise ValueError('Overlapping OOS folds')
        target=out/f'{name}_predictions.csv.gz';temporary=out/f'{name}_predictions.pending.gz'
        pred.to_csv(temporary,compression='gzip',chunksize=10000)
        temporary.replace(target)
        ic=daily_ic(pred);ics[name]=ic;ic.to_csv(out/f'{name}_ic.csv')
        if name=='E4_regime':
            prob=pred[['p0','p1','p2']].groupby(level='date').first()
            conditional=[]
            for year,z in ic.join(prob).dropna().groupby(lambda date:date.year):
                for state in range(3):
                    w=z[f'p{state}']
                    conditional.append({'year':year,'state':state,'probability_weighted_RankIC':float((w*z.RankIC).sum()/w.sum()),'effective_days':float(w.sum()),'note':'state IDs are local to each annual fit; compare within year only'})
            pd.DataFrame(conditional).to_csv(out/'regime_conditional_ic.csv',index=False)
        for year,g in ic.groupby(ic.index.year):yearly.append({'experiment':name,'year':year,'RankIC':g.RankIC.mean(),'daily_ICIR':g.RankIC.mean()/g.RankIC.std(),'days':len(g)})
        for slippage in [0.,.001,.002]:
            bt,h=portfolio(panel,pred,calendar[(calendar>=pred.index.get_level_values('date').min())&(calendar<=pd.Timestamp(a.end))],slippage=slippage)
            bt['benchmark_return']=benchmark.reindex(bt.index)
            perf=performance(bt);costs.append({'experiment':name,'slippage_per_side':slippage,**perf})
            if slippage==0:
                bt.to_csv(out/f'{name}_portfolio.csv');h.to_csv(out/f'{name}_holdings.csv',index=False)
                summary.append({'experiment':name,'RankIC':ic.RankIC.mean(),'daily_ICIR':ic.RankIC.mean()/ic.RankIC.std(),**perf})
                for year,g in bt.groupby(bt.index.year):
                    z=g.copy();z['nav']=(1+z.net_return).cumprod()
                    yearly_portfolio.append({'experiment':name,'year':year,**performance(z)})
    pd.DataFrame(summary).to_csv(out/'summary.csv',index=False)
    pd.DataFrame(yearly).to_csv(out/'yearly_ic.csv',index=False);pd.DataFrame(costs).to_csv(out/'cost_sensitivity.csv',index=False)
    pd.DataFrame(yearly_portfolio).to_csv(out/'yearly_portfolio.csv',index=False)
    oos=frame.loc[(frame.index.get_level_values('date')>=pd.Timestamp(a.start))&(frame.index.get_level_values('date')<=pd.Timestamp(a.end))]
    quantiles=[]
    for c in cuts+['resid_'+c for c in cuts]:
        daily_ic(oos[[c,'label']].rename(columns={c:'score'})).to_csv(out/f'{c}_single_factor_ic.csv')
        for date,g in oos[[c,'label']].dropna().groupby(level='date'):
            if len(g)<20 or g[c].nunique()<5:continue
            bins=pd.qcut(g[c].rank(method='first'),5,labels=False)
            for q,part in g.groupby(bins):quantiles.append({'factor':c,'date':date,'quintile_low_to_high':int(q)+1,'mean_forward_label':part.label.mean(),'n':len(part)})
    pd.DataFrame(quantiles).to_csv(out/'single_factor_quantile_forward_returns.csv',index=False)
    baseic=ics['E0_baseline'].RankIC
    tests={name:paired_bootstrap(baseic,g.RankIC,block=max(21,a.horizon+1)) for name,g in ics.items() if name!='E0_baseline'}
    (out/'paired_increment_test.json').write_text(json.dumps(tests,indent=2))
    meta={'args':vars(a),'python':platform.python_version(),'features':basecols,'cut_columns':cuts,
          'learned_columns':learned,'label':f'close(t+{a.horizon+1})/close(t+1)-1',
          'wind_download_executed_here':False,'trade_simulator':'monthly close, 1-day signal lag, drifting weights, 9.5% limit proxy; no volume capacity; explicit last-known-price stale valuation',
          'research_status':'historical development evaluation, not untouched holdout or paper-exact reproduction',
          'regime_experts':'Ridge weighted experts; same feature columns; evaluate against Ridge baseline for matched models',
          'entrypoint_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'research').glob('*.py')}}
    (out/'run_metadata.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
    print(pd.DataFrame(summary).to_string(index=False))

if __name__=='__main__':main()

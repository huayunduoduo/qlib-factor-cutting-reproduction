"""Independent recomputation of published metrics, fold timing and saved predictions."""
import json,hashlib
from pathlib import Path
import numpy as np,pandas as pd
from research.data import ROOT
from research.evaluation import daily_ic

def verify():
    checks=[]
    calendar=pd.to_datetime(pd.read_csv(ROOT/'data/research_2009_2025/calendars/day.txt',header=None)[0]);tail_start=calendar.iloc[-21]
    for dirname in ['results_2015_2025','results_with_reversal','results_raw_factors','results_replication','results_stacking_2021_2025']:
        folder=ROOT/dirname
        if not (folder/'summary.csv').exists():continue
        s=pd.read_csv(folder/'summary.csv');key='factor' if 'factor' in s else 'experiment'
        for row in s.to_dict('records'):
            bt=pd.read_csv(folder/f'{row[key]}_portfolio.csv',parse_dates=['date']).set_index('date')
            nav=(1+bt.net_return).cumprod()
            assert np.allclose(nav,bt.nav,rtol=1e-9,atol=1e-10)
            gross=(1+bt.gross_return)*(1-bt.cost)-1
            assert np.allclose(gross,bt.net_return,atol=1e-12)
            expected=bt.buy_turnover*.0005+bt.sell_turnover*.0015
            assert np.allclose(expected,bt.cost,atol=1e-12)
            cagr=float(nav.iloc[-1]**(252/len(nav))-1)
            mdd=float((nav/np.maximum.accumulate(np.r_[1.,nav])[1:]-1).min())
            assert abs(cagr-row['CAGR'])<1e-10
            assert abs(mdd-row['MDD'])<1e-10
            assert bt.index.max()==pd.Timestamp('2025-12-31')
            if dirname!='results_stacking_2021_2025':assert set(bt.index.year)==set(range(2015,2026))
            checks.append({'folder':dirname,'experiment':row[key],'nav_cost_cagr_mdd':'PASS','days':len(bt)})
        if (folder/'folds.csv').exists():
            f=pd.read_csv(folder/'folds.csv')
            assert (pd.to_datetime(f.train_max_label_end)<pd.to_datetime(f.valid_start)).all()
            assert (pd.to_datetime(f.valid_max_label_end)<pd.to_datetime(f.test_start)).all()
        for p in folder.glob('*_predictions.csv.gz'):
            z=pd.read_csv(p,parse_dates=['date']).set_index(['date','instrument']);assert not z.index.has_duplicates
            assert z.loc[z.index.get_level_values('date')>=tail_start,'label'].isna().all()
    # Replay one held-out cross-section from every saved neural checkpoint.
    import torch
    torch.set_num_threads(2)
    from research.neural import SequenceModel
    replay=[]
    for factor in ['amplitude','reversal']:
        cache=ROOT/'sequence_cache'/factor
        if not (cache/'index.parquet').exists():continue
        index=pd.read_parquet(cache/'index.parquet');mi=pd.MultiIndex.from_frame(index[['date','instrument']]);x=np.load(cache/'x.npy',mmap_mode='r')
        for info in sorted((ROOT/'results_neural_grid'/factor).glob('*.json')):
            meta=json.loads(info.read_text());pred=pd.read_parquet(info.with_name(info.stem+'_predictions.parquet'))
            assert not pred.index.has_duplicates and (pred.fit_end<pred.index.get_level_values('date')).all()
            assert meta['epochs_run']<=100 and meta['patience']==20
            state=torch.load(info.with_suffix('.pt'),map_location='cpu',weights_only=False)
            model=SequenceModel(dual=meta['dual']);model.load_state_dict(state['state_dict']);model.eval()
            first=pred.loc[[pred.index.get_level_values('date').min()]];pos=mi.get_indexer(first.index);assert (pos>=0).all()
            raw=np.asarray(x[pos]);scaled=np.clip((raw-state['median'])/state['scale'],-5,5)
            with torch.no_grad():y=model(torch.tensor(scaled,dtype=torch.float32),torch.tensor(raw,dtype=torch.float32)).numpy()
            error=float(np.max(np.abs(y-first.score.to_numpy())))
            assert error<1e-5,(info,error)
            replay.append({'factor':factor,'model':info.stem,'rows':len(first),'max_absolute_error':error,'status':'PASS'})
    out={'portfolio_checks':checks,'neural_checkpoint_replay':replay,'completed_neural_models':len(replay)}
    (ROOT/'verification_current.json').write_text(json.dumps(out,indent=2));print('PASS',len(checks),'portfolios',len(replay),'neural models')
if __name__=='__main__':verify()

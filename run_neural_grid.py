"""Memory-mapped annual GRU/DBD runs. No training years or epochs are omitted.

Uses the project's 5 train + 1 validation protocol; not paper-exact universe.
Each optimizer batch is one complete weekly cross-section, IC loss.
"""
import argparse,json,os,time,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from research.data import ROOT,frozen_panel,load_wind

def prepare():
    out=ROOT/'sequence_cache';out.mkdir(exist_ok=True)
    panel,cal=frozen_panel(ROOT/'data/research_2009_2025')
    wind,contract=load_wind(ROOT/'verified_wind_join',cal)
    panel=panel.join(wind[['trade_count']])
    for factor in ['amplitude','reversal']:
        dest=out/factor;dest.mkdir(exist_ok=True)
        if (dest/'ready.json').exists():continue
        xs=[];indices=[];ys=[];ends=[]
        for code,g in panel.groupby(level='instrument'):
            g=g.droplevel('instrument').reindex(cal)
            ret=g.close/g.close.shift()-1
            ab=np.c_[g.close,g.high/g.low-1] if factor=='amplitude' else np.c_[g.amount*1000/g.trade_count.where(g.trade_count>0),ret]
            seq=np.lib.stride_tricks.sliding_window_view(ab,20,axis=0).transpose(0,2,1)
            ok=np.isfinite(seq).all((1,2)) & (np.lib.stride_tricks.sliding_window_view(g.volume.to_numpy(),20)>0).all(1)
            ok &= g.member.fillna(False).to_numpy(dtype=bool)[19:] & (cal[19:]>=pd.Timestamp('2009-01-01'))
            ii=np.flatnonzero(ok)+19
            xs.append(seq[ok].astype('float32'))
            indices.append(pd.DataFrame({'date':cal[ii],'instrument':code}))
            ys.append((g.close.shift(-21)/g.close.shift(-1)-1).to_numpy()[ii].astype('float32'))
            ends.append(pd.Series(cal,index=cal).shift(-21).to_numpy()[ii])
        index=pd.concat(indices,ignore_index=True);order=index.sort_values(['date','instrument']).index.to_numpy()
        index=index.iloc[order].reset_index(drop=True)
        np.save(dest/'x.npy',np.concatenate(xs)[order]);np.save(dest/'y.npy',np.concatenate(ys)[order])
        index['label_end']=np.concatenate(ends)[order];index.to_parquet(dest/'index.parquet')
        (dest/'ready.json').write_text(json.dumps({'factor':factor,'rows':len(index),'label':'t+21 / t+1 -1','latest_date':str(index.date.max()),'amount_to_yuan':1000}))
        print('CACHE',factor,len(index),flush=True)

def work(job):
    # Spawned children read shared mmap arrays instead of duplicating full panels.
    factor,year,dual=job
    import torch
    from research.neural import fit_predict
    dest=ROOT/'results_neural_grid'/factor;dest.mkdir(parents=True,exist_ok=True)
    name='dbd' if dual else 'gru';stem=f'{name}_{year}'
    if (dest/f'{stem}.json').exists():return stem+' cached'
    cache=ROOT/'sequence_cache'/factor
    x=np.load(cache/'x.npy',mmap_mode='r');y=np.load(cache/'y.npy',mmap_mode='r')
    idx=pd.read_parquet(cache/'index.parquet')
    cal=pd.DatetimeIndex(pd.read_csv(ROOT/'data/research_2009_2025/calendars/day.txt',header=None)[0])
    weeks=set(pd.Series(cal,index=cal).groupby(cal.to_period('W-FRI')).max())
    test_start=pd.Timestamp(year,1,1);valid_start=pd.Timestamp(year-1,1,1);train_start=pd.Timestamp(year-6,1,1)
    train=[];valid=[];test=[]
    for date,ii in idx.groupby('date',sort=True).indices.items():
        if date<train_start or date>=pd.Timestamp(year+1,1,1):continue
        if date<valid_start:
            if date not in weeks:continue
            ii=ii[(idx.label_end.iloc[ii]<valid_start).to_numpy() & np.isfinite(y[ii])];part=train
        elif date<test_start:
            if date not in weeks:continue
            ii=ii[(idx.label_end.iloc[ii]<test_start).to_numpy() & np.isfinite(y[ii])];part=valid
        else:part=test
        if len(ii)<20:continue
        mi=pd.MultiIndex.from_frame(idx.iloc[ii][['date','instrument']])
        part.append((date,mi,np.asarray(x[ii]),np.asarray(y[ii])))
    assert train and valid and test
    started=time.time()
    log=dest/f'{stem}.log'
    import contextlib
    with log.open('a',buffering=1) as f,contextlib.redirect_stdout(f):
        print('TRAIN',job,len(train),len(valid),len(test),flush=True)
        result,state=fit_predict(train,valid,test,epochs=100,patience=20,seed=2026,dual=dual,checkpoint=dest/f'{stem}_resume.pt')
    torch.save(state,dest/f'{stem}.pt')
    pred=pd.concat([pd.Series(v,index=i,name='score') for i,v in result]).to_frame()
    label=pd.Series(y,index=pd.MultiIndex.from_frame(idx[['date','instrument']]),name='label')
    pred=pred.join(label)
    fit_end=idx.loc[(idx.date>=train_start)&(idx.date<test_start)&(idx.label_end<test_start),'label_end'].max()
    pred['fit_end']=fit_end
    assert (pred.index.get_level_values('date')>fit_end).all()
    pred.to_parquet(dest/f'{stem}_predictions.parquet')
    meta={'factor':factor,'year':year,'dual':dual,'seconds':time.time()-started,'train_start':str(train_start),'valid_start':str(valid_start),'test_start':str(test_start),'fit_end':str(fit_end),'train_dates':len(train),'valid_dates':len(valid),'train_samples':sum(len(v[1]) for v in train),'valid_samples':sum(len(v[1]) for v in valid),'test_samples':len(pred),'epochs_run':len(state['history']),'best_epoch':min(state['history'],key=lambda r:r['valid_negative_ic'])['epoch'],'max_epochs':100,'patience':20,'seed':2026,'batch':'one weekly cross-section','hidden':64,'layers':2,'loss':'negative Pearson IC','optimizer':'Adam lr=0.001 (implementation assumption)','normalization':'training median/MAD, clip +/-5 (implementation assumption)'}
    (dest/f'{stem}.json').write_text(json.dumps(meta,indent=2));return f'{factor} {stem} done {meta["epochs_run"]} epochs'

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=3);p.add_argument('--prepare-only',action='store_true');a=p.parse_args()
    prepare()
    if not a.prepare_only:
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor,as_completed
        jobs=[(factor,year,dual) for year in range(2015,2026) for factor in ['amplitude','reversal'] for dual in [False,True]]
        with ProcessPoolExecutor(max_workers=a.workers,mp_context=mp.get_context('spawn')) as pool:
            for f in as_completed([pool.submit(work,j) for j in jobs]):print(f.result(),flush=True)

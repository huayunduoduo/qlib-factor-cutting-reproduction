"""Generate honest OOS GRU/DBD-GRU factor scores, with provenance for stacking."""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from research.data import ROOT,frozen_panel,load_wind
from research.features import build_features
from research.splits import walk_forward
from research.evaluation import daily_ic

def main():
    p=argparse.ArgumentParser();p.add_argument('--start',default='2015-01-01');p.add_argument('--end',default='2025-12-31')
    p.add_argument('--epochs',type=int,default=100);p.add_argument('--seed',type=int,default=10026)
    p.add_argument('--factor',choices=['amplitude','reversal','active'],default='amplitude');p.add_argument('--wind')
    p.add_argument('--active-rule',choices=['return','close']);p.add_argument('--out',default='results_neural')
    p.add_argument('--provider',default='data/research_2009_2025')
    a=p.parse_args()
    import torch
    from research.neural import fit_predict
    out=ROOT/a.out;out.mkdir(parents=True,exist_ok=True)
    panel,calendar=frozen_panel(ROOT/a.provider)
    if a.factor!='amplitude':
        if not a.wind:raise ValueError('Wind data required')
        w,contract=load_wind(ROOT/a.wind,calendar)
        if a.factor=='reversal':
            if not contract.get('trade_count_verified'):raise ValueError('Unverified genuine trade_count')
            w['avg_trade_amount']=w.amount*contract['amount_to_yuan']/w.trade_count.where(w.trade_count>0)
            panel=panel.join(w[['avg_trade_amount']])
        else:
            if not a.active_rule or not contract.get('small_active_flow_verified'):raise ValueError('Review active-flow rule and fields')
            panel=panel.join(w[['small_buy','small_sell']])
    frame=build_features(panel,calendar,20,a.active_rule)
    records=[]
    for code,g in panel.groupby(level='instrument'):
        g=g.droplevel('instrument').reindex(calendar);r=g.close/g.close.shift()-1
        if a.factor=='amplitude':ab=np.c_[g.close,g.high/g.low-1]
        elif a.factor=='reversal':ab=np.c_[g.avg_trade_amount,r]
        else:ab=np.c_[r if a.active_rule=='return' else g.close,(g.small_buy-g.small_sell)/(g.small_buy+g.small_sell).replace(0,np.nan)]
        seq=np.lib.stride_tricks.sliding_window_view(ab,20,axis=0).transpose(0,2,1)
        valid=np.isfinite(seq).all((1,2))&np.lib.stride_tricks.sliding_window_view(g.volume.to_numpy(),20).min(1).__gt__(0)
        valid &= g.member.fillna(False).to_numpy(dtype=bool)[19:]
        for j in np.flatnonzero(valid):
            date=calendar[j+19];key=(date,code)
            records.append((key,seq[j].astype('float32')))
    xmap=dict(records);frame=frame.loc[frame.index.isin(xmap)]
    weeks=set(pd.Series(calendar,index=calendar).groupby(calendar.to_period('W-FRI')).max())
    def batches(part,weekly=False,predict=False):
        result=[]
        for date,g in part.groupby(level='date'):
            if weekly and date not in weeks:continue
            if not predict:g=g.dropna(subset=['label'])
            if len(g)<20:continue
            result.append((date,g.index,np.stack([xmap[k] for k in g.index]),g.label.to_numpy(dtype='float32')))
        return result
    outputs=[]
    for fold in walk_forward(frame,a.start,a.end):
        tr=frame.loc[fold['train']];va=frame.loc[fold['valid']];te=frame.loc[fold['test']]
        train=batches(tr,True);valid=batches(va,True);test=batches(te,predict=True)
        if min(len(train),len(valid),len(test))<2:raise ValueError('Insufficient sequence dates for fold')
        result=None
        for dual,name in [(False,'gru_score'),(True,'dbd_score')]:
            print('TRAIN',fold['test_start'],name,len(train),flush=True)
            pred,state=fit_predict(train,valid,test,epochs=a.epochs,seed=a.seed,dual=dual)
            series=pd.concat([pd.Series(v,index=idx,name=name) for idx,v in pred])
            result=series.to_frame() if result is None else result.join(series)
            torch.save(state,out/f'{name}_{fold["test_start"].year}.pt')
            (out/f'{name}_{fold["test_start"].year}_history.json').write_text(json.dumps(state['history'],indent=2))
        # Includes the latest validation outcome used in checkpoint selection.
        result['fit_end']=max(tr.label_end.max(),va.label_end.max())
        outputs.append(result);pd.concat(outputs).reset_index().to_csv(out/'oof_scores.csv',index=False)
    pred=pd.concat(outputs).join(frame.label)
    for name in ['gru_score','dbd_score']:daily_ic(pred.rename(columns={name:'score'})[['score','label']]).to_csv(out/f'{name}_ic.csv')
    (out/'metadata.json').write_text(json.dumps({'args':vars(a),'status':'adaptation, not exact paper replication',
       'mask':'raw A >= arithmetic median; low < median; zero entire time-step inputs',
       'label':'close(t+21)/close(t+1)-1','sampling':'weekly train/valid, daily OOS',
       'stacking':'provide OOS scores starting early enough to cover downstream training folds'},indent=2))

if __name__=='__main__':main()

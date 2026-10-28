import numpy as np
import pandas as pd

def cut(a,b,fraction=.5,aggregation='sum'):
    if len(a)!=20 or not np.isfinite(a).all() or not np.isfinite(b).all(): return np.nan
    # Stable ties: older day first. Exactly k observations per group.
    order=np.argsort(a,kind='stable'); k=int(len(a)*fraction)
    fn=np.sum if aggregation=='sum' else np.mean
    return float(fn(b[order[-k:]])-fn(b[order[:k]]))

def build_features(panel,calendar,horizon=20,active_rule=None):
    out=[]
    for code,g in panel.groupby(level='instrument'):
        g=g.droplevel('instrument').reindex(calendar)
        c=g.close; r=c/c.shift(1)-1; amp=g.high/g.low-1
        f=pd.DataFrame(index=calendar)
        # A small fallback baseline is explicitly not Alpha158.
        for n in [5,10,20,60]:
            f[f'MOM{n}']=c/c.shift(n)-1
            f[f'VOL{n}']=r.rolling(n).std()
            f[f'VOLUME{n}']=g.volume/g.volume.rolling(n).mean()
        f['KMID']=(g.close-g.open)/g.open
        f['AMPLITUDE']=amp
        sources={'cut_amplitude':(c,amp,.25,'mean')}
        if 'avg_trade_amount' in g: sources['cut_reversal']=(g.avg_trade_amount,r,.5,'sum')
        if active_rule:
            if active_rule not in ['return','close']: raise ValueError('active_rule must be return or close')
            if not all(k in g for k in ['small_buy','small_sell']): raise ValueError('Missing active small-order amounts')
            ratio=(g.small_buy-g.small_sell)/(g.small_buy+g.small_sell).replace(0,np.nan)
            sources['cut_active']=(r if active_rule=='return' else c,ratio,.25,'mean')
        for name,(a,b,frac,agg) in sources.items():
            vals=np.full(len(g),np.nan)
            if len(g)>=20:
                aw=np.lib.stride_tricks.sliding_window_view(a.to_numpy(),20)
                bw=np.lib.stride_tricks.sliding_window_view(b.to_numpy(),20)
                vw=np.lib.stride_tricks.sliding_window_view(g.volume.to_numpy(),20)
                order=np.argsort(aw,axis=1,kind='stable');k=int(20*frac)
                ordered=np.take_along_axis(bw,order,axis=1)
                fn=np.sum if agg=='sum' else np.mean
                v=(np.mean(ordered[:,:k],axis=1) if name=='cut_active' else fn(ordered[:,-k:],axis=1)-fn(ordered[:,:k],axis=1))
                valid=np.isfinite(aw).all(1)&np.isfinite(bw).all(1)&(vw>0).all(1)
                vals[19:]=np.where(valid,v,np.nan)
            f[name]=vals
        # t close signal, t+1 close entry, t+21 close exit for horizon 20.
        f['label']=c.shift(-(horizon+1))/c.shift(-1)-1
        f['label_end']=pd.Series(calendar,index=calendar).shift(-(horizon+1))
        f['member']=g.member.fillna(False).astype(bool)
        f['instrument']=code; f.index.name='date';out.append(f.reset_index())
    return pd.concat(out).set_index(['date','instrument']).sort_index()

def alpha158(provider,start,end):
    import qlib
    from qlib.constant import REG_CN
    from qlib.data import D
    from qlib.contrib.data.loader import Alpha158DL
    qlib.init(provider_uri=str(provider),region=REG_CN,kernels=2,expression_cache=None,dataset_cache=None)
    expressions,names=Alpha158DL.get_feature_config()
    from qlib.data.cache import H
    import hashlib
    key=hashlib.sha256(str(provider.resolve()).encode()+(provider/'calendars/day.txt').read_bytes()+(provider/'instruments/csi300.txt').read_bytes()).hexdigest()[:12]
    cache=provider.parent/'research_cache'/key;cache.mkdir(parents=True,exist_ok=True)
    parts=[]
    for year in range(pd.Timestamp(start).year,pd.Timestamp(end).year+1):
        lo=max(pd.Timestamp(start),pd.Timestamp(year,1,1));hi=min(pd.Timestamp(end),pd.Timestamp(year,12,31))
        target=cache/f'alpha158_{lo.date()}_{hi.date()}.parquet'
        if target.exists():f=pd.read_parquet(target)
        else:
            f=D.features(D.instruments('csi300'),expressions,start_time=str(lo.date()),end_time=str(hi.date()))
            f=f.astype('float32');f.columns=names;f.index=f.index.set_names(['instrument','date'])
            f=f.reorder_levels(['date','instrument']);f.to_parquet(target)
        parts.append(f)
        H.clear()
        print('Alpha158 loaded',year,len(f),flush=True)
    return pd.concat(parts).sort_index()

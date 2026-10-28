import numpy as np
import pandas as pd
from scipy.special import ndtri

class Transform:
    def __init__(self,method='robust'): self.method=method
    def fit(self,x):
        self.median=x.median().fillna(0)
        self.scale=(x-self.median).abs().median().mul(1.4826).replace(0,1).fillna(1)
        return self
    def transform(self,x):
        x=x.replace([np.inf,-np.inf],np.nan)
        if self.method=='robust': y=((x-self.median)/self.scale).clip(-5,5)
        elif self.method=='rank_gauss':
            def rank(g):
                p=(g.rank(method='average')-.5)/g.count()
                return pd.DataFrame(ndtri(p.clip(.001,.999)),index=g.index,columns=g.columns)
            y=x.groupby(level='date',group_keys=False).apply(rank)
        else: raise ValueError(self.method)
        return y.fillna(0).astype('float32')

def residualize(target,controls,ridge=1e-6):
    """Same-date observable exposures only. Never accepts future label as a control."""
    joined=controls.join(target.rename('__target')).dropna()
    result=pd.Series(np.nan,index=target.index)
    for _,g in joined.groupby(level='date'):
        x=g.drop(columns='__target').to_numpy(float); y=g.__target.to_numpy(float)
        if len(y)<max(20,x.shape[1]+2): continue
        x=(x-x.mean(0))/(x.std(0)+1e-8); x=np.c_[np.ones(len(x)),x]
        penalty=np.eye(x.shape[1])*ridge; penalty[0,0]=0
        beta=np.linalg.solve(x.T@x+penalty,x.T@y)
        result.loc[g.index]=y-x@beta
    return result

def neutralize(features,exposures):
    if not {'industry','market_cap'}.issubset(exposures): raise ValueError('PIT industry and market_cap required')
    e=exposures.copy(); e['logcap']=np.log(e.market_cap.where(e.market_cap>0))
    ctrl=pd.get_dummies(e.industry,prefix='industry',dtype=float).join(e.logcap)
    ctrl.loc[e.industry.isna(),:]=np.nan
    return pd.DataFrame({c:residualize(features[c],ctrl) for c in features})

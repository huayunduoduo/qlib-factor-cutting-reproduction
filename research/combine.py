"""Same-date, label-free symmetric orthogonalization of observable scores."""
import numpy as np
import pandas as pd

def symmetric_combine(scores,epsilon=1e-6):
    result=pd.Series(np.nan,index=scores.index,name='score')
    for _,g in scores.groupby(level='date'):
        g=g.dropna()
        if len(g)<20:continue
        x=g.to_numpy(float);sd=x.std(0)
        if (sd<epsilon).any():continue
        z=(x-x.mean(0))/sd
        eigen,v=np.linalg.eigh(z.T@z/len(z))
        # Symmetric inverse square root, not order-dependent Gram-Schmidt.
        white=z@(v@np.diag(1/np.sqrt(np.maximum(eigen,epsilon)))@v.T)
        result.loc[g.index]=white.mean(1)
    return result

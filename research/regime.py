"""Diagonal Gaussian HMM: training-only EM, causal forward filtering at inference."""
import numpy as np
import pandas as pd
from scipy.special import logsumexp
from sklearn.cluster import KMeans

class CausalHMM:
    def __init__(self,k=3,iterations=25,seed=2026): self.k=k; self.iterations=iterations;self.seed=seed
    def emissions(self,x):
        return -.5*(np.log(2*np.pi*self.var).sum(1)[None,:]+((x[:,None,:]-self.mu)**2/self.var).sum(2))
    def fit(self,x):
        x=np.asarray(x,float);self.mean=x.mean(0);self.scale=x.std(0)+1e-8;x=(x-self.mean)/self.scale
        km=KMeans(self.k,random_state=self.seed,n_init=5).fit(x)
        self.mu=km.cluster_centers_;self.var=np.ones_like(self.mu)
        self.trans=np.full((self.k,self.k),.1/(self.k-1));np.fill_diagonal(self.trans,.9)
        self.pi=np.ones(self.k)/self.k
        for _ in range(self.iterations):
            e=self.emissions(x); a=np.empty_like(e);b=np.zeros_like(e)
            a[0]=np.log(self.pi)+e[0]
            for t in range(1,len(x)): a[t]=e[t]+logsumexp(a[t-1,:,None]+np.log(self.trans),axis=0)
            for t in range(len(x)-2,-1,-1):b[t]=logsumexp(np.log(self.trans)+e[t+1]+b[t+1],axis=1)
            ll=logsumexp(a[-1]);g=np.exp(a+b-ll);mass=g.sum(0)+1e-8
            xi=np.zeros_like(self.trans)
            for t in range(len(x)-1):xi+=np.exp(a[t,:,None]+np.log(self.trans)+e[t+1]+b[t+1]-ll)
            self.trans=(xi+1e-3)/(xi+1e-3).sum(1,keepdims=True)
            self.pi=(g[0]+1e-8)/(g[0]+1e-8).sum()
            self.mu=g.T@x/mass[:,None]
            self.var=np.maximum((g[:,:,None]*(x[:,None,:]-self.mu)**2).sum(0)/mass[:,None],.01)
        return self
    def filter(self,x):
        e=self.emissions((np.asarray(x,float)-self.mean)/self.scale)
        p=self.pi.copy();out=[]
        for t,row in enumerate(e):
            prior=p if t==0 else p@self.trans
            logp=np.log(prior+1e-300)+row;p=np.exp(logp-logsumexp(logp));out.append(p)
        return np.asarray(out)

def market_features(panel,calendar):
    p=panel.close.unstack().reindex(calendar); member=panel.member.unstack().reindex(calendar).fillna(False)
    ret=p.pct_change(fill_method=None).where(member)
    market=ret.mean(1)
    return pd.DataFrame({'market_return':market,'volatility':market.rolling(20).std(),
          'breadth':ret.gt(0).sum(1)/ret.count(1).replace(0,np.nan),
          'dispersion':ret.std(1),'trend':market.rolling(20).sum()}).dropna()

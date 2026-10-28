import numpy as np
import pandas as pd

def daily_ic(df):
    rows=[]
    for date,g in df.groupby(level='date'):
        g=g[['score','label']].dropna()
        if len(g)<20 or g.score.nunique()<2 or g.label.nunique()<2:continue
        rows.append({'date':date,'IC':g.score.corr(g.label),'RankIC':g.score.corr(g.label,method='spearman'),'n':len(g)})
    return pd.DataFrame(rows).set_index('date')

def paired_bootstrap(a,b,block=21,rep=500):
    """Paired circular moving-block bootstrap, preserving local serial dependence."""
    d=(b-a).dropna().to_numpy();n=len(d)
    if n<block:return {'n':n,'mean_delta':None,'ci_low':None,'ci_high':None}
    rng=np.random.default_rng(2026);stats=[]
    for _ in range(rep):
        starts=rng.integers(n,size=int(np.ceil(n/block)))
        idx=((starts[:,None]+np.arange(block))%n).ravel()[:n]
        stats.append(d[idx].mean())
    return {'n':n,'mean_delta':float(d.mean()),'ci_low':float(np.quantile(stats,.025)),
            'ci_high':float(np.quantile(stats,.975)),'block':block,'replications':rep}

def portfolio(panel,pred,calendar,topk=50,buy_cost=.0005,sell_cost=.0015,slippage=0):
    """Monthly long-only, lagged close fills, drifting weights, explicit proxy limits.

    Research simulator: no lot sizes, capacity, delisting proceeds, or full exchange rules.
    Missing quotes use explicit last-known-price marking with stale-position counts.
    This is a research valuation convention, not verified suspension/delisting recovery.
    """
    rawclose=panel.close.unstack().reindex(calendar)
    close=rawclose.ffill()
    vol=panel.volume.unstack().reindex(calendar)
    members=panel.member.unstack().reindex(calendar).fillna(False)
    scores=pred.score.unstack().reindex(calendar)
    returns=close.pct_change(fill_method=None)
    w=pd.Series(0.,index=close.columns);cash=1.;nav=1.;rows=[];holding_rows=[]
    first=scores.dropna(how='all').index.min()
    for j,date in enumerate(calendar):
        if j==0 or date<first:continue
        prev=calendar[j-1];r=returns.loc[date]
        held=w>1e-12
        if r[held].isna().any():raise ValueError(f'Missing valuation on held security at {date}; reconcile corporate actions')
        gross=float((w*r.fillna(0)).sum()); w=w*(1+r.fillna(0))/(1+gross);cash=cash/(1+gross)
        buy=sell=cost=0.; rebalance=(prev.month!=date.month or not rows)
        if rebalance:
            # Trade at today's close with yesterday's complete information.
            eligible=members.loc[prev]&scores.loc[prev].notna()
            # Proxy: reject all trades when |return|>=9.5%; explicit in metadata.
            tradable=(vol.loc[date]>0)&rawclose.loc[date].notna()&(r.abs()<.095)
            # Freeze ranking using yesterday's information. Do not substitute a
            # lower-ranked name after observing today's rejected closing fill.
            chosen=scores.loc[prev].where(eligible).dropna().nlargest(topk).index
            locked=(w>0)&~tradable
            target=pd.Series(0.,index=w.index);target.loc[locked]=w.loc[locked]
            intended=chosen[~locked.reindex(chosen).to_numpy()]
            filled=intended[tradable.reindex(intended).to_numpy()]
            if len(intended):target.loc[filled]+=(max(0.,.95-target.sum())/len(intended))
            delta=target-w;sell=float(-delta.clip(upper=0).sum());buy=float(delta.clip(lower=0).sum())
            cost=buy*(buy_cost+slippage)+sell*(sell_cost+slippage)
            if target.sum()+cost>1+1e-9:raise ValueError('Insufficient cash including costs')
            cash=(1-target.sum()-cost)/(1-cost);w=target/(1-cost)
        net=(1+gross)*(1-cost)-1;nav*=1+net
        rows.append({'date':date,'gross_return':gross,'net_return':net,'nav':nav,
                     'cost':cost,'buy_turnover':buy,'sell_turnover':sell,'gross_exposure':w.sum(),
                     'holdings':int((w>1e-12).sum()),'stale_holdings':int(((w>1e-12)&rawclose.loc[date].isna()).sum()),'rebalance':rebalance})
        if rebalance:
            holding_rows.extend({'date':date,'instrument':code,'weight':weight} for code,weight in w[w>0].items())
    return pd.DataFrame(rows).set_index('date'),pd.DataFrame(holding_rows)

def performance(bt):
    r=bt.net_return;n=len(r);v=bt.nav
    peak=np.maximum.accumulate(np.r_[1.,v.to_numpy()])[1:]
    result={'days':n,'CAGR':float(v.iloc[-1]**(252/n)-1),'Sharpe_rf0':float(r.mean()/r.std()*np.sqrt(252)),
            'MDD':float((v/peak-1).min()),'mean_two_sided_turnover':float((bt.buy_turnover+bt.sell_turnover).mean()),
            'stale_position_days':int(bt.stale_holdings.sum())}
    if 'benchmark_return' in bt and bt.benchmark_return.notna().all():
        b=bt.benchmark_return;excess=r-b
        result.update(benchmark_CAGR=float((1+b).prod()**(252/n)-1),
            arithmetic_excess_annual=float(excess.mean()*252),IR=float(excess.mean()/excess.std()*np.sqrt(252)))
    return result

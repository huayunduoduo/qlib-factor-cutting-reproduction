"""Predeclared 21/42/63 trading-day block sensitivity, no model selection."""
import json
import pandas as pd
from research.data import ROOT
from research.evaluation import paired_bootstrap
rows=[]
for folder in ['results_2015_2025','results_with_reversal','results_replication','results_stacking_2021_2025']:
    out=ROOT/folder
    if not (out/'summary.csv').exists():continue
    if folder=='results_replication':
        pairs=[(f'{c}_{f}',f'dbd_{f}') for f in ['amplitude','reversal'] for c in ['raw','gru']]+[('alpha158_lgb','dbd_two_factor')]
    elif folder=='results_stacking_2021_2025':pairs=[('A_alpha158','B_plus_two_dbd'),('A_alpha158','C_plus_residual_two_dbd')]
    else:pairs=[('E0_baseline','E1_raw_cut'),('E0_baseline','E2_residual_cut')]
    for a,b in pairs:
        ai=pd.read_csv(out/f'{a}_ic.csv',index_col=0).RankIC;bi=pd.read_csv(out/f'{b}_ic.csv',index_col=0).RankIC
        for block in [21,42,63]:rows.append({'folder':folder,'control':a,'candidate':b,**paired_bootstrap(ai,bi,block=block,rep=2000)})
pd.DataFrame(rows).to_csv(ROOT/'bootstrap_sensitivity.csv',index=False)
print('WROTE',len(rows),'paired block confidence intervals')

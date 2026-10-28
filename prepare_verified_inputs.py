"""Audit community amount versus adjusted VWAP; combine genuine Wind counts."""
import pandas as pd,numpy as np,json,hashlib
from research.data import ROOT,frozen_panel,read_bin
provider=ROOT/'data/research_2009_2025'
panel,cal=frozen_panel(provider)
counts=pd.read_csv(ROOT/'wind_normalized/trade_count.csv.gz',parse_dates=['date']).set_index(['date','instrument'])
panel=panel.join(counts[['trade_count']])
rows=[]
for year,g in panel.loc['2009-01-01':'2025-12-31'].groupby(lambda k:k[0].year):
    m=g.member;valid=m & g.close.notna() & g.volume.gt(0)
    rows.append({'year':year,'member_rows':int(m.sum()),'valid_price_rows':int(valid.sum()),'count_rows':int((valid & g.trade_count.gt(0)).sum()),'zero_count_on_traded_days':int((valid & g.trade_count.eq(0)).sum()),'null_count_on_traded_days':int((valid & g.trade_count.isna()).sum()),'amount_rows':int((valid & g.amount.gt(0)).sum()),'both_rows':int((valid & g.trade_count.gt(0) & g.amount.gt(0)).sum())})
pd.DataFrame(rows).to_csv(ROOT/'coverage_2009_2025.csv',index=False)
errors=[]
for code in panel.index.get_level_values('instrument').unique():
    g=panel.xs(code,level='instrument');vwap=read_bin(provider,code,'vwap',cal)
    ratio=(g.volume*vwap)/(g.amount*10)
    mask=g.member & g.amount.gt(0) & g.volume.gt(0) & ratio.notna()
    errors.extend(np.abs(ratio[mask]-1).tolist())
err=np.asarray(errors)
audit={'amount_to_yuan':1000,'source':'community original amount from Tushare; Wind trade_count','identity_checked_rows':len(err),'relative_error_p99':float(np.quantile(err,.99)),'relative_error_max':float(err.max()),'identity_pass':bool(np.quantile(err,.99)<1e-4),'not_a_Wind_amount_download':True,'small_active_flow_verified':False}
assert audit['identity_pass'],audit
out=ROOT/'verified_wind_join';out.mkdir(exist_ok=True)
w=panel[['amount','trade_count']].dropna().reset_index();w=w[(w.date>='2009-01-01')&(w.date<='2025-12-31')]
w.to_csv(out/'daily.csv',index=False)
contract={'reviewed':True,'price_adjustment':'none','field_config_sha256':hashlib.sha256(json.dumps(audit,sort_keys=True).encode()).hexdigest(),'trade_count_verified':True,'amount_to_yuan':1000,'small_active_flow_verified':False,'pit_exposures_verified':False,'provenance':audit}
(out/'field_contract.json').write_text(json.dumps(contract,indent=2))
(ROOT/'amount_audit.json').write_text(json.dumps(audit,indent=2))
print(audit,flush=True)

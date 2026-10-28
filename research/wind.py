"""Run on the Windows computer with a signed-in Wind terminal, never with RDP passwords."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import pandas as pd

def run():
    p=argparse.ArgumentParser();p.add_argument('--config',default='configs/wind_fields.json')
    p.add_argument('--probe',action='store_true');p.add_argument('--codes',default='configs/wind_codes.txt')
    p.add_argument('--start',default='2010-01-01');p.add_argument('--end',required=True)
    p.add_argument('--out',default='data/wind_raw');a=p.parse_args()
    cfg=json.loads(Path(a.config).read_text(encoding='utf-8'))
    config_hash=hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest()
    from WindPy import w
    if w.start().ErrorCode!=0 or not w.isconnected(): raise RuntimeError('WindPy connection failed on this machine')
    codes=[s.strip() for s in Path(a.codes).read_text().splitlines() if s.strip()]
    if a.probe:codes=codes[:1]
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    log=[];mapping=cfg['fields'];active={k:v for k,v in mapping.items() if v}
    for code in codes:
        frame=None
        for canonical,field in active.items():
            # Individual requests prevent an optional unsupported field killing all daily prices.
            result=w.wsd(code,field,a.start,a.end,'PriceAdj=N;Period=D;Days=Trading;Fill=Blank')
            row={'code':code,'canonical':canonical,'wind_field':field,'error':result.ErrorCode,
                 'options':'PriceAdj=N;Period=D;Days=Trading;Fill=Blank','requested_start':a.start,'requested_end':a.end}
            if result.ErrorCode==0 and result.Data:
                s=pd.Series(result.Data[0],index=pd.to_datetime(result.Times),name=canonical)
                row.update(observations=int(s.notna().sum()),first_valid=str(s.first_valid_index()),last_valid=str(s.last_valid_index()))
                if canonical not in ['industry']:s=pd.to_numeric(s,errors='coerce')
                frame=s.to_frame() if frame is None else frame.join(s,how='outer')
            log.append(row);time.sleep(.1)
        if frame is not None:
            frame.index.name='date';frame['instrument']=code.split('.')[1]+code.split('.')[0]
            frame=frame.reset_index();target=out/(code.replace('.','_')+'.csv')
            frame.to_csv(target,index=False)
            log.append({'file':target.name,'sha256':hashlib.sha256(target.read_bytes()).hexdigest()})
        (out/'requests.json').write_text(json.dumps(log,ensure_ascii=False,indent=2),encoding='utf-8')
    # Never auto-approve field semantics based only on successful API responses.
    contract=out/'field_contract.json'
    old=json.loads(contract.read_text()) if contract.exists() else {}
    if old.get('field_config_sha256')!=config_hash:contract.write_text(json.dumps({'reviewed':False,'price_adjustment':'none','field_config_sha256':config_hash,
        'trade_count_verified':False,'amount_to_yuan':1,'small_active_flow_verified':False,
        'pit_exposures_verified':False,'price_fields_verified':False,'adjustment_factor_verified':False,
        'historical_universe_verified':False,'volume_to_shares':1,
        'notes':'Verify field meaning, units, historical coverage and publication availability in Wind code generator.'},indent=2),encoding='utf-8')
    print('Saved probe/download receipts:',out)

if __name__=='__main__':run()

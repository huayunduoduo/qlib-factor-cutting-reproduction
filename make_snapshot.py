"""Make a frozen, smaller binary-data slice for this historical experiment.

Preserves original calendar indices and original float32 values. It only drops
instruments unused by the experiment and bars later than 2025-12-31.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
ROOT=Path(__file__).resolve().parent
SRC=ROOT/'data/cn_2025';OUT=ROOT/'data/research_2009_2025'
if __name__=='__main__':
    calendar=(SRC/'calendars/day.txt').read_text().splitlines()
    end='2025-12-31';end_idx=max(i for i,d in enumerate(calendar) if d<=end)
    rows=[]
    for line in (SRC/'instruments/csi300.txt').read_text().splitlines():
        stock,start,stop=line.split()
        if start<=end and stop>='2008-01-01':rows.append((stock,start,min(stop,end)))
    stocks=set(x[0] for x in rows)|{'SH000300'}
    for directory in ['calendars','instruments','features']:(OUT/directory).mkdir(parents=True,exist_ok=True)
    for name in ['day.txt','day_future.txt']:
        (OUT/'calendars'/name).write_text('\n'.join(calendar[:end_idx+1])+'\n')
    (OUT/'instruments/csi300.txt').write_text(''.join('\t'.join(x)+'\n' for x in rows))
    all_rows=[]
    for line in (SRC/'instruments/all.txt').read_text().splitlines():
        stock,start,stop=line.split()
        if stock in stocks and start<=end:all_rows.append((stock,start,min(stop,end)))
    (OUT/'instruments/all.txt').write_text(''.join('\t'.join(x)+'\n' for x in all_rows))
    files={};total=0
    for stock in sorted(stocks):
        directory=SRC/'features'/stock.lower()
        for path in directory.glob('*.bin'):
            raw=np.fromfile(path,dtype='<f4')
            start_idx=int(raw[0]);values=raw[:max(1,end_idx-start_idx+2)]
            target=OUT/'features'/stock.lower()/path.name;target.parent.mkdir(exist_ok=True)
            values.tofile(target);total+=target.stat().st_size
            files[str(target.relative_to(OUT))]=hashlib.sha256(target.read_bytes()).hexdigest()
    receipt={'release_tag':'2026-10-26','url':'https://github.com/chenditc/investment_data/releases/download/2026-10-26/qlib_bin.tar.gz'}
    manifest={'source':receipt,'cutoff':end,'instrument_count':len(stocks),'bytes':total,
              'file_sha256':files,'method':'truncate future bars and unused instruments; preserve calendar indices and all retained float32 bytes'}
    (OUT/'snapshot_manifest.json').write_text(json.dumps(manifest,indent=2))
    print('Frozen instruments',len(stocks),'bytes',total)

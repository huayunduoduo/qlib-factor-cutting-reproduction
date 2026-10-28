"""Validate every frozen binary data file against the saved SHA256 manifest."""
import hashlib,json
from pathlib import Path
from research.data import ROOT
p=ROOT/'data/research_2009_2025';m=json.loads((p/'snapshot_manifest.json').read_text())
for name,digest in m['file_sha256'].items():
    f=p/name
    assert hashlib.sha256(f.read_bytes()).hexdigest()==digest,name
calendar=(p/'calendars/day.txt').read_text().splitlines()
assert calendar[-1]=='2025-12-31'
assert all(a<b for a,b in zip(calendar,calendar[1:]))
for line in (p/'instruments/csi300.txt').read_text().splitlines():
    code,start,end=line.split();assert start<=end<='2025-12-31'
print('PASS',len(m['file_sha256']),'data hashes and calendar/membership cutoff')

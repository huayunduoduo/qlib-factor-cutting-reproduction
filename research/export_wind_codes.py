"""Export the historical CSI300 code union from the in-package interval file.

Reads data/frozen_cn_data/instruments/csi300.txt (qlib-style CODE<TAB>start<TAB>end
intervals) and writes configs/wind_codes_historical_csi300.txt plus a receipt.
The interval file is the package's own snapshot; verify true entry/exit dates
in Wind before any strict point-in-time backtest.
"""
import argparse
import hashlib
import json

import pandas as pd

from research.data import ROOT

BENCHMARK = '000300.SH'


def qlib_to_wind(code):
    code = code.strip().upper()
    if code.startswith('SH'):
        return code[2:] + '.SH'
    if code.startswith('SZ'):
        return code[2:] + '.SZ'
    raise ValueError('Unrecognized instrument prefix: ' + code)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--intervals', default='data/frozen_cn_data/instruments/csi300.txt')
    p.add_argument('--out', default='configs/wind_codes_historical_csi300.txt')
    p.add_argument('--receipt', default='data/wind_acquisition/code_list_receipt.json')
    a = p.parse_args()

    src = ROOT / a.intervals
    if not src.exists():
        raise SystemExit('Interval file not found: ' + str(src))
    raw = src.read_bytes()
    df = pd.read_csv(src, sep='\t', header=None, names=['instrument', 'start', 'end'])
    df['instrument'] = df['instrument'].astype(str).str.strip().str.upper()
    df = df[df['instrument'].str.len() > 0]
    codes = sorted({qlib_to_wind(c) for c in df['instrument']})
    lines = [BENCHMARK] + [c for c in codes if c != BENCHMARK]

    out = ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text('\n'.join(lines) + '\n', encoding='ascii')

    rec = ROOT / a.receipt
    rec.parent.mkdir(parents=True, exist_ok=True)
    receipt = {
        'source_file': a.intervals,
        'source_sha256': hashlib.sha256(raw).hexdigest(),
        'interval_rows': int(len(df)),
        'unique_codes': int(len(codes)),
        'benchmark': BENCHMARK,
        'min_start': str(df['start'].min()),
        'max_end': str(df['end'].max()),
        'output': a.out,
        'note': ('Union of historical CSI300 constituents from the package snapshot. '
                 'This snapshot is NOT verified point-in-time membership; confirm true '
                 'entry/exit dates in Wind before strict backtests.'),
    }
    rec.write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding='utf-8')
    print('Wrote %d codes (incl. benchmark) to %s' % (len(lines), out))
    print('Receipt:', rec)


if __name__ == '__main__':
    main()

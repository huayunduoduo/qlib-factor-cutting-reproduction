"""Probe Wind field candidates and download daily bars in small batches.

--mode probe    : try candidate WSD mnemonics for one code and write receipts
                  (field_probe.csv, probe_values.csv, probe_summary.json,
                  field_map_suggestions.json) under data/wind_acquisition/.
--mode download : batch daily bars per security. Refuses a full run unless
                  --confirm-full is given; --limit N always caps the batch.

Run only on the Windows computer with a signed-in Wind terminal.
API success never verifies field semantics or units: confirm dealnum/amt/volume
definitions in the Wind code generator before treating outputs as reviewed.
"""
import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from research.data import ROOT

BASE_FIELDS = ['open', 'high', 'low', 'close', 'volume', 'amount']
# canonical -> candidate WSD mnemonics, tried in order
CANDIDATES = {
    'open': ['open'],
    'high': ['high'],
    'low': ['low'],
    'close': ['close'],
    'volume': ['volume'],
    'amount': ['amt'],
    'trade_count': ['dealnum', 'trades', 'deal_num'],
}
WS_OPTIONS = 'PriceAdj=N;Period=D;Days=Trading;Fill=Blank'
ACQ_DIR = Path('data/wind_acquisition')


def connect():
    try:
        from WindPy import w
    except ImportError:
        raise SystemExit('WindPy is not importable here. Run on the Wind computer '
                         'with WindPy installed and the terminal signed in.')
    if w.start().ErrorCode != 0 or not w.isconnected():
        raise SystemExit('WindPy connection failed. Sign in to the Wind terminal first.')
    return w


def wsd(w, code, field, start, end):
    return w.wsd(code, field, start, end, WS_OPTIONS)


def looks_like_counts(series):
    vals = pd.to_numeric(series, errors='coerce').dropna()
    if len(vals) == 0:
        return False
    if (vals < 0).any():
        return False
    return bool(((vals % 1) == 0).all())


def run_probe(w, code, start, end, acq_dir, verbose=True):
    rows, value_rows, suggestions = [], [], {}
    for canonical, candidates in CANDIDATES.items():
        chosen = None
        for field in candidates:
            result = wsd(w, code, field, start, end)
            non_null, sample, series = 0, [], pd.Series(dtype=float)
            if result.ErrorCode == 0 and result.Data:
                series = pd.Series(result.Data[0], index=pd.to_datetime(result.Times))
                series = pd.to_numeric(series, errors='coerce')
                non_null = int(series.notna().sum())
                sample = [float(v) for v in series.dropna().head(3)]
                for ts, v in series.dropna().head(50).items():
                    value_rows.append({'date': str(ts.date()), 'code': code,
                                       'canonical': canonical, 'wind_field': field,
                                       'value': float(v)})
            count_like = looks_like_counts(series)
            rows.append({'canonical': canonical, 'wind_field': field,
                         'error_code': int(result.ErrorCode), 'non_null': non_null,
                         'sample_values': json.dumps(sample),
                         'plausible_non_negative_integer': count_like if canonical == 'trade_count' else '',
                         'note': 'API success does not verify semantics or units'})
            ok = (result.ErrorCode == 0 and non_null > 0
                  and (canonical != 'trade_count' or count_like))
            if chosen is None and ok:
                chosen = field
            time.sleep(.1)
        if chosen:
            suggestions[canonical] = chosen
        elif verbose:
            print('  no working candidate for ' + canonical)

    acq_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(acq_dir / 'field_probe.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(value_rows).to_csv(acq_dir / 'probe_values.csv', index=False, encoding='utf-8-sig')
    (acq_dir / 'field_map_suggestions.json').write_text(json.dumps({
        'suggestions': suggestions,
        'note': 'Candidates whose API call returned data. Verify meaning and units '
                'in the Wind code generator before use.'},
        ensure_ascii=False, indent=2), encoding='utf-8')
    base_ok = all(k in suggestions for k in BASE_FIELDS)
    summary = {
        'probed_code': code, 'start': start, 'end': end,
        'generated_at': datetime.now(timezone.utc).isoformat(),
        'suggested_mapping': suggestions,
        'base_fields_ok': base_ok,
        'trade_count_ok': 'trade_count' in suggestions,
        'can_attempt_ideal_reversal_factor': base_ok and 'trade_count' in suggestions,
    }
    (acq_dir / 'probe_summary.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    return summary


def load_or_probe(w, args, first_code):
    path = ROOT / ACQ_DIR / 'probe_summary.json'
    if path.exists():
        return json.loads(path.read_text(encoding='utf-8'))
    print('No probe summary found; probing %s first.' % first_code)
    end = args.end
    start = max(args.start, '2025-01-01')
    return run_probe(w, first_code, start, end, ROOT / ACQ_DIR)


def download(args):
    w = connect()
    codes = [s.strip() for s in (ROOT / args.codes).read_text(encoding='utf-8').splitlines()
             if s.strip() and not s.startswith('#')]
    if not codes:
        raise SystemExit('Empty code list: ' + args.codes)
    if args.limit:
        codes = codes[:args.limit]
    elif not args.confirm_full:
        raise SystemExit('Refusing a full download without --confirm-full. '
                         'Use --limit N for a small sample first.')

    summary = load_or_probe(w, args, codes[0])
    if not summary.get('base_fields_ok'):
        raise SystemExit('Base price/volume fields failed the probe. Fix field access '
                         'before downloading; see data/wind_acquisition/field_probe.csv.')
    if not summary.get('trade_count_ok'):
        print('WARNING: trade_count (dealnum) failed the probe. Downloading prices only; '
              'the ideal reversal factor CANNOT be reproduced from this batch.')

    mapping = {k: summary['suggested_mapping'][k] for k in BASE_FIELDS}
    if summary.get('trade_count_ok'):
        mapping['trade_count'] = summary['suggested_mapping']['trade_count']

    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    out.mkdir(parents=True, exist_ok=True)
    log = []
    for code in codes:
        frame = None
        for canonical, field in mapping.items():
            # Individual requests prevent one optional unsupported field killing all prices.
            result = wsd(w, code, field, args.start, args.end)
            row = {'code': code, 'canonical': canonical, 'wind_field': field,
                   'error': int(result.ErrorCode), 'options': WS_OPTIONS,
                   'requested_start': args.start, 'requested_end': args.end}
            if result.ErrorCode == 0 and result.Data:
                s = pd.Series(result.Data[0], index=pd.to_datetime(result.Times),
                              name=canonical)
                row.update(observations=int(s.notna().sum()),
                           first_valid=str(s.first_valid_index()),
                           last_valid=str(s.last_valid_index()))
                s = pd.to_numeric(s, errors='coerce')
                frame = s.to_frame() if frame is None else frame.join(s, how='outer')
            log.append(row)
            time.sleep(.1)
        if frame is not None:
            frame.index.name = 'date'
            frame['instrument'] = code.split('.')[1] + code.split('.')[0]
            frame = frame.reset_index()
            target = out / (code.replace('.', '_') + '.csv')
            frame.to_csv(target, index=False)
            log.append({'file': target.name,
                        'sha256': hashlib.sha256(target.read_bytes()).hexdigest()})
        print('done ' + code)
    (out / 'requests.json').write_text(
        json.dumps(log, ensure_ascii=False, indent=2), encoding='utf-8')

    # Never auto-approve field semantics based only on successful API responses.
    contract = out / 'field_contract.json'
    if not contract.exists():
        contract.write_text(json.dumps({
            'reviewed': False, 'price_adjustment': 'none',
            'trade_count_verified': False, 'amount_to_yuan': 1,
            'volume_to_shares': 1, 'adjustment_factor_verified': False,
            'historical_universe_verified': False,
            'notes': 'Verify field meaning, units, historical coverage and publication '
                     'availability in the Wind code generator before research use.'},
            indent=2), encoding='utf-8')
    print('Saved %d securities to %s' % (len(codes), out))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=['probe', 'download'], required=True)
    p.add_argument('--code', default='600000.SH')
    p.add_argument('--codes', default='configs/wind_codes_historical_csi300.txt')
    p.add_argument('--start', required=True)
    p.add_argument('--end', required=True)
    p.add_argument('--limit', type=int, default=0)
    p.add_argument('--confirm-full', action='store_true')
    p.add_argument('--out', default='data/wind_sample')
    args = p.parse_args()

    if args.mode == 'probe':
        w = connect()
        summary = run_probe(w, args.code, args.start, args.end, ROOT / ACQ_DIR)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        download(args)


if __name__ == '__main__':
    main()

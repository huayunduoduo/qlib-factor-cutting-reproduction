"""Download news for a small list of codes via w.news and map to unified columns.

If returned columns cannot be recognized, the run stops and saves an error
receipt; it never guesses column order. Requires a prior successful probe
(data/news_raw/wind_news_api_probe.json with query_succeeded=true).
Refuses a full run unless --confirm-full is given.
"""
import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from research.data import ROOT

COLUMN_ALIASES = {
    'title': ['title', 'headline', '标题', '新闻标题'],
    'content': ['content', 'body', 'text', '正文', '新闻正文', '内容'],
    'published_at': ['datetime', 'time', 'publish_time', 'pub_time', 'date',
                     '发布时间', '日期', '日期时间'],
    'source': ['source', 'src', '来源', '媒体'],
    'url': ['url', 'link', '链接', '网址'],
}


def _query(w, code, start, end):
    try:
        return w.news(code, start, end)
    except TypeError:
        return w.news(code, 'starttime=%s;endtime=%s' % (start, end))


def _map_columns(fields):
    lower = {str(f).strip().lower(): f for f in fields}
    mapping, unknown = {}, []
    for canonical, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            if alias in lower:
                mapping[lower[alias]] = canonical
                break
    for f in fields:
        if f not in mapping:
            unknown.append(str(f))
    return mapping, unknown


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--codes', default='configs/wind_codes_historical_csi300.txt')
    p.add_argument('--start', required=True)
    p.add_argument('--end', required=True)
    p.add_argument('--limit', type=int, default=0)
    p.add_argument('--confirm-full', action='store_true')
    p.add_argument('--out', default='data/news_raw')
    a = p.parse_args()

    probe_path = ROOT / 'data/news_raw/wind_news_api_probe.json'
    if probe_path.exists():
        probe = json.loads(probe_path.read_text(encoding='utf-8'))
        if not probe.get('query_succeeded'):
            raise SystemExit('News probe did not succeed; resolve API access first '
                             '(see WIND_SUPPORT_REQUEST.txt).')
    else:
        raise SystemExit('Run python -m research.wind_news_probe first.')

    try:
        from WindPy import w
    except ImportError:
        raise SystemExit('WindPy is not importable here. Run on the Wind computer.')
    if w.start().ErrorCode != 0 or not w.isconnected():
        raise SystemExit('WindPy connection failed; sign in to the Wind terminal.')

    codes = [s.strip() for s in (ROOT / a.codes).read_text(encoding='utf-8').splitlines()
             if s.strip() and not s.startswith('#')]
    if a.limit:
        codes = codes[:a.limit]
    elif not a.confirm_full:
        raise SystemExit('Refusing a full download without --confirm-full. '
                         'Use --limit N for a small sample first.')

    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    receipts = []
    for code in codes:
        r = _query(w, code, a.start, a.end)
        rec = {'code': code, 'error_code': int(r.ErrorCode),
               'requested_start': a.start, 'requested_end': a.end}
        if r.ErrorCode != 0:
            rec['error'] = 'w.news ErrorCode %d; stopped without guessing' % r.ErrorCode
            receipts.append(rec)
            _save_receipts(out, receipts)
            raise SystemExit(rec['error'])
        fields = [str(f) for f in (getattr(r, 'Fields', None) or [])]
        data = getattr(r, 'Data', None) or []
        mapping, unknown = _map_columns(fields)
        if 'title' not in mapping.values() or 'published_at' not in mapping.values():
            err = {'code': code, 'fields_seen': fields,
                   'error': 'Unrecognized column layout; stopped without guessing '
                            'column order.'}
            (out / 'wind_news_column_error.json').write_text(
                json.dumps(err, ensure_ascii=False, indent=2), encoding='utf-8')
            raise SystemExit(err['error'] + ' See data/news_raw/wind_news_column_error.json')
        df = pd.DataFrame({str(f): col for f, col in zip(fields, data)})
        df = df.rename(columns=mapping)
        df['code'] = code
        rec.update(rows=int(len(df)), columns=fields, mapped=mapping,
                   unmapped_columns=unknown)
        target = out / ('wind_news_%s.csv' % code.replace('.', '_'))
        df.to_csv(target, index=False, encoding='utf-8-sig')
        rec['file'] = target.name
        rec['sha256'] = hashlib.sha256(target.read_bytes()).hexdigest()
        receipts.append(rec)
        print('done %s (%d rows)' % (code, len(df)))
        time.sleep(.1)
    _save_receipts(out, receipts)
    print('Saved news for %d codes to %s' % (len(codes), out))


def _save_receipts(out, receipts):
    (out / 'wind_news_requests.json').write_text(json.dumps({
        'generated_at': datetime.now(timezone.utc).isoformat(),
        'requests': receipts}, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()

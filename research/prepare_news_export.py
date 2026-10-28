"""Normalize a Wind-exported news CSV into the schema research.news consumes.

Output columns: instrument, published_at, available_at, title, content, source,
url, event_id. Timestamps are emitted as ISO-8601 with an explicit offset;
naive times are assumed Asia/Shanghai and flagged.

Refuses to fabricate availability evidence: without a first-available column,
pass --assume-published-available to use published_at as a temporary proxy.
That proxy is NOT strict PIT evidence and must be disclosed in any backtest.
"""
import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

HEADER_ALIASES = {
    'instrument': ['code', 'windcode', 'wind_code', '证券代码', '代码', '关联证券'],
    'published_at': ['published_at', 'publish_time', 'pub_time', 'datetime', 'date',
                     '发布时间', '日期', '日期时间'],
    'available_at': ['available_at', 'first_available', 'ingest_time',
                     '首次可得时间', '首次入库时间', '入库时间'],
    'title': ['title', 'headline', '标题', '新闻标题'],
    'content': ['content', 'body', 'text', '正文', '新闻正文', '内容'],
    'source': ['source', 'src', '来源', '媒体'],
    'url': ['url', 'link', '链接', '网址'],
}
REQUIRED = ['instrument', 'published_at', 'title']


def _to_iso_with_offset(series):
    parsed = pd.to_datetime(series, errors='coerce', utc=False)
    raw = series.astype(str)
    has_offset = raw.str.contains(r'(?:Z|[+-]\d{2}:?\d{2})\s*$', regex=True)
    if parsed.isna().any():
        raise SystemExit('Unparseable timestamps in column; fix the export first.')
    if not has_offset.all():
        print('WARNING: naive timestamps assumed Asia/Shanghai (+08:00).')
        parsed = parsed.dt.tz_localize('Asia/Shanghai')
    else:
        parsed = parsed.dt.tz_convert('UTC').dt.tz_convert('Asia/Shanghai')
    return parsed.dt.strftime('%Y-%m-%dT%H:%M:%S%z').str.replace(
        r'(\+\d{2})(\d{2})$', r'\1:\2', regex=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('csv')
    p.add_argument('--out', default='data/news_raw/news_normalized.csv')
    p.add_argument('--assume-published-available', action='store_true')
    a = p.parse_args()

    src = Path(a.csv)
    if not src.exists():
        raise SystemExit('File not found: ' + a.csv)
    df = pd.read_csv(src, encoding='utf-8-sig')
    lower = {str(c).strip().lower(): c for c in df.columns}
    rename = {}
    for canonical, aliases in HEADER_ALIASES.items():
        for alias in aliases:
            if alias in lower:
                rename[lower[alias]] = canonical
                break
    df = df.rename(columns=rename)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise SystemExit('Missing required columns after mapping: %s. '
                         'Seen columns: %s' % (missing, list(df.columns)))

    assumed = False
    if 'available_at' not in df.columns:
        if not a.assume_published_available:
            raise SystemExit('No first-available/ingest column found. Re-export with '
                             'that field, or pass --assume-published-available to use '
                             'published_at as a temporary (disclosed) proxy.')
        df['available_at'] = df['published_at']
        assumed = True
        print('WARNING: available_at assumed equal to published_at; not strict PIT.')

    for col in ['content', 'source', 'url']:
        if col not in df.columns:
            df[col] = ''
    df['published_at'] = _to_iso_with_offset(df['published_at'])
    df['available_at'] = _to_iso_with_offset(df['available_at'])
    df['instrument'] = df['instrument'].astype(str).str.strip().str.upper()
    df['event_id'] = [hashlib.sha1((str(i) + str(t) + str(x)).encode('utf-8')).hexdigest()[:16]
                      for i, t, x in zip(df['instrument'], df['published_at'], df['title'])]

    out_cols = ['instrument', 'published_at', 'available_at', 'title', 'content',
                'source', 'url', 'event_id']
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df[out_cols].to_csv(out, index=False, encoding='utf-8')
    receipt = {'input': str(src), 'output': str(out), 'rows': int(len(df)),
               'available_at_assumed_from_published_at': assumed}
    out.with_suffix('.receipt.json').write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2), encoding='utf-8')
    print('Normalized %d rows -> %s' % (len(df), out))


if __name__ == '__main__':
    main()

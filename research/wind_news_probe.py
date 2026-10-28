"""Check whether this WindPy exposes w.news and whether a small query succeeds.

Writes data/news_raw/wind_news_api_probe.json. A negative result does NOT prove
you have no news data; it may be an API-version or account-permission issue.
Do not invent alternative interface names.
"""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from research.data import ROOT


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--code', default='600000.SH')
    p.add_argument('--start', required=True)
    p.add_argument('--end', required=True)
    p.add_argument('--out', default='data/news_raw/wind_news_api_probe.json')
    a = p.parse_args()

    report = {'code': a.code, 'start': a.start, 'end': a.end,
              'generated_at': datetime.now(timezone.utc).isoformat(),
              'windpy_importable': False, 'connected': False,
              'method_available': False, 'query_succeeded': False,
              'error_code': None, 'columns': [], 'row_count': 0,
              'error': None,
              'note': ('method_available/query_succeeded false does not prove no news '
                       'entitlement; confirm with Wind support before concluding.')}

    try:
        from WindPy import w
        report['windpy_importable'] = True
    except ImportError:
        report['error'] = 'WindPy is not importable on this machine.'
        _save(a, report)
        return
    try:
        if w.start().ErrorCode != 0 or not w.isconnected():
            report['error'] = 'WindPy connection failed; sign in to the Wind terminal.'
            _save(a, report)
            return
        report['connected'] = True
        report['method_available'] = hasattr(w, 'news')
        if report['method_available']:
            try:
                r = w.news(a.code, a.start, a.end)
            except TypeError:
                r = w.news(a.code, 'starttime=%s;endtime=%s' % (a.start, a.end))
            report['error_code'] = int(r.ErrorCode)
            if r.ErrorCode == 0:
                report['query_succeeded'] = True
                fields = getattr(r, 'Fields', None) or []
                report['columns'] = [str(f) for f in fields]
                data = getattr(r, 'Data', None) or []
                report['row_count'] = len(data[0]) if data and data[0] else 0
            else:
                report['error'] = 'w.news returned ErrorCode %d' % r.ErrorCode
    except Exception as exc:  # record, never guess another interface
        report['error'] = type(exc).__name__ + ': ' + str(exc)
    _save(a, report)


def _save(a, report):
    path = ROOT / a.out
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print('Saved:', path)


if __name__ == '__main__':
    main()

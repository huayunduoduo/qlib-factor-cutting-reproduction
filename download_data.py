"""Download the fixed community data snapshot; validate its published SHA256."""
import hashlib
import json
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TAG = '2026-10-26'
BASE = f'https://github.com/chenditc/investment_data/releases/download/{TAG}/'
ROOT.joinpath('data').mkdir(exist_ok=True)

def fetch(name):
    path = ROOT/'data'/name
    if path.exists():
        print('Already downloaded', name, flush=True)
        return path
    started = time.time()
    with urllib.request.urlopen(BASE+name, timeout=60) as response, path.with_suffix(path.suffix+'.part').open('wb') as stream:
        total = 0
        last = 0
        while True:
            chunk = response.read(1024*1024)
            if not chunk:
                break
            stream.write(chunk)
            total += len(chunk)
            if time.time()-last >= 20:
                print(f'{name}: {total/1024/1024:.1f} MiB, elapsed {time.time()-started:.0f}s', flush=True)
                last = time.time()
    path.with_suffix(path.suffix+'.part').replace(path)
    return path

if __name__ == '__main__':
    manifest_path = fetch('qlib_bin.manifest.json')
    archive_path = fetch('qlib_bin.tar.gz')
    manifest = json.loads(manifest_path.read_text())
    print('Manifest', manifest, flush=True)
    digest = hashlib.file_digest(archive_path.open('rb'), 'sha256').hexdigest()
    print('Actual SHA256', digest, flush=True)
    expected = manifest.get('archive_sha256','').removeprefix('sha256:')
    if not expected or digest != expected:
        raise ValueError('Archive digest does not match manifest')
    if archive_path.stat().st_size != manifest['archive_size_bytes']:
        raise ValueError('Archive size does not match manifest')
    (ROOT/'data'/'download_receipt.json').write_text(json.dumps({'release_tag':TAG,'url':BASE+'qlib_bin.tar.gz','sha256':digest,'bytes':archive_path.stat().st_size,'manifest':manifest},indent=2))

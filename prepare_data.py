"""Safely extract the downloaded data; do not extract links or special files."""
from pathlib import Path
import shutil
import tarfile
from download_data import ROOT

if __name__ == '__main__':
    target = ROOT/'data'/'cn_2025'
    target.mkdir(parents=True, exist_ok=True)
    count = 0
    with tarfile.open(ROOT/'data'/'qlib_bin.tar.gz', 'r:gz') as archive:
        for member in archive:
            parts = Path(member.name).parts
            if not parts or parts[0] != 'qlib_bin' or '..' in parts or Path(member.name).is_absolute():
                raise ValueError(f'Unexpected archive path: {member.name}')
            if not member.isfile():
                if not member.isdir():
                    raise ValueError(f'Unexpected non-regular file: {member.name}')
                continue
            output = target.joinpath(*parts[1:])
            output.parent.mkdir(parents=True,exist_ok=True)
            with archive.extractfile(member) as src, output.open('wb') as dst:
                shutil.copyfileobj(src,dst)
            count += 1
            if count % 20000 == 0:
                print('Extracted files',count,flush=True)
    print('Completed',count,'files at',target,flush=True)

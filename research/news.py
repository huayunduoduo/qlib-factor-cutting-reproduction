"""Point-in-time alignment of supplied text sentiment scores; does not invent news."""
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from research.data import ROOT

def align_news(news,calendar):
    required={'instrument','published_at','available_at','sentiment','event_id'}
    if not required.issubset(news):raise ValueError('News needs '+str(required))
    f=news.copy()
    for c in ['published_at','available_at']:
        if not f[c].astype(str).str.contains(r'(?:Z|[+-]\d{2}:\d{2})$',regex=True).all():
            raise ValueError(c+' must contain an explicit timezone offset')
        f[c]=pd.to_datetime(f[c],utc=True)
    f=f.sort_values('available_at').drop_duplicates(['instrument','event_id'],keep='first')
    if (f.available_at<f.published_at).any():raise ValueError('available_at cannot precede publication')
    if not f.sentiment.between(-1,1).all():raise ValueError('Sentiment must lie in [-1,1]')
    # canonical timestamps require explicit UTC offset in input. One hour processing budget.
    effective=f.available_at+pd.Timedelta(hours=1)
    cutoffs=(pd.DatetimeIndex(calendar).tz_localize('Asia/Shanghai')+pd.Timedelta(hours=15)).tz_convert('UTC')
    positions=cutoffs.searchsorted(effective)
    f=f.loc[positions<len(calendar)].copy();positions=positions[positions<len(calendar)]
    f['date']=pd.DatetimeIndex(calendar)[positions];f['instrument']=f.instrument.str.upper()
    rows=f.groupby(['date','instrument']).agg(news_sentiment=('sentiment','mean'),news_count=('event_id','size'))
    return rows

def main():
    p=argparse.ArgumentParser();p.add_argument('csv');p.add_argument('--out',default='data/news_features.csv');a=p.parse_args()
    cal=pd.DatetimeIndex(pd.read_csv(ROOT/'data/frozen_cn_data/calendars/day.txt',header=None)[0])
    result=align_news(pd.read_csv(a.csv),cal)
    path=ROOT/a.out;path.parent.mkdir(parents=True,exist_ok=True);result.to_csv(path)
    print('PIT-aligned external news scores:',path)
if __name__=='__main__':main()

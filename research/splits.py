import pandas as pd

def walk_forward(frame,start,end,train_years=5,valid_months=12,step_months=12):
    start=pd.Timestamp(start); end=pd.Timestamp(end)
    while start<=end:
        valid_start=start-pd.DateOffset(months=valid_months)
        train_start=valid_start-pd.DateOffset(years=train_years)
        test_end=min(end,start+pd.DateOffset(months=step_months)-pd.Timedelta(days=1))
        dates=frame.index.get_level_values('date'); le=frame.label_end
        # Strict maturity before the following segment starts.
        tr=(dates>=train_start)&(dates<valid_start)&(le<valid_start)&frame.label.notna()
        va=(dates>=valid_start)&(dates<start)&(le<start)&frame.label.notna()
        te=(dates>=start)&(dates<=test_end)
        yield {'test_start':start,'test_end':test_end,'valid_start':valid_start,
               'train_start':train_start,'train':tr,'valid':va,'test':te}
        start+=pd.DateOffset(months=step_months)

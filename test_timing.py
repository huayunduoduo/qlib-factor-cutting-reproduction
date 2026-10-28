import unittest
import numpy as np
import pandas as pd
from research.features import build_features
from research.splits import walk_forward
class TimingTests(unittest.TestCase):
 def test_label_has_twenty_intervals_after_next_close(self):
  dates=pd.bdate_range('2019-01-01',periods=80);c=np.arange(80)+100.
  p=pd.DataFrame({'close':c,'open':c,'high':c+1,'low':c-1,'volume':1.,'member':True,'instrument':'SH000001'},index=dates);p.index.name='date';p=p.reset_index().set_index(['date','instrument'])
  f=build_features(p,dates)
  self.assertAlmostEqual(f.iloc[30].label,c[51]/c[31]-1)
  self.assertEqual(f.iloc[30].label_end,dates[51])
  self.assertTrue(f.iloc[-21:].label.isna().all())
 def test_purge_mature_labels(self):
  dates=pd.bdate_range('2009-01-01','2015-12-31');idx=pd.MultiIndex.from_product([dates,['SH000001']],names=['date','instrument']);f=pd.DataFrame({'label':1.,'label_end':pd.Series(dates,index=dates).shift(-21).to_numpy()},index=idx)
  z=next(walk_forward(f,'2015-01-01','2015-12-31'))
  self.assertTrue((f.loc[z['train']].label_end<pd.Timestamp('2014-01-01')).all())
  self.assertTrue((f.loc[z['valid']].label_end<pd.Timestamp('2015-01-01')).all())
  self.assertEqual(z['train_start'],pd.Timestamp('2009-01-01'))
if __name__=='__main__':unittest.main()

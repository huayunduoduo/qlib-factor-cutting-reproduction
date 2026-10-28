import unittest
import numpy as np,pandas as pd
from research.features import build_features
from research.combine import symmetric_combine
from research.evaluation import portfolio
class Fixes(unittest.TestCase):
    def test_rejected_buy_does_not_select_runner_up(self):
        dates=pd.bdate_range('2020-01-01',periods=3)
        idx=pd.MultiIndex.from_product([dates,['A','B']],names=['date','instrument'])
        p=pd.DataFrame({'close':[100.,100.,110.,100.,110.,100.],'volume':100.,'member':True},index=idx)
        scores=pd.DataFrame({'score':[2.,1.]*3},index=idx)
        bt,h=portfolio(p,scores,dates,topk=1)
        self.assertEqual(bt.iloc[0].buy_turnover,0)
        self.assertEqual(bt.iloc[0].holdings,0)
        self.assertTrue(h.empty)
    def test_active_is_low_group_only(self):
        d=pd.bdate_range('2020-01-01',periods=30);x=np.arange(30)+10.
        p=pd.DataFrame({'date':d,'instrument':'A','open':x,'close':x,'high':x+1,'low':x-1,'volume':100.,'member':True,'small_buy':x,'small_sell':1.}).set_index(['date','instrument'])
        f=build_features(p,d,active_rule='close')
        self.assertAlmostEqual(f.iloc[19].cut_active,np.mean((x[:5]-1)/(x[:5]+1)))
    def test_symmetric_combine_permutation_invariant(self):
        r=np.random.default_rng(42);idx=pd.MultiIndex.from_product([[pd.Timestamp('2020-01-01')],range(100)],names=['date','instrument'])
        x=pd.DataFrame(r.normal(size=(100,3)),index=idx,columns=['a','b','c']);x['b']+=x.a
        np.testing.assert_allclose(symmetric_combine(x),symmetric_combine(x[['c','a','b']]),atol=1e-10)
if __name__=='__main__':unittest.main()

import unittest
import numpy as np
import pandas as pd
import tempfile,json,sys
from pathlib import Path
from unittest.mock import patch
from research.features import cut,build_features
from research.splits import walk_forward
from research.preprocess import Transform,residualize
from research.regime import CausalHMM
from research.evaluation import portfolio,paired_bootstrap

class ResearchTests(unittest.TestCase):
    def test_cut_exact_groups(self):
        a=np.arange(20);b=np.arange(20)/100
        self.assertAlmostEqual(cut(a,b),b[10:].sum()-b[:10].sum())
        self.assertAlmostEqual(cut(a,b,.25,'mean'),b[-5:].mean()-b[:5].mean())
        a=a.astype(float);a[0]=np.nan;self.assertTrue(np.isnan(cut(a,b)))
    def test_label_and_feature_causality(self):
        cal=pd.bdate_range('2015-01-01',periods=100)
        idx=pd.MultiIndex.from_product([cal,['SH600000']],names=['date','instrument'])
        x=pd.DataFrame({'close':np.arange(100)+100.,'high':np.arange(100)+102.,'low':np.arange(100)+99.,'open':np.arange(100)+100.,'volume':100.,'member':True},index=idx)
        f=build_features(x,cal)
        self.assertAlmostEqual(f.loc[(cal[30],'SH600000'),'label'],151/131-1)
        self.assertEqual(f.loc[(cal[30],'SH600000'),'label_end'],cal[51])
        xx=x.copy();xx.loc[(cal[60],'SH600000'),'close']*=2
        ff=build_features(xx,cal)
        pd.testing.assert_series_equal(f.loc[:cal[59],'cut_amplitude'],ff.loc[:cal[59],'cut_amplitude'])
    def test_purge(self):
        cal=pd.bdate_range('2008-01-01','2020-12-31')
        idx=pd.MultiIndex.from_product([cal,['A']],names=['date','instrument'])
        f=pd.DataFrame({'label':1.,'label_end':pd.Series(cal).shift(-21).to_numpy()},index=idx)
        for s in walk_forward(f,'2017-01-01','2020-12-31'):
            self.assertTrue((f.loc[s['train'],'label_end']<s['valid_start']).all())
            self.assertTrue((f.loc[s['valid'],'label_end']<s['test_start']).all())
    def test_training_only_scaler(self):
        idx=pd.MultiIndex.from_product([pd.bdate_range('2020-01-01',periods=10),['A','B']],names=['date','instrument'])
        x=pd.DataFrame({'a':np.arange(20,dtype=float)},index=idx)
        t=Transform().fit(x.iloc[:10]);a=t.transform(x.iloc[:10]);x.iloc[10:]=1e10
        pd.testing.assert_frame_equal(a,t.transform(x.iloc[:10]))
    def test_hmm_causal_filter(self):
        rng=np.random.default_rng(20);x=rng.normal(size=(150,3))
        h=CausalHMM(iterations=3).fit(x[:80]);a=h.filter(x)
        y=x.copy();y[100:]*=100;b=h.filter(y)
        np.testing.assert_allclose(a[:100],b[:100]);np.testing.assert_allclose(a.sum(1),1)
    def test_portfolio_cost_and_drift(self):
        cal=pd.bdate_range('2020-01-02',periods=4)
        idx=pd.MultiIndex.from_product([cal,['A']],names=['date','instrument'])
        p=pd.DataFrame({'close':[100.,100.,101.,102.],'volume':100.,'member':True},index=idx)
        pred=pd.DataFrame({'score':1.},index=idx)
        bt,_=portfolio(p,pred,cal,topk=1)
        self.assertAlmostEqual(bt.iloc[0].nav,1-.95*.0005)
        expected=.95*1.01+(1-.95-.95*.0005)
        self.assertAlmostEqual(bt.iloc[1].nav,expected)
        self.assertAlmostEqual(bt.iloc[1].buy_turnover,0)
    def test_paired_test(self):
        a=pd.Series(np.arange(100)/100);b=a+.01
        d=paired_bootstrap(a,b,rep=20)
        self.assertAlmostEqual(d['mean_delta'],.01);self.assertGreater(d['ci_low'],0)
    def test_residual(self):
        idx=pd.MultiIndex.from_product([pd.to_datetime(['2020-01-01']),list(map(str,range(40)))],names=['date','instrument'])
        x=pd.DataFrame({'x':np.arange(40)},index=idx);y=x.x*2+4
        self.assertLess(residualize(y,x).abs().max(),1e-4)
    def test_news_availability(self):
        from research.news import align_news
        cal=pd.to_datetime(['2020-01-03','2020-01-06'])
        news=pd.DataFrame({'instrument':['A','A'],'event_id':['1','2'],'sentiment':[.2,.5],
             'published_at':['2020-01-03T05:00:00Z','2020-01-03T07:01:00Z'],
             'available_at':['2020-01-03T05:00:00Z','2020-01-03T07:01:00Z']})
        result=align_news(news,cal)
        self.assertEqual(list(result.index.get_level_values('date')),list(cal))
    def test_wind_provider_adjustment_roundtrip(self):
        import import_wind_provider
        from research.data import frozen_panel
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);raw=root/'raw';raw.mkdir()
            contract={k:True for k in ['reviewed','price_fields_verified','adjustment_factor_verified','historical_universe_verified']}
            contract.update(price_adjustment='none',field_config_sha256='synthetic-test',volume_to_shares=1,amount_to_yuan=1)
            (raw/'field_contract.json').write_text(json.dumps(contract))
            dates=pd.bdate_range('2020-01-01',periods=3)
            pd.DataFrame({'date':dates,'instrument':'SH600000','open':[100,50,51],'high':[100,50,51],
              'low':[100,50,51],'close':[100,50,51],'volume':100,'amount':[10000,5000,5100],'adj_factor':[1,2,2]}).to_csv(raw/'SH600000.csv',index=False)
            pd.Series(dates.strftime('%Y-%m-%d')).to_csv(root/'calendar.txt',index=False,header=False)
            pd.DataFrame({'instrument':['SH600000'],'start':[dates[0]],'end':[dates[-1]]}).to_csv(root/'universe.csv',index=False)
            with patch.object(sys,'argv',['import_wind_provider.py','--raw',str(raw),'--calendar',str(root/'calendar.txt'),
                 '--universe',str(root/'universe.csv'),'--out',str(root/'provider')]):import_wind_provider.main()
            p,cal=frozen_panel(root/'provider')
            np.testing.assert_allclose(p.close.to_numpy(),[1,1,1.02],rtol=1e-6)
            self.assertTrue(p.member.all())
    def test_noninteger_trade_counts_fail(self):
        from research.data import load_wind
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            (root/'field_contract.json').write_text(json.dumps({'reviewed':True,'price_adjustment':'none','field_config_sha256':'synthetic-test'}))
            pd.DataFrame({'date':['2020-01-01'],'instrument':['SH600000'],'trade_count':[2.5]}).to_csv(root/'x.csv',index=False)
            with self.assertRaises(ValueError):load_wind(root,None)
    def test_mask_and_gradient(self):
        try:
            import torch
            from research.neural import masks,SequenceModel,ic_loss
        except ImportError:self.skipTest('Optional torch unavailable')
        x=torch.arange(40,dtype=torch.float32).reshape(1,20,2).repeat(30,1,1)
        h,l=masks(x);self.assertEqual(h.sum().item(),300);self.assertTrue(torch.equal(h|l,torch.ones_like(h)))
        for dual in [False,True]:
            m=SequenceModel(dual);s=m(x);self.assertEqual(tuple(s.shape),(30,));ic_loss(s,torch.randn(30)).backward()

if __name__=='__main__':unittest.main()

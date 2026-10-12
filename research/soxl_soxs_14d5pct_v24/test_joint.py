import unittest
import numpy as np
import pandas as pd
from joint_loss import *
from audit_probability import manual_probability,check_fit
from test_early import EarlyTrainingTests


class JointTests(unittest.TestCase):
    def panel(self):return paired_panel(EarlyTrainingTests().panel(210))

    def test_unknown_pair_has_no_profit_loss_class(self):
        p=EarlyTrainingTests().panel(5);p.iloc[1,p.columns.get_loc('C')]=np.nan;q=paired_panel(p)
        self.assertEqual(q.joint_class.iloc[1],-1)

    def test_future_pair_truth_cannot_change_fit(self):
        p=self.panel();at=p.index[180]
        p.loc[p.index[165:],'terminal_release']=pd.Timestamp('2030-01-01',tz='UTC')
        a,ra,_=fit_daily(p,at,[],'hgb')
        altered=p.copy();altered.loc[p.index[165:],['rL','rS','C','D','M','joint_class']]=999.
        b,rb,_=fit_daily(altered,at,[],'hgb')
        np.testing.assert_allclose(ra['raw'],rb['raw'],atol=1e-12)
        self.assertEqual(ra['fit'],rb['fit'])

    def test_calibration_uses_issued_prior_only(self):
        p=self.panel();at=p.index[200];h=[{'at':t,'raw':np.array([.1,.2,.3,.4])} for t in p.index]
        _,a,_=fit_daily(p,at,h,'logit')
        changed=[dict(e,raw=np.array([1.,0.,0.,0.])) if e['at']>=at else e for e in h]
        _,b,_=fit_daily(p,at,changed,'logit');np.testing.assert_allclose(a['adjusted'],b['adjusted'],atol=1e-12)
        self.assertEqual(a['calibration'],b['calibration'])

    def test_joint_posterior_matches_all_four_class_masses(self):
        p=self.panel();ids=list(p.index[:120]);at=p.index[150]
        p.loc[ids,'joint_class']=np.arange(120)%4
        d=scenarios(p,ids,at,[.1,.2,.3,.4],'global');self.assertTrue(d['ready'])
        q=np.array(d['posterior']);y=p.loc[ids,'joint_class'].to_numpy()
        np.testing.assert_allclose([q[y==c].sum() for c in range(4)],[.1,.2,.3,.4],atol=1e-12)

    def test_unobserved_predicted_class_forces_cash(self):
        p=self.panel();ids=list(p.index[:120]);p.loc[ids,'joint_class']=2
        d=scenarios(p,ids,p.index[150],[.1,0.,.9,0.],'global')
        self.assertFalse(d['ready']);self.assertEqual(choose(d)['action'],'CASH')

    def test_both_etfs_lose_cash_dominates_every_fraction(self):
        d={'ready':True,'reason':'ready','posterior':[.5,.5],'returns':[[-.06,-.04],[-.05,-.03]]}
        e=choose(d);self.assertEqual(e['action'],'CASH');self.assertEqual(e['allocation'],0.)

    def test_capital_fraction_preference_and_exact_ties(self):
        d={'ready':True,'reason':'ready','posterior':[.5,.5],'returns':[[.05,.05],[.05,.05]]}
        e=choose(d);self.assertEqual(e['action'],'SOXL');self.assertEqual(e['allocation'],1.)
        d['returns']=[[0.,0.],[0.,0.]];self.assertEqual(choose(d)['action'],'CASH')

    def test_manual_multiclass_and_binary_probabilities(self):
        rng=np.random.default_rng(24);X=rng.normal(size=(150,5))
        for count in [2,4]:
            y=np.arange(150)%count;w=weights(150,126)
            for kind in ['hgb','logit']:
                model,info=fit_classifier(X,y,w,kind,120)
                check_fit(model,info,X,y,126,120)
                np.testing.assert_allclose(predict(model,X[[-1]]),manual_probability(model,X[[-1]]),atol=1e-12)


if __name__=='__main__':unittest.main()

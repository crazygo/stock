"""Bounded-fit, temporal and joint-resampling contracts, without market outcomes."""
import unittest
from copy import deepcopy
from adaptive_correction import *
from scipy.optimize import lsq_linear
from test_shared import fixture


class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from hierarchical_error import fit_day as calibration_day
        cls.p,cls.h,cls.market=fixture(65);cls.at=cls.p.index[-4]
        cls.reg=registry(cls.p,cls.market)
        cls.source_day=calibration_day(cls.p,cls.h,cls.reg,cls.at)
        cls.source_day['variants']['full']=cls.source_day['variants']['partial']
        cls.records={t:{'raws':{'full':raw},'estimates':{'full':{'ready':True,'point':np.asarray(raw)+[.005,-.004],'radius':.03}}} for t,raw in cls.h.items()}
        cls.daily=fit_day(cls.p,cls.records,cls.reg,cls.at,cls.source_day)

    def arrays(self):
        rng=np.random.default_rng(163);n=80
        raw=rng.normal(0,.04,(n*4,2));cal=raw+rng.normal(0,.02,(n*4,2));truth=rng.normal(0,.05,(n*4,2))
        clocks=np.tile(np.arange(4),n);groups=np.repeat(np.arange(n),4);w=np.repeat(2.**(-np.arange(n)[::-1]/63),4)/4
        return raw,cal,truth,clocks,groups,w

    def test_inverse(self):
        x=self.arrays()[0];np.testing.assert_allclose(restore(components(x)),x,atol=1e-16)

    def test_bounded_numerical_solution(self):
        args=self.arrays();fit=blend_fit(*args,'clock');raw,cal,truth=[components(x) for x in args[:3]]
        for head in fit['heads']:
            c,k=head['clock'],head['component'];m=args[3]==c;w=args[-1][m];w/=w.sum();d=cal[m,k]-raw[m,k];y=truth[m,k]-raw[m,k]
            A=np.r_[np.sqrt(w)*d,np.sqrt(head['penalty'])][:,None];b=np.r_[np.sqrt(w)*y,np.sqrt(head['penalty'])/2]
            ref=lsq_linear(A,b,bounds=(0,1),method='bvls',tol=1e-12)
            self.assertTrue(ref.success);self.assertAlmostEqual(head['alpha'],ref.x[0],places=10)

    def test_identical_experts(self):
        a=list(self.arrays());a[1]=a[0].copy();out=blend_fit(*a,'clock');np.testing.assert_equal(out['alpha'],np.full((4,2),.5))

    def test_raw_perfect_prefers_raw(self):
        a=list(self.arrays());a[2]=a[0].copy();self.assertLess(np.max(blend_fit(*a,'clock')['alpha']),.02)

    def test_cal_perfect_prefers_cal(self):
        a=list(self.arrays());a[2]=a[1].copy();self.assertGreater(np.min(blend_fit(*a,'clock')['alpha']),.98)

    def test_shared_effective_dates(self):
        a=self.arrays();fit=blend_fit(*a,'shared');w=np.bincount(a[4],weights=a[-1]);want=w.sum()**2/(w@w)
        for h in fit['heads']:self.assertAlmostEqual(h['effective_dates'],want)

    def test_zero_weight_rows_cannot_change_fit(self):
        a=list(self.arrays());a[-1][:16]=0;out=blend_fit(*a,'clock');b=deepcopy(a)
        for x in b[:3]:x[:16]=1e30
        self.assertEqual(out,blend_fit(*b,'clock'))

    def test_scale_invariance(self):
        a=self.arrays();b=[x*10 for x in a[:3]]+list(a[3:]);np.testing.assert_allclose(blend_fit(*a,'clock')['alpha'],blend_fit(*b,'clock')['alpha'])

    def test_halfday_weight_is_one_date(self):
        parents=[{'members':list(range(n))} for n in [4,3,4]];dw,rw,g=weights(parents)
        np.testing.assert_allclose(np.bincount(g,weights=rw),dw);self.assertAlmostEqual(rw.sum(),3.)

    def test_future_and_unknown_prefix(self):
        panel,hist,market=fixture();at=panel.index[-1];reg=registry(panel,market)
        first,_=pool(reg,hist,at);changed=deepcopy(reg);changed[-1]['release']=at-pd.Timedelta(days=100)
        again,_=pool(changed,hist,at);self.assertEqual(first,again)
        changed=deepcopy(reg);changed[12]['release']=None;p,fb=pool(changed,hist,at)
        self.assertEqual(len(p),12);self.assertTrue(fb['expired_unresolved_prefix'])

    def test_missing_source_calibration_excludes_entire_date(self):
        panel,hist,market=fixture();at=panel.index[-1];reg=registry(panel,market);hist[panel.index[9]]=None
        ps,_=pool(reg,hist,at);self.assertFalse(any(panel.index[9] in p['members'] for p in ps))

    def test_union_date_counts_share_overlaps(self):
        base=list(range(30,150));meta=list(range(0,120));union=sorted(set(base+meta))
        at=pd.Timestamp('2026-01-02',tz='UTC');counts,starts,seed=block_counts(len(union),at,10)
        for count,draw in zip(counts,starts):
            days=[(int(t)+j)%len(union) for t in draw for j in range(10)][:len(union)]
            self.assertEqual(count.tolist(),np.bincount(days,minlength=len(union)).tolist())
            for d in set(base)&set(meta):self.assertEqual(count[union.index(d)],days.count(d))

    def test_radius_floor_and_directional_components(self):
        raw=np.array([.06,-.08]);cal=np.array([-.02,.01]);alpha=np.array([1.,0.]);m=mixture(raw,cal,alpha)
        self.assertAlmostEqual(components(m)[0],components(cal)[0]);self.assertAlmostEqual(components(m)[1],components(raw)[1])
        info=issue(self.p,self.at,self.records[self.at],self.daily)
        self.assertEqual(info['raw']['joint_radius'],0.)
        for v in info.values():self.assertGreaterEqual(v['radius'],.03)

    def test_entire_fit_unchanged_by_current_future_records(self):
        from run_planning import json_ready
        import json
        canonical=lambda x:json.dumps(json_ready(x),default=str,sort_keys=True)
        p=self.p.copy();p.loc[p.index>=day_cutoff(self.at),['C','D']]=999
        records=deepcopy(self.records)
        for t,r in records.items():
            if t>=day_cutoff(self.at):r['raws']['full']=np.array([999.,-999.]);r['estimates']['full']['point']=[-999.,999.]
        altered=fit_day(p,records,registry(p,self.market),self.at,self.source_day)
        self.assertEqual(canonical(self.daily),canonical(altered))

    def test_independent_full_fit_audit(self):
        from verify_correction import daily_audit,canonical
        # The synthetic expert points intentionally need not equal the current
        # calibrator: they stand for earlier as-issued models.
        _,checks=daily_audit(self.p,self.market,self.records,self.at,canonical(self.source_day),canonical(self.daily))
        self.assertEqual(checks,2400)

    def test_unready_abstains(self):
        record=deepcopy(self.records[self.at]);record['estimates']['full']['ready']=False
        estimates=issue(self.p,self.at,record,self.daily)
        for info in estimates.values():self.assertFalse(info['ready'])

    def test_meta_warmup_preserves_source_prediction(self):
        record=deepcopy(self.records[self.at]);old=record['estimates']['full']
        old['lower']=np.asarray(old['point'])-.03;old['bootstrap_predictions']=[old['point']]*199
        daily=deepcopy(self.daily);daily['ready']=False
        estimates=issue(self.p,self.at,record,daily)
        for v in estimates.values():
            self.assertTrue(v['source_fallback']);np.testing.assert_equal(v['point'],old['point']);np.testing.assert_equal(v['lower'],old['lower'])


if __name__=='__main__':unittest.main()

import unittest
import json
from pathlib import Path
import tempfile
import shutil
from types import SimpleNamespace
import numpy as np
import pandas as pd

from .acquire_pilot import normalize_sessions
from .acquire_batch import make_calendar, quality
from . import data as vd
from .prepare import audit_2026_pairs, validate_acquisition, join_2025_parts
from . import acquire_batch as batch
from . import acquire_recovery as recovery
from .evaluate import sha


def _r03_part(run, symbol, start, end):
    month=start[:7]
    part=run/'parts'/symbol/month
    part.mkdir(parents=True)
    day={'2025-08':'04','2025-09':'04','2025-10':'06',
         '2025-11':'04','2025-12':'04'}[month]
    raw=pd.DataFrame([{'time_key':f'{month}-{day} 09:35:00','open':10.,'high':10.2,
                       'low':9.9,'close':10.1,'volume':100.,'turnover':1000.}])
    frame=recovery.normalize(raw,symbol)
    frame.to_parquet(part/'bars.parquet',index=False)
    (part/'audit.json').write_text('[]\n')
    result={'symbol':symbol,'month':month,'tier':'futu','pagination_complete':True,
            'failure':None,'quality':batch.quality(frame,start,end,batch.make_calendar()),
            'bars_sha256':sha(part/'bars.parquet')}
    (part/'result.json').write_text(json.dumps(result))
    return part


def _r03_original(root, symbol='TEST'):
    run=root/'original'
    run.mkdir()
    snapshot=run/'source_snapshot'
    snapshot.mkdir()
    for name in ('acquire_batch.py','acquire_pilot.py'):
        shutil.copy2(Path(__file__).parent/name,snapshot/name)
    config={'symbols':[symbol],'months':[list(m) for m in batch.MONTHS],
            'session':'ALL','basis':'NONE','no_upload':True,
            'code_sha':sha(snapshot/'acquire_batch.py'),
            'normalizer_sha':sha(snapshot/'acquire_pilot.py')}
    (run/'registration.json').write_text(json.dumps({'config':config}))
    (run/'calendar.json').write_text(json.dumps(batch.make_calendar()))
    for start,end in batch.MONTHS:
        _r03_part(run,symbol,start,end)
    return run


def _r03_recovery(root, source, symbol='TEST'):
    run=root/'recovery'
    recovery._initialize(source,run)
    for start,end in batch.MONTHS[:-1]:
        month=start[:7]
        recovery._inherit(source/'parts'/symbol/month,
                          run/'parts'/symbol/month,symbol,month,run)
    start,end=batch.MONTHS[-1]
    part=run/'parts'/symbol/start[:7]
    part.mkdir(parents=True)
    raw=part/'raw_001.parquet'
    pd.DataFrame().to_parquet(raw,index=False)
    audit=[{'page':1,'attempt':1,'ok':True,'rows':0,'sha':sha(raw),'has_more':False}]
    recovery._new_result(part,symbol,start[:7],'futu',recovery.canonical_empty(),
                         True,None,audit,batch.make_calendar(),0.)
    return run


class DataTests(unittest.TestCase):
    def test_original_r03_parts_pass_full_month_gate(self):
        with tempfile.TemporaryDirectory() as t:
            run=_r03_original(Path(t))
            provenance,coverage,empty,gaps=validate_acquisition(run,['TEST'])
            self.assertEqual(len(coverage),5)
            self.assertFalse(empty)
            self.assertEqual({x['role'] for x in provenance} &
                             {'acquisition_registration','acquisition_calendar',
                              'acquisition_audit','acquisition_result','acquisition_bars'},
                             {'acquisition_registration','acquisition_calendar',
                              'acquisition_audit','acquisition_result','acquisition_bars'})
            self.assertEqual(len(gaps),5)  # partial market coverage remains unknown
            reg=json.loads((run/'registration.json').read_text())
            reg['config']['symbols']=[]
            (run/'registration.json').write_text(json.dumps(reg))
            with self.assertRaisesRegex(ValueError,'frozen R03 scope'):
                validate_acquisition(run,['TEST'])

    def test_recovery_inherited_and_successful_empty_are_distinct(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);source=_r03_original(root);run=_r03_recovery(root,source)
            provenance,coverage,empty,gaps=validate_acquisition(run,['TEST'])
            self.assertEqual(len(coverage),5)
            self.assertEqual(empty,[{'symbol':'TEST','month':'2025-12',
                                     'data_state':'provider_returned_no_bars_unknown_listing_state'}])
            self.assertEqual(coverage[-1]['rows'],0)
            self.assertFalse(coverage[-1]['rth_complete'])
            self.assertEqual(coverage[-1]['missing_rth_bars'],coverage[-1]['expected_rth_bars'])
            self.assertIn('inherited_record',{x['role'] for x in provenance})
            self.assertIn('acquisition_raw_page',{x['role'] for x in provenance})
            self.assertEqual(len(gaps),5)

    def test_recovery_rejects_changed_source_link_hash_and_failure(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);source=_r03_original(root);run=_r03_recovery(root,source)
            inherited=source/'parts/TEST/2025-08'
            audit=inherited/'audit.json'
            saved=audit.read_bytes()
            audit.write_text('[{"changed": true}]')
            with self.assertRaisesRegex(ValueError,'inherited source changed'):
                validate_acquisition(run,['TEST'])
            audit.write_bytes(saved)
            link=run/'parts/TEST/2025-08'
            link.unlink()
            link.symlink_to(source/'parts/TEST/2025-09',target_is_directory=True)
            with self.assertRaisesRegex(ValueError,'inherited link source changed'):
                validate_acquisition(run,['TEST'])
            link.unlink()
            link.symlink_to(inherited,target_is_directory=True)
            empty=run/'parts/TEST/2025-12'
            result=empty/'result.json'
            meta=json.loads(result.read_text())
            meta['bars_sha256']='0'*64
            result.write_text(json.dumps(meta))
            with self.assertRaisesRegex(ValueError,'changed downloaded bars'):
                validate_acquisition(run,['TEST'])
            meta['bars_sha256']=sha(empty/'bars.parquet')
            meta['pagination_complete']=False
            meta['failure']='page_cap'
            result.write_text(json.dumps(meta))
            with self.assertRaisesRegex(ValueError,'incomplete acquisition'):
                validate_acquisition(run,['TEST'])

    def test_recovery_rejects_unknown_version_missing_part_and_false_empty_claim(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);source=_r03_original(root);run=_r03_recovery(root,source)
            registration=run/'registration.json'
            saved=registration.read_bytes()
            doc=json.loads(saved)
            doc['identity']['version']='unknown_recovery'
            registration.write_text(json.dumps(doc))
            with self.assertRaisesRegex(ValueError,'unsupported acquisition recovery version'):
                validate_acquisition(run,['TEST'])
            registration.write_bytes(saved)
            missing=run/'parts/TEST/2025-10'
            missing.unlink()
            with self.assertRaisesRegex(ValueError,'missing acquisition'):
                validate_acquisition(run,['TEST'])
            missing.symlink_to(source/'parts/TEST/2025-10',target_is_directory=True)
            result=run/'parts/TEST/2025-12/result.json'
            meta=json.loads(result.read_text())
            meta['empty_success']=False
            result.write_text(json.dumps(meta))
            with self.assertRaisesRegex(ValueError,'inconsistent successful empty state'):
                validate_acquisition(run,['TEST'])

    def test_all_canonical_empty_months_join_with_2026_or_explicitly_fail(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t)
            months=[recovery.canonical_empty() for _ in batch.MONTHS]
            current=root/'2026.parquet'
            frame=recovery.normalize(pd.DataFrame([{'time_key':'2026-08-04 09:35:00',
                'open':10.,'high':10.2,'low':9.9,'close':10.1,'volume':100.,'turnover':1000.}]),'TEST')
            frame.to_parquet(current,index=False)
            joined=join_2025_parts(months,current,'TEST')
            self.assertTrue(joined.empty)
            self.assertEqual(list(joined.columns),list(recovery.CANONICAL_COLUMNS))
            bars=root/'market_data/us_5m/TEST';bars.mkdir(parents=True)
            joined.to_parquet(bars/'2025.parquet',index=False)
            shutil.copy2(current,bars/'2026.parquet')
            sessions=[s for s in batch.make_calendar()['sessions'] if s['session_date']=='2026-08-04']
            old_root=vd.ROOT
            try:
                vd.ROOT=root
                self.assertIsNotNone(vd.load_symbol('TEST',sessions,
                    {'source_dir':'market_data/us_5m','feature_start':'2025-08-01',
                     'data_end':'2026-09-24','decision_latency_seconds':30}))
            finally:
                vd.ROOT=old_root
            recovery.canonical_empty().to_parquet(current,index=False)
            with self.assertRaisesRegex(ValueError,'no observed bars in 2025 or 2026'):
                join_2025_parts(months,current,'TEST')

    def test_split_exclusion_covers_hidden_rolling_denominator(self):
        dates=[(pd.Timestamp('2025-08-01')+pd.Timedelta(days=i)).strftime('%Y-%m-%d') for i in range(150)]
        self.assertTrue(vd._split_in_input_window(dates,149,{dates[19]}))  # 130 sessions earlier
        self.assertFalse(vd._split_in_input_window(dates,149,{dates[2]}))

    def test_peer_membership_must_be_known_as_of_decision(self):
        dates=['2025-08-04','2025-08-05']
        sessions=[{'open_at':pd.Timestamp(d+' 09:30',tz='America/New_York').tz_convert('UTC').isoformat()}
                  for d in dates]
        version={'effective_from':sessions[0]['open_at'],
                 'effective_to':pd.Timestamp('2025-08-11 09:30',tz='America/New_York').isoformat(),
                 'feature_cutoff_at':sessions[0]['open_at']}
        own={('2025-W32','trend_15','A'):('trend_15:up',{'history_end':'2025-08-01'},version),
             ('2025-W32','trend_15','B'):('trend_15:up',{'history_end':'2025-08-05'},version)}
        peers={('2025-W32','trend_15:up'):{'A','B'}}
        views={'B':SimpleNamespace(cutoff_return=np.array([.02,.03]),cutoff_rvol=np.array([1.,1.]))}
        hidden=vd._group_state('A',1,dates,sessions,views,own,peers)
        self.assertEqual(hidden[0,9],0)
        own[('2025-W32','trend_15','B')]=('trend_15:up',{'history_end':'2025-08-01'},version)
        visible=vd._group_state('A',1,dates,sessions,views,own,peers)
        self.assertEqual(visible[0,9],1)
        self.assertAlmostEqual(visible[0,3],.03)

    def test_half_day_counts_only_its_actual_regular_minutes_in_label(self):
        allcal=make_calendar()
        wanted={'2025-11-26','2025-11-28','2025-12-01','2025-12-02','2025-12-03','2025-12-04'}
        sessions=[s for s in allcal['sessions'] if s['session_date'] in wanted]
        raw=np.full((len(sessions),192,6),np.nan,np.float32)
        available=np.full((len(sessions),192),np.datetime64('NaT','ns'),dtype='datetime64[ns]')
        for i,s in enumerate(sessions):
            n=s['duration_minutes']//5
            raw[i,66:66+n]=[100,101,99,100,1000,100000]
            opened=pd.Timestamp(s['open_at']).tz_localize(None)
            available[i,66:66+n]=np.array([opened.to_datetime64()+np.timedelta64((j+1)*5,'m')
                                             for j in range(n)])
        outcome=vd._outcomes(SimpleNamespace(raw5=raw,effective_available=available),sessions,0,
                             {'thresholds':[.03,.05,.08]},set())
        self.assertIsNotNone(outcome)
        self.assertEqual(pd.Timestamp(outcome[2]).tz_convert('America/New_York').strftime('%Y-%m-%d %H:%M'),
                         '2025-12-04 14:35')
        self.assertIsNone(vd._outcomes(SimpleNamespace(raw5=raw,effective_available=available),sessions,0,
                                       {'thresholds':[.03,.05,.08]},{'2025-11-28'}))

    def test_paired_2026_audit_streams_arrays_and_guards_labels(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);old=root/'old';new=root/'new';old.mkdir();new.mkdir()
            old_rows=pd.DataFrame([{'symbol':'A','session_date':'2026-08-03','sample_id':'A:v8',
                                    'entry_price':100.,'terminal_3d':.05,'available_daily_days':100}])
            new_rows=pd.concat([pd.DataFrame([{'symbol':'A','session_date':'2025-12-01','sample_id':'A:v9',
                                                'entry_price':90.,'terminal_3d':0.,'available_daily_days':20}]),
                                old_rows.assign(sample_id='A:v9',available_daily_days=126)],ignore_index=True)
            old_rows.to_parquet(old/'rows.parquet',index=False)
            new_rows.to_parquet(new/'rows.parquet',index=False)
            for path,rows in ((old,1),(new,2)):
                arrays={name:np.zeros((rows,2,3),np.float32)
                        for name in ('x5','x60','xday','y')}
                arrays['group']=np.zeros((rows,6,10),np.float32)
                arrays['group_seq']=np.zeros((rows,6,6,10),np.float32)
                if path==new:
                    arrays['xday'][1,0,1]=1
                    arrays['group'][1,5,4]=1
                np.savez_compressed(path/'features.npz',**arrays)
                (path/'manifest.json').write_text('{}')
            report=root/'audit.json'
            result=audit_2026_pairs(old,new,report)
            self.assertEqual(result['paired_rows'],1)
            self.assertEqual(result['arrays']['xday']['changed_channel_rows'],[0,1,0])
            self.assertEqual(result['arrays']['group']['changed_slot_channel_rows'][5][4],1)
            self.assertEqual(result['array_channel_labels']['group']['labels_by_slot'][5][4],
                             'qqq_cutoff_rvol')
            self.assertEqual(result['array_channel_labels']['group']['labels_by_slot'][0][4],
                             'peer_return_iqr')
            self.assertEqual(result['arrays']['y']['changed_rows'],0)
            self.assertEqual(result['fields']['available_daily_days']['changed_rows'],1)
            new_rows.loc[1,'entry_price']=101.
            new_rows.to_parquet(new/'rows.parquet',index=False)
            with self.assertRaisesRegex(ValueError,'paired labels changed'):
                audit_2026_pairs(old,new,report)

    def test_midnight_remains_overnight_and_prices_preserved(self):
        starts=pd.to_datetime(['2025-08-01 23:55','2025-08-04 09:25',
                               '2025-08-04 09:30','2025-08-04 15:55','2025-08-04 16:00']).tz_localize('America/New_York')
        f=pd.DataFrame({'start_at':starts, 'end_at':starts+pd.Timedelta(minutes=5),
                        'open':[1.,2.,3.,4.,5.], 'session_type':['incorrect']*5})
        out=normalize_sessions(f)
        self.assertEqual(out.session_type.tolist(),['overnight','pre_market','regular','regular','post_market'])
        pd.testing.assert_frame_equal(out.drop(columns='session_type'),f.drop(columns='session_type'))

    def test_calendar_holidays_half_days_and_dst(self):
        c=make_calendar();s={x['session_date']:x for x in c['sessions']}
        for d in ['2025-09-01','2025-11-27','2025-12-25','2026-07-03']:
            self.assertNotIn(d,s)
        self.assertEqual(s['2025-11-28']['duration_minutes'],210)
        self.assertEqual(s['2025-12-24']['duration_minutes'],210)
        self.assertIn('13:30:00',s['2025-10-31']['open_at'])
        self.assertIn('14:30:00',s['2025-11-03']['open_at'])
        self.assertEqual(s['2026-09-24']['duration_minutes'],390)

    def test_multiyear_read_half_day_boundary_and_future_invariance(self):
        allcal=make_calendar();wanted=['2025-11-26','2025-11-28','2025-12-01','2026-01-02']
        sessions=[s for s in allcal['sessions'] if s['session_date'] in wanted]
        frames=[]
        for s in sessions:
            d=s['session_date'];start=pd.date_range(d+' 04:00',periods=192,freq='5min',tz='America/New_York')
            close=pd.Timestamp(s['close_at']);end=start+pd.Timedelta(minutes=5)
            types=['pre_market' if x<pd.Timestamp(s['open_at']) else 'regular' if x<close else 'post_market' for x in start]
            frames.append(pd.DataFrame({'session_date':d,'start_at_et':[x.isoformat() for x in start],
                'end_at':[x.tz_convert('UTC').isoformat() for x in end],
                'available_at':[(x+pd.Timedelta(seconds=1)).tz_convert('UTC').isoformat() for x in end],
                'session_type':types,'price_basis':'NONE','open':100.,'high':101.,'low':99.,'close':100.5,'volume':1000.,'turnover':100000.}))
        frame=pd.concat(frames,ignore_index=True)
        config={'source_dir':'bars','feature_start':wanted[0],'data_end':wanted[-1],'decision_latency_seconds':30}
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);(root/'bars/A').mkdir(parents=True)
            for year in ('2025','2026'):
                frame[frame.session_date.str.startswith(year)].to_parquet(root/f'bars/A/{year}.parquet',index=False)
            oldroot=vd.ROOT
            try:
                vd.ROOT=root
                first=vd.load_symbol('A',sessions,config)
                self.assertEqual(len(first.source),2);self.assertTrue(first.daily_full.all())
                self.assertEqual(first.invalid,0)
                crossing=next(i for i,(a,b,k) in enumerate(vd.HOUR_BINS) if a<108<b)
                self.assertEqual(first.seq60[1,crossing,10],0)
                self.assertEqual(first.seq5[1,108,7],2)
                broken=frame[frame.session_date.str.startswith('2025')].copy()
                broken.loc[(broken.session_date=='2025-11-28') &
                           (broken.start_at_et.str.contains('09:30')), 'available_at']=None
                broken.to_parquet(root/'bars/A/2025.parquet',index=False)
                unavailable=vd.load_symbol('A',sessions,config)
                self.assertGreater(unavailable.invalid,0)
                self.assertFalse(unavailable.daily_full[1])
                frame[frame.session_date.str.startswith('2025')].to_parquet(root/'bars/A/2025.parquet',index=False)
                future=frame[frame.session_date.str.startswith('2026')].copy()
                future[['open','high','low','close']]*=2
                future['received_at']=future['available_at']
                future.to_parquet(root/'bars/A/2026.parquet',index=False)
                second=vd.load_symbol('A',sessions,config)
                np.testing.assert_array_equal(first.seqday[:3],second.seqday[:3])
                np.testing.assert_array_equal(first.seq5[:3],second.seq5[:3])
                np.testing.assert_array_equal(first.seq60[:3],second.seq60[:3])
                shorter=vd.load_symbol('A',sessions[:3],{**config,'data_end':'2025-12-01'})
                np.testing.assert_array_equal(first.seqday[:3],shorter.seqday)
            finally:vd.ROOT=oldroot


if __name__=='__main__':unittest.main()

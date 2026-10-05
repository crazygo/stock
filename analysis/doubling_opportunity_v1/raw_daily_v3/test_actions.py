import unittest
import numpy as np,pandas as pd
from actions import parse_html,combine_events,audited_events,unresolved_overnight_gaps
from market import labels,normalized

class ActionCoverageTests(unittest.TestCase):
    def html(self,title,body):
        return '<div id="rightWideCOL"><p>Wednesday, January 4, 2023</p><h2>Equity Corporate Actions Alert #2023-7</h2><h3>'+title+'</h3><p>'+body+'</p></div>'
    def test_explicit_effective_date_does_not_use_publication_and_ratio_direction(self):
        r=parse_html(self.html('Reverse Stock Split for Company (TEST)','Company will effect a one-for-twenty (1-20) reverse split. The split will become effective on Thursday, January 5, 2023.'),'source')
        self.assertEqual((r['published_at'],r['ex_date'],r['split_ratio']),('2023-01-04','2023-01-05',20.))
    def test_ambiguous_and_cancelled_terms_cannot_be_applied(self):
        r=parse_html(self.html('Stock Split for Company (TEST)','A (1-20) split effective January 5, 2023; a (1-10) split effective January 6, 2023.'),'source')
        self.assertEqual(r['status'],'split_terms_or_identity_unresolved')
        r=parse_html(self.html('Reverse Stock Split for Company (TEST)','The (1-20) reverse split effective January 5, 2023 is cancelled.'),'source')
        self.assertEqual(r['status'],'cancelled_or_postponed_unresolved')
    def test_primary_omission_correction_avoids_nominal_doubling_and_conflict_is_unknown(self):
        idx=pd.bdate_range('2024-01-02','2024-04-30');day='2024-01-16';c=np.where(idx<pd.Timestamp(day),10.,250.)
        raw=pd.DataFrame({'time_key':idx.strftime('%Y-%m-%d'),'open':c,'high':c*1.01,'low':c*.99,'close':c,'volume':1000.})
        record={'ex_date':day,'split_ratio':25.,'status':'verified_split','url':'actual_primary_source'}
        events,audit=combine_events(pd.DataFrame(),[record],'2024-04-30')
        _,q,_,unknown,_,_,_=normalized(raw,idx,events,'2024-04-30');lab=labels(q,idx,unknown,'2024-04-30',60)
        self.assertEqual(lab.loc['2024-01-02','y60'],0);self.assertEqual(len(unresolved_overnight_gaps(raw,idx,events,'2024-04-30')),0)
        native=pd.DataFrame([{'ex_div_date':day,'split_ratio':20.}]);events,audit=combine_events(native,[record],'2024-04-30')
        _,q,_,unknown,_,_,_=normalized(raw,idx,events,'2024-04-30');lab=labels(q,idx,unknown,'2024-04-30',60)
        self.assertTrue(pd.isna(lab.loc['2024-01-02','y60']))
    def test_true_intraday_surge_is_not_an_overnight_basis_gap(self):
        idx=pd.bdate_range('2024-01-02','2024-01-31');c=np.full(len(idx),10.);c[5]=200.
        opens=np.r_[10.,c[:-1]];raw=pd.DataFrame({'time_key':idx.strftime('%Y-%m-%d'),'open':opens,'high':np.maximum(c,opens),'low':np.minimum(c,opens),'close':c,'volume':1000.})
        self.assertEqual(unresolved_overnight_gaps(raw,idx,pd.DataFrame(),'2024-01-31'),[])
    def test_repeated_primary_articles_do_not_double_apply_a_split(self):
        r={'ex_date':'2024-01-16','split_ratio':25.,'status':'verified_split','url':'a'}
        events,audit=combine_events(pd.DataFrame(),[r,r|{'url':'b'}],'2024-04-30')
        self.assertEqual(len(events),1);self.assertEqual(len(audit),2);self.assertEqual(events.split_ratio.iloc[0],25.)
        events,audit=combine_events(pd.DataFrame(),[r,r|{'split_ratio':20.}],'2024-04-30')
        self.assertEqual(events.source_action_uncertain.iloc[0],1.)
        self.assertNotIn('split_ratio',events)
    def test_uncertain_future_gap_retains_emitted_label_as_unknown(self):
        idx=pd.bdate_range('2024-01-02','2024-04-30');c=np.full(len(idx),10.);c[10:]=25.
        raw=pd.DataFrame({'time_key':idx.strftime('%Y-%m-%d'),'open':c,'high':c,'low':c,'close':c,'volume':1000.})
        gaps=unresolved_overnight_gaps(raw,idx,pd.DataFrame(),'2024-04-30')
        self.assertEqual(len(gaps),1)
        events,_=audited_events(pd.DataFrame(),{'unknown_events':gaps},'2024-04-30')
        _,q,_,unknown,feature_unknown,_,_=normalized(raw,idx,events,'2024-04-30')
        lab=labels(q,idx,unknown,'2024-04-30',60)
        self.assertTrue(pd.isna(lab.iloc[0].y60));self.assertEqual(lab.iloc[0].label_status60,'unsupported_corporate_action')
        self.assertTrue(feature_unknown[10]);self.assertFalse(feature_unknown[0])
    def test_depositary_ratio_changes_are_not_assumed_pure_splits(self):
        r=parse_html(self.html('Reverse Stock Split and Ratio Change for Company (TEST)','A (1-20) reverse split of American Depositary Shares effective January 5, 2023.'),'source')
        self.assertEqual(r['status'],'depositary_or_ratio_change_retained_unknown')
    def test_undated_action_cannot_assume_publication_is_effective(self):
        idx=pd.bdate_range('2024-01-02','2024-04-30');c=np.full(len(idx),10.)
        raw=pd.DataFrame({'time_key':idx.strftime('%Y-%m-%d'),'open':c,'high':c,'low':c,'close':c,'volume':1000.})
        events,_=audited_events(pd.DataFrame(),{'whole_history_unknown':True},'2024-04-30')
        _,_,_,unknown,feature_unknown,_,_=normalized(raw,idx,events,'2024-04-30')
        self.assertTrue(unknown.all());self.assertTrue(feature_unknown.all())
    def test_explicit_new_symbol_table_retains_the_affected_current_security(self):
        r=parse_html(self.html('Business Combination of Company (OLD)','Business combination effective January 5, 2023. Current Symbol: OLD New Symbol: NEW Current Symbol: OLDW New Symbol: NEWW'),'source')
        self.assertEqual(r['explicit_table_symbols'],['OLD','NEW','OLDW','NEWW'])
        self.assertEqual(r['candidate_effective_dates'],['2023-01-05'])
    def test_spac_derivative_shorthand_cannot_match_unrelated_w_or_u_companies(self):
        r=parse_html(self.html('Business Combination of Company (OLD/W/U)','Current Symbol: OLD New Symbol: NEW'),'source')
        self.assertEqual(r['tickers'],['OLD'])
        r=parse_html(self.html('Merger of Company (W) and Company (U)','Details unavailable'),'source')
        self.assertEqual(r['tickers'],['W','U'])
    def test_official_looking_wrong_year_cannot_add_a_second_native_split(self):
        html=self.html('Reverse Stock Split for Company (TEST)','Company will effect a (1-20) reverse split. The split will become effective on Thursday, January 5, 2022.')
        r=parse_html(html,'source')
        self.assertEqual(r['status'],'source_effective_date_inconsistent');self.assertFalse(r['effective_date_resolved'])
        self.assertEqual({x['reason'] for x in r['date_integrity_issues']},{'explicit_weekday_date_contradiction','future_tense_action_precedes_publication'})
        self.assertNotIn('split_ratio',r);self.assertNotIn('ex_date',r)
    def test_past_year_without_weekday_is_unresolved_when_the_action_is_future_tense(self):
        r=parse_html(self.html('Reverse Stock Split for Company (TEST)','Company will effect a (1-20) reverse split effective January 5, 2022.'),'source')
        self.assertEqual(r['status'],'source_effective_date_inconsistent');self.assertFalse(r['effective_date_resolved'])
    def test_parent_spin_off_or_business_combination_is_not_a_pure_ticker_split(self):
        for title in ['Spin-off and Reverse Split information for Parent Company (TEST)','Business Combination and Reverse Split of Company (TEST)']:
            r=parse_html(self.html(title,'A (1-2) reverse split will become effective January 5, 2023.'),'source')
            self.assertEqual(r['status'],'combined_corporate_action_retained_unknown');self.assertNotIn('split_ratio',r)

if __name__=='__main__':unittest.main()

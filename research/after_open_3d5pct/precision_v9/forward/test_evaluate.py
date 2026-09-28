from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

from .evaluate import evaluate_cohort, seal_report, verify_report
from .ledger import Ledger, _at, issue_once, record_protocol_violation, register_cohort
from .test_ledger import fixture, prediction


class EvaluateTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.dates, self.spec, self.route, self.cohort, self.artifacts = fixture(self.root)
        self.now = [_at(self.dates[0], 8, 0)]
        self.ledger = Ledger(self.root / 'forward.sqlite', clock=lambda: self.now[0])
        register_cohort(self.cohort, self.ledger)

    def tearDown(self):
        self.ledger.close()
        self.temp.cleanup()

    def _buy(self, symbols):
        self.now[0] = _at(self.dates[0], 11, 30, 10)
        rows = [prediction(s, self.dates[0], self.route, self.artifacts) for s in symbols]
        return issue_once(self.spec, self.route, self.cohort, 'fixture-run', rows,
                          {'g2_eligible': True}, self.ledger)

    def _append_outcome(self, buy, status):
        fields = {k: buy[k] for k in ('route_id', 'model_version', 'cohort_id', 'run_id',
                                      'symbol', 'session_date', 'cutoff_at', 'decision_at')}
        fields['sample_id'] = buy['event_id']
        fields['event_type'] = 'outcome'
        payload = {'signal_id': buy['event_id'], 'target': '3d_5pct',
                   'supersedes': None,
                   'outcome': {'target': '3d_5pct', 'status': status,
                               'y': 1 if status == 'tp' else 0 if status == 'fp' else None}}
        with self.ledger.transaction():
            return self.ledger.append(fields, payload)

    def test_all_five_buy_denominator_unknown_range_and_full_sessions(self):
        buys = self._buy(['AVGO', 'AMD', 'NVDA', 'INTC', 'QCOM'])
        for buy, status in zip(buys, ['tp', 'tp', 'fp', 'pending', 'missing']):
            self._append_outcome(buy, status)
        self.now[0] = _at(self.dates[61], 17, 0)
        report = evaluate_cohort(self.cohort['cohort_id'], self.ledger.verify_chain(),
                                 self.cohort, ledger=self.ledger)
        cell = report['cells']['B_group:chips']['main']
        self.assertEqual(report['official_sessions'], 60)
        self.assertEqual((cell['issued'], cell['tp'], cell['fp'], cell['unresolved']),
                         (5, 2, 1, 2))
        self.assertAlmostEqual(cell['known_precision'], 2 / 3)
        self.assertEqual(cell['possible_precision'], [.4, .8])
        self.assertEqual(cell['unresolved_by_reason'], {'pending': 1, 'missing': 1})
        self.assertAlmostEqual(cell['signals_per_5_sessions'], 5 / 12)
        self.assertFalse(cell['pass'])
        self.assertEqual(report['cells']['B_group:optics']['main']['issued'], 0)
        self.assertIsNone(report['cells']['B_group:optics']['main']['known_precision'])
        self.assertEqual(len(report['cells']), 15)
        self.assertGreater(report['bootstrap']['5']['cells']['B_group:chips']['zero_signal_draws'], 0)
        self.assertFalse(report['final_target_pass'])

    def test_one_perfect_buy_cannot_pass_minimum_or_fixture_gate(self):
        buy = self._buy(['AMD'])[0]
        self._append_outcome(buy, 'tp')
        self.now[0] = _at(self.dates[61], 17, 0)
        cell = evaluate_cohort(self.cohort['cohort_id'], self.ledger.verify_chain(),
                               self.cohort, ledger=self.ledger)['cells']['B_group:chips']['main']
        self.assertEqual(cell['known_precision'], 1.)
        self.assertFalse(cell['gate']['minimum_mature_signals'])
        self.assertFalse(cell['gate']['minimum_signal_dates'])
        self.assertFalse(cell['gate']['weekly_supply'])
        self.assertFalse(cell['gate']['live_evidence'])
        self.assertFalse(cell['pass'])

    def test_period_not_ended_and_model_bytes_tamper_fail(self):
        buy = self._buy(['AMD'])[0]
        self._append_outcome(buy, 'tp')
        early = evaluate_cohort(self.cohort['cohort_id'], self.ledger.verify_chain(),
                                self.cohort, ledger=self.ledger)
        self.assertFalse(early['period_ended'])
        self.assertFalse(early['cells']['B_group:chips']['main']['gate']['period_ended'])
        (self.root / 'model.bin').write_bytes(b'tampered')
        with self.assertRaisesRegex(ValueError, 'source or model bytes changed'):
            evaluate_cohort(self.cohort['cohort_id'], self.ledger.verify_chain(),
                            self.cohort, ledger=self.ledger)

    def test_preregistration_and_hash_chain_tamper_rejected(self):
        self._buy(['AMD'])
        wrong = dict(self.cohort, session_dates=self.cohort['session_dates'][1:])
        with self.assertRaisesRegex(ValueError, 'preregistration differs'):
            evaluate_cohort(self.cohort['cohort_id'], self.ledger.verify_chain(),
                            wrong, ledger=self.ledger)
        with self.assertRaisesRegex(Exception, 'append-only'):
            self.ledger.db.execute("UPDATE events SET symbol='XXX' WHERE event_type='buy'")

    def test_future_ledger_cut_rejected(self):
        self._buy(['AMD'])
        with self.assertRaisesRegex(ValueError, 'ledger cut is beyond'):
            evaluate_cohort(self.cohort['cohort_id'], self.ledger.verify_chain() + 1,
                            self.cohort, ledger=self.ledger)

    def test_wrapper_cannot_hide_outer_route_contract_change(self):
        wrapper = deepcopy(self.cohort)
        wrapper['registration'] = deepcopy(self.cohort)
        wrapper['route_contracts']['B_group']['thresholds']['chips'] = .01
        with self.assertRaisesRegex(ValueError, 'registration wrapper is forbidden'):
            register_cohort(wrapper, self.ledger)
        self.now[0] = _at(self.dates[0], 11, 30, 10)
        with self.assertRaisesRegex(ValueError, 'registration wrapper is forbidden'):
            issue_once(self.spec, self.route, wrapper, 'wrapped',
                       [prediction('AMD', self.dates[0], self.route, self.artifacts)],
                       {'g2_eligible': True}, self.ledger)
        with self.assertRaisesRegex(ValueError, 'registration wrapper is forbidden'):
            evaluate_cohort(self.cohort['cohort_id'], self.ledger.verify_chain(),
                            wrapper, ledger=self.ledger)

    def test_sealed_report_detects_byte_damage(self):
        self._buy(['AMD'])
        report = evaluate_cohort(self.cohort['cohort_id'], self.ledger.verify_chain(),
                                 self.cohort, ledger=self.ledger)
        path = self.root / 'report.json'
        seal_report(report, path, ledger=self.ledger)
        self.assertEqual(verify_report(path, ledger=self.ledger)['cohort_id'], self.cohort['cohort_id'])
        with self.assertRaises(FileExistsError):
            seal_report(report, path, ledger=self.ledger)
        path.write_text('{}')
        with self.assertRaisesRegex(ValueError, 'changed after seal'):
            verify_report(path, ledger=self.ledger)

    def test_discovered_violation_keeps_buy_in_denominator(self):
        buy = self._buy(['AMD'])[0]
        self._append_outcome(buy, 'tp')
        record_protocol_violation(buy['event_id'], 'receipt_audit_failed', self.ledger)
        report = evaluate_cohort(self.cohort['cohort_id'], self.ledger.verify_chain(),
                                 self.cohort, ledger=self.ledger)
        cell = report['cells']['B_group:chips']['main']
        self.assertEqual(cell['issued'], 1)
        self.assertEqual(cell['forward_violations'], 1)
        self.assertFalse(cell['gate']['no_forward_violation'])


if __name__ == '__main__':
    unittest.main()

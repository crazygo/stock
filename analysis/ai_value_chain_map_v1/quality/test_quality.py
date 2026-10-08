import importlib.util
import json
import sys
import tempfile
import unittest
from datetime import date
from math import isclose
from pathlib import Path

HERE = Path(__file__).resolve().parent

def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

pf = load("prepare_financials_tested", "prepare_financials.py")
rate = load("rate_tested", "rate.py")


def fact(start, end, value, tag="Revenue", unit="USD", filed="2026-08-01", priority=0):
    return {"field": "revenue", "taxonomy": "us-gaap", "tag": tag, "tag_priority": priority, "unit": unit,
            "value": value, "start": start, "end": end, "filed": filed, "form": "10-Q", "accession": filed,
            "duration_days": (date.fromisoformat(end)-date.fromisoformat(start)).days+1,
            "source_url": "https://data.sec.gov/test", "cache_path": "test", "cache_sha256": "x", "received_at": "2026-10-06"}


class FinancialExtractionTests(unittest.TestCase):
    def test_ttm_window_starts_after_comparable_ytd_and_is_one_year(self):
        rows = [fact("2024-01-01", "2024-12-31", 1000, filed="2025-02-01"),
                fact("2025-01-01", "2025-06-30", 600),
                fact("2024-01-01", "2024-06-30", 400, filed="2024-08-01")]
        _, ttm = pf.ytd_ttm(rows)
        self.assertEqual(ttm["start"], "2024-07-01")
        self.assertEqual(ttm["end"], "2025-06-30")
        self.assertEqual(ttm["duration_days"], 365)
        self.assertEqual(ttm["value"], 1200)

    def test_ttm_yoy_uses_comparable_prior_ttm_or_null(self):
        rows = [fact("2024-01-01", "2024-12-31", 1000, filed="2025-02-01"),
                fact("2025-01-01", "2025-06-30", 600), fact("2024-01-01", "2024-06-30", 400, filed="2024-08-01"),
                fact("2023-01-01", "2023-12-31", 800, filed="2024-02-01"), fact("2023-01-01", "2023-06-30", 300, filed="2023-08-01")]
        _, current = pf.ytd_ttm(rows)
        prior = pf.prior_year_ttm(rows, current)
        self.assertEqual(prior[0], 900)
        self.assertTrue(isclose(current["value"] / prior[0] - 1, 1/3))
        _, unmatched = pf.ytd_ttm(rows[:3])
        self.assertIsNone(pf.prior_year_ttm(rows[:3], unmatched))

    def test_diluted_shares_are_duration_weighted(self):
        rows = [fact("2024-01-01", "2024-12-31", 100, tag="WeightedAverageNumberOfDilutedSharesOutstanding", unit="shares", filed="2025-02-01"),
                fact("2025-01-01", "2025-06-30", 120, tag="WeightedAverageNumberOfDilutedSharesOutstanding", unit="shares"),
                fact("2024-01-01", "2024-06-30", 80, tag="WeightedAverageNumberOfDilutedSharesOutstanding", unit="shares", filed="2024-08-01")]
        for r in rows:
            r["field"] = "diluted_shares"
        _, ttm = pf.ytd_ttm(rows)
        self.assertTrue(ttm["formula"].startswith("weighted_average_shares:"))
        denom = rows[0]["duration_days"] + rows[1]["duration_days"] - rows[2]["duration_days"]
        expected = (100*rows[0]["duration_days"] + 120*rows[1]["duration_days"] - 80*rows[2]["duration_days"])/denom
        self.assertAlmostEqual(ttm["value"], expected)

    def test_unmatched_interim_and_restricted_cash_and_debt_coverage(self):
        rows = [fact("2025-01-01", "2025-06-30", 600)]
        self.assertIsNone(pf.ytd_ttm(rows)[1])
        cash_tags = pf.TAGS["cash"]["us-gaap"]
        self.assertNotIn("CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents", cash_tags)
        def debt(tag, value, priority=0):
            return {"field":"__debt", "tag":tag,"value":value,"unit":"USD","end":"2026-06-30","filed":"2026-08-01","taxonomy":"us-gaap","accession":"a","form":"10-Q","tag_priority":priority,"start":None}
        partial = pf.debt_value([debt("LongTermDebtCurrent", 10), debt("LongTermDebtNoncurrent", 50)])
        self.assertEqual(partial["formula"], "partial_long_term_debt_current_plus_noncurrent")
        self.assertEqual(partial["coverage"], "partial_long_term_components")
        compact=rate._compact_financials({"metrics":{"total_debt":{"latest":partial}},"derived":{}})
        self.assertEqual(compact["metrics"]["total_debt"]["coverage"],"partial_long_term_components")
        total = pf.debt_value([debt("ShortTermBorrowingsAndCurrentPortionOfLongTermDebt", 12, 1), debt("LongTermDebtNoncurrent", 50)])
        self.assertEqual(total["formula"], "current_debt_component + noncurrent_debt_component")
        self.assertEqual(total["value"], 62)
        self.assertEqual(total["coverage"],"total_current_and_noncurrent")


class RatingTests(unittest.TestCase):
    def good_review(self, observed="2026-10-06"):
        return {"review_status":"official_review","observed_at":observed,"stage":"scaling","materiality":3,"commercial":3,"moat":2,
                "risk_reviewed":True,"risk_flags":[],"reasons":["evidence"],
                "sources":[{"url":"https://issuer.example/ir","title":"IR","observed_at":observed,"published_at":"2026-08-01","evidence":"specific product revenue"}],
                "risk_evidence":[{"domain":d,"risk":"reviewed risk","evidence":"checked","source_url":"https://issuer.example/ir"} for d in ("customer","competition","funding")]}

    def finance(self, capture="2026-10-06"):
        return {"as_of":"2026-10-07","records":{"US.TEST":{"status":"available","source_capture_at":capture,"latest_period_end":"2026-06-30","latest_filed":"2026-08-01",
            "metrics":{"revenue":{"ttm":{"value":100,"unit":"USD","start":"2025-07-01","end":"2026-06-30","filed":"2026-08-01"}},"net_income":{"ttm":{"value":10,"unit":"USD","start":"2025-07-01","end":"2026-06-30","filed":"2026-08-01"}},
                       "operating_cashflow":{"ttm":{"value":12,"unit":"USD","start":"2025-07-01","end":"2026-06-30","filed":"2026-08-01"}},"capex":{"ttm":{"value":2,"unit":"USD","start":"2025-07-01","end":"2026-06-30","filed":"2026-08-01"}}},
            "derived":{"revenue_yoy":{"value":0.2,"unit":"ratio","period_end":"2026-06-30"}},"sources":[],"caveats":[]}}}

    def test_future_business_and_future_financial_records_abstain(self):
        future = self.good_review("2026-10-08")
        row = {"code":"US.TEST","industry":"Technology"}
        result = rate.evaluate_stock(row, future, self.finance(), date(2026,10,7))
        self.assertNotIn(result["grade"], ("A+", "A"))
        self.assertEqual(result["decision"], "pending")
        bad_finance = self.finance()
        bad_finance["records"]["US.TEST"]["latest_filed"] = "2026-10-08"
        self.assertEqual(rate.finance_assessment(bad_finance,"US.TEST",date(2026,10,7))["support"],"unknown")
        stale=self.finance(capture="2026-09-01")
        self.assertEqual(rate.finance_assessment(stale,"US.TEST",date(2026,10,7))["support"],"unknown")
        future_fact=self.finance()
        future_fact["records"]["US.TEST"]["metrics"]["net_income"]["ttm"]["filed"]="2026-10-08"
        self.assertEqual(rate.finance_assessment(future_fact,"US.TEST",date(2026,10,7))["support"],"unknown")

    def test_missing_capex_and_unmatched_period_do_not_claim_strong(self):
        f = self.finance()
        f["records"]["US.TEST"]["metrics"]["capex"]["ttm"] = None
        a = rate.finance_assessment(f,"US.TEST",date(2026,10,7))
        self.assertEqual(a["support"],"sound")
        self.assertIn("资本开支缺失或不可比；FCF及FCF率保持未知",a["missing"])
        f["records"]["US.TEST"]["metrics"]["operating_cashflow"]["ttm"]["end"]="2026-03-31"
        self.assertEqual(rate.finance_assessment(f,"US.TEST",date(2026,10,7))["support"],"unknown")

    def test_mismatched_capex_does_not_invalidate_aligned_core_financials(self):
        f=self.finance()
        capex=f["records"]["US.TEST"]["metrics"]["capex"]["ttm"]
        capex.update({"start":"2025-01-01","end":"2025-12-31"})
        assessment=rate.finance_assessment(f,"US.TEST",date(2026,10,7))
        self.assertEqual(assessment["support"],"sound")
        self.assertIsNone(assessment["periods"]["fcf"])
        self.assertIn("资本开支事实期间或币种与核心期间不匹配，本轮排除该项，FCF保持未知；原始期间仍保留供查看",assessment["missing"])

    def test_strong_grade_requires_complete_review_and_no_unexplained_severe_risk(self):
        row={"code":"US.TEST","industry":"Technology"}
        result=rate.evaluate_stock(row,self.good_review(),self.finance(),date(2026,10,7))
        self.assertEqual(result["grade"],"A+")
        risky=self.good_review(); risky["risk_flags"]=["bankruptcy risk"]
        result=rate.evaluate_stock(row,risky,self.finance(),date(2026,10,7))
        self.assertNotIn(result["grade"], ("A+", "A"))
        self.assertTrue(any("严重风险" in m for m in result["missing"]))
        no_risk_review=self.good_review(); no_risk_review["risk_reviewed"]=False
        self.assertIsNone(rate.evaluate_stock(row,no_risk_review,self.finance(),date(2026,10,7))["grade"])

    def test_a_minus_is_not_a_fallback_for_missing_a_risk_evidence_and_b_protocol(self):
        row={"code":"US.TEST","industry":"Technology"}
        mature=self.good_review(); mature["risk_evidence"]=[]
        result=rate.evaluate_stock(row,mature,self.finance(),date(2026,10,7))
        self.assertIsNone(result["grade"])
        developing_business=self.good_review(); developing_business["commercial"]=1
        developing_business["risk_evidence"]=developing_business["risk_evidence"][:1]
        self.assertEqual(rate.evaluate_stock(row,developing_business,self.finance(),date(2026,10,7))["grade"],"A-")
        low_materiality=self.good_review(); low_materiality.update({"materiality":1,"commercial":1,"moat":0,"risk_reviewed":False})
        self.assertEqual(rate.evaluate_stock(row,low_materiality,self.finance(),date(2026,10,7))["grade"],"B")

    def test_committed_funding_can_prevent_a_false_strained_b(self):
        row={"code":"US.TEST","industry":"Technology"}
        review=self.good_review(); review.update({"commercial":1,"materiality":2,"moat":1})
        f=self.finance(); rec=f["records"]["US.TEST"]
        rec["metrics"]["net_income"]["ttm"]["value"]=-10
        rec["metrics"]["operating_cashflow"]["ttm"]["value"]=-12
        rec["metrics"]["cash"]={"latest":{"value":2,"unit":"USD","end":"2026-06-30","filed":"2026-08-01"}}
        rec["derived"]["cash_runway_months"]={"value":2,"unit":"months","period_end":"2026-06-30"}
        self.assertEqual(rate.finance_assessment(f,"US.TEST",date(2026,10,7))["support"],"strained")
        review["funding_protection"]={"source_type":"committed_facility","committed":True,"undrawn_amount_value":15,"amount_unit":"USD","source_url":"https://issuer.example/ir","evidence":"committed and available","available_through":"2027-10-08","unrestricted_for_operations":True}
        result=rate.evaluate_stock(row,review,f,date(2026,10,7))
        self.assertEqual(result["financial_support"],"developing")
        review["funding_protection"]={"source_type":"cash_raise","amount_value":15,"amount_unit":"USD","source_url":"https://issuer.example/ir","evidence":"received funds","received_at":"2026-06-15","unrestricted_for_operations":True}
        result=rate.evaluate_stock(row,review,f,date(2026,10,7))
        self.assertEqual(result["financial_support"],"strained")
        self.assertEqual(result["grade"],None)
        self.assertTrue(any("可流动投资" in x for x in result["missing"]))

    def test_strained_b_requires_documented_liquid_assets_and_financing_review(self):
        row={"code":"US.TEST","industry":"Technology"}
        review=self.good_review(); review.update({"materiality":2,"commercial":1,"moat":1})
        review["risk_evidence"]=[{"domain":"funding","risk":"liquidity","evidence":"liquid securities and latest financing reviewed; still insufficient","source_url":"https://issuer.example/ir","liquid_assets_reviewed":True,"latest_financing_reviewed":True,"funding_capacity":"insufficient"}]
        f=self.finance(); rec=f["records"]["US.TEST"]
        rec["metrics"]["net_income"]["ttm"]["value"]=-10
        rec["metrics"]["operating_cashflow"]["ttm"]["value"]=-12
        rec["metrics"]["cash"]={"latest":{"value":2,"unit":"USD","end":"2026-06-30","filed":"2026-08-01"}}
        rec["derived"]["cash_runway_months"]={"value":2,"unit":"months","period_end":"2026-06-30"}
        result=rate.evaluate_stock(row,review,f,date(2026,10,7))
        self.assertEqual(result["grade"],"B")

    def test_first_run_replay_upgrade_downgrade_and_exit_history(self):
        records={"US.X":{"grade":"A","previous_grade":None,"change":None}}
        self.assertEqual(rate.attach_history(records,None,"r1"),[])
        old={"run_id":"r1","rule_version":rate.RULE_VERSION,"records":{"US.X":{"grade":"A","previous_grade":None,"change":None},"US.GONE":{"grade":"B"}},"changes":[]}
        same={"US.X":{"grade":"A","previous_grade":None,"change":None}}
        self.assertEqual(rate.attach_history(same,old,"r1"),[])
        up={"US.X":{"grade":"A+"}}
        changes=rate.attach_history(up,old,"r2")
        self.assertEqual(up["US.X"]["previous_grade"],"A")
        self.assertEqual(up["US.X"]["change"],"升级")
        self.assertEqual([x["change"] for x in changes],["升级","退出"])
        expanded={"US.X":{"grade":"A"},"US.NEW":{"grade":None}}
        changes=rate.attach_history(expanded,old,"r4")
        self.assertEqual(expanded["US.NEW"]["change"],"新增")
        self.assertTrue(any(x["change"]=="新增" for x in changes))
        down={"US.X":{"grade":None}}
        changes=rate.attach_history(down,old,"r3")
        self.assertEqual(down["US.X"]["change"],"转待评级")
        old["summary"]={"input_hashes":{"rate_code_sha256":"old-code","protocol_sha256":"old-protocol"}}
        revised={"US.X":{"grade":"A+"}}
        changes=rate.attach_history(revised,old,"r5",{"rate_code_sha256":"new-code","protocol_sha256":"old-protocol"})
        self.assertEqual(revised["US.X"]["change"],"规则版本变化（不可解读为基本面变化）")
        self.assertIn("不可解读",changes[0]["reason"])

    def test_finance_sector_does_not_use_industrial_cash_thresholds(self):
        self.assertEqual(rate.finance_assessment(self.finance(),"US.TEST",date(2026,10,7),"Regional Bank")["support"],"unknown")

    def test_preview_does_not_modify_formal_current(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); current=root/"current.json"; preview=root/"preview.json"
            current.write_text('{"run_id":"formal"}\n',encoding="utf-8")
            before=current.read_bytes()
            rate.write_preview(preview,{"run_id":"preview","records":{},"changes":[]},current)
            self.assertEqual(current.read_bytes(),before)
            self.assertTrue(json.loads(preview.read_text(encoding="utf-8"))["preview"])
            self.assertTrue(preview.with_suffix(".csv").exists())
            with self.assertRaises(ValueError):
                rate.write_preview(current,{"run_id":"bad"},current)

    def test_explicit_review_priority_and_verified_same_issuer_class_sharing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            old=root/"reviews_core.json"; new=root/"reviews_core_top.json"
            incomplete={"code":"US.GOOGL","review_status":"incomplete","observed_at":"2026-10-06","stage":"scaling","risk_reviewed":False,"sources":[]}
            complete=self.good_review(); complete.update({"code":"US.GOOG","review_priority":2})
            old.write_text(json.dumps({"records":[incomplete]}),encoding="utf-8")
            new.write_text(json.dumps({"records":[complete]}),encoding="utf-8")
            picked,_=rate.load_reviews([old,new],date(2026,10,7),{"US.GOOG","US.GOOGL"})
            self.assertEqual(picked["US.GOOG"]["review_status"],"official_review")
            pool=root/"data.json"; fin=root/"financial_snapshot.json"
            pool.write_text(json.dumps({"rows":[{"kind":"STOCK","code":"US.GOOG","business":{"tier":1}},{"kind":"STOCK","code":"US.GOOGL","business":{"tier":1}}]}),encoding="utf-8")
            fin.write_text(json.dumps({"as_of":"2026-10-07","records":{c:{"sources":[{"cik":1652044}] } for c in ("US.GOOG","US.GOOGL")}}),encoding="utf-8")
            saved=(rate.HERE,rate.POOL_PATH,rate.FINANCIALS_PATH)
            try:
                rate.HERE=root; rate.POOL_PATH=pool; rate.FINANCIALS_PATH=fin
                result=rate.build(date(2026,10,7))
            finally:
                rate.HERE,rate.POOL_PATH,rate.FINANCIALS_PATH=saved
            self.assertTrue(result["records"]["US.GOOGL"]["issuer_review_shared"])
            self.assertEqual(result["records"]["US.GOOGL"]["inherited_from"],"US.GOOG")

    def test_run_id_ignores_generated_pool_quality_and_snapshot_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); pool=root/"data.json"; fin=root/"financial.json"; review=root/"reviews_core_top.json"
            base={"rows":[{"kind":"STOCK","code":"US.X","name":"X","industry":"Tech","business":{"tier":1}}],"quality":{"generated":"old"}}
            pool.write_text(json.dumps(base),encoding="utf-8")
            fin.write_text(json.dumps({"as_of":"2026-10-07","records":{"US.X":{"metrics":{}}},"validation":{"checked":1}}),encoding="utf-8")
            review.write_text(json.dumps({"records":[]}),encoding="utf-8")
            first=rate.compute_run_id(date(2026,10,7),pool,[review],fin)[0]
            base["quality"]={"generated":"new"}; pool.write_text(json.dumps(base),encoding="utf-8")
            newer=json.loads(fin.read_text()); newer["validation"]={"checked":999}; fin.write_text(json.dumps(newer),encoding="utf-8")
            second=rate.compute_run_id(date(2026,10,7),pool,[review],fin)[0]
            self.assertEqual(first,second)


if __name__ == "__main__":
    unittest.main()

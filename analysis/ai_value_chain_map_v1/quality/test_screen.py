import importlib.util
import json
import tempfile
import unittest
from datetime import date
from pathlib import Path


SCRIPT = Path(__file__).with_name("screen.py")
spec = importlib.util.spec_from_file_location("screen_under_test", SCRIPT)
screen = importlib.util.module_from_spec(spec)
spec.loader.exec_module(screen)


class ScreenRulesTest(unittest.TestCase):
    def test_role_mapping_requires_known_value_path(self):
        self.assertEqual(screen._value_role("compute", ["compute"], 2, ["dedicated accelerator product"]), "core")
        self.assertEqual(screen._value_role("power", ["power"], 2, ["general power supply"]), "enabler")
        self.assertEqual(screen._value_role("energy", ["energy"], 2, ["general utility"]), "indirect")
        self.assertEqual(screen._value_role("energy", ["energy"], 2, ["AI数据中心供电 infrastructure"]), "enabler")
        self.assertEqual(screen._value_role(None, [], 0, []), "unknown")

    def test_low_confidence_caps_unverified_grade(self):
        grade, _ = screen._initial_grade(3, "core", {}, {}, ["specific mechanism"], "strong", "low", date(2026, 10, 7))
        self.assertEqual(grade, "A-")

    def test_early_tier3_is_observation_priority(self):
        grade, _ = screen._initial_grade(3, "core", {}, {}, ["商业化尚未兑现"], "unknown", "high", date(2026, 10, 7))
        self.assertEqual(grade, "A-")

    def test_sources_need_actual_recent_observation(self):
        cutoff = date(2026, 10, 7)
        base = {"url": "https://issuer.example/product", "status": "luna_official_review", "observed_at": "2026-10-06"}
        self.assertTrue(screen._approved_source(base, cutoff)[0])
        self.assertFalse(screen._approved_source({**base, "observed_at": "2026-10-08"}, cutoff)[0])
        self.assertFalse(screen._approved_source({**base, "observed_at": "2026-06-01"}, cutoff)[0])
        self.assertFalse(screen._approved_source({**base, "observed_at": None}, cutoff)[0])

    def test_semantic_hashes_ignore_screen_and_price_fields_but_track_membership(self):
        row = {"code": "US.TEST", "kind": "STOCK", "name": "Example", "industry": "IT", "business": {"tier": 2, "primary": "compute", "groups": ["compute"], "reasons": ["specific path"]}, "price": {"close": 1}, "research_screen": {"grade": "A"}, "quality": {"grade": "B"}}
        first = screen._pool_projection([row])
        row["price"]["close"] = 99
        row["research_screen"]["grade"] = "B"
        row["quality"]["grade"] = "A"
        self.assertEqual(screen.canonical_hash(first), screen.canonical_hash(screen._pool_projection([row])))
        row["business"]["tier"] = 3
        self.assertNotEqual(screen.canonical_hash(first), screen.canonical_hash(screen._pool_projection([row])))

    def test_formal_grade_requires_same_date_non_preview_current(self):
        as_of = date(2026, 10, 7)
        row = {"grade": "A", "decision": "current", "expires_at": "2026-10-20"}
        self.assertEqual(screen._verified_grade(row, {"preview": False, "as_of": "2026-10-01"}, as_of), "A")
        self.assertIsNone(screen._verified_grade(row, {"preview": True, "as_of": "2026-10-07"}, as_of))
        self.assertIsNone(screen._verified_grade(row, {"preview": False, "as_of": "2026-10-08"}, as_of))
        self.assertIsNone(screen._verified_grade({**row, "expires_at": "2026-10-06"}, {"preview": False, "as_of": "2026-10-07"}, as_of))

    def test_quality_hash_tracks_all_screen_consumed_grade_fields(self):
        q1 = {"as_of": "2026-10-07", "records": {"US.X": {"grade": "A", "review_status": "reviewed", "business_reviewed_at": "2026-10-06", "stage": "mature", "commercial": 3}}}
        q2 = {"as_of": "2026-10-07", "records": {"US.X": {"grade": "A", "review_status": "incomplete", "business_reviewed_at": "2026-10-06", "stage": "precommercial", "commercial": 1}}}
        self.assertNotEqual(screen.canonical_hash(screen._quality_projection(q1, ["US.X"])), screen.canonical_hash(screen._quality_projection(q2, ["US.X"])))

    def test_incomplete_legacy_commercial_placeholder_is_not_early_evidence(self):
        cutoff = date(2026, 10, 7)
        reasons = ["HBM 直接用于 AI 加速计算；DRAM/NAND 还有其他需求。"]
        mu_like = {"review_status": "incomplete", "business_reviewed_at": "2026-10-07", "stage": "mature", "commercial": 1}
        self.assertFalse(screen._early_or_unproven({}, mu_like, reasons, cutoff))
        self.assertFalse(screen._early_or_unproven({}, {**mu_like, "review_status": "reviewed", "business_reviewed_at": "2026-06-01"}, reasons, cutoff))
        self.assertTrue(screen._early_or_unproven({}, {"review_status": "reviewed", "business_reviewed_at": "2026-10-06", "stage": "precommercial", "commercial": 1}, reasons, cutoff))
        self.assertTrue(screen._early_or_unproven({}, mu_like, ["光子集成平台面向 AI 光网络；产品关联不证明规模化收入。"], cutoff))

    def test_verified_record_has_consistent_status_and_uses_effective_financial_freshness(self):
        row = {"code": "US.X", "kind": "STOCK", "name": "Example", "industry": "IT", "business": {"tier": 2, "primary": "compute", "groups": ["compute"], "reasons": ["Official dedicated AI accelerator product evidence."], "review_status": "official_review", "confidence": "high", "sources": [{"url": "https://issuer.example/product", "title": "Product", "observed_at": "2026-10-06", "status": "luna_official_review", "evidence": "Dedicated accelerator product.", "date_basis": "page reviewed"}]}}
        quality = {"as_of": "2026-10-01", "preview": False, "records": {"US.X": {"grade": "A", "decision": "current", "expires_at": "2026-10-20", "financial_support": "strong", "stage": "mature", "commercial": 3, "business_reviewed_at": "2026-10-01"}}}
        pool = {"rows": [row]}
        financials = {"as_of": "2026-10-07", "records": {"US.X": {"status": "complete", "source_capture_at": "2026-09-01", "latest_period_end": "2026-06-30", "latest_filed": "2026-08-01", "metrics": {}, "derived": {}}}}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name, obj in (("pool.json", pool), ("quality.json", quality), ("finance.json", financials)):
                (root / name).write_text(json.dumps(obj), encoding="utf-8")
            built = screen.build(date(2026, 10, 7), root / "pool.json", root / "quality.json", root / "finance.json")
        rec = built["records"]["US.X"]
        self.assertFalse(built["preview"])
        self.assertEqual(built["rating_type"], "research_priority")
        self.assertEqual(rec["status"], "verified")
        self.assertEqual(rec["formal_quality_as_of"], "2026-10-01")
        self.assertFalse(any("尚无同截止日有效" in x for x in rec["limitations"]))
        self.assertFalse(any("进入正式quality评级" in x for x in rec["next_actions"]))
        self.assertEqual(rec["financial_support"], "unknown")
        self.assertEqual(rec["historical_financial_support"], "strong")
        self.assertEqual(rec["business_sources"][0]["evidence"], "Dedicated accelerator product.")
        self.assertEqual(rec["business_sources"][0]["date_basis"], "page reviewed")


if __name__ == "__main__":
    unittest.main()

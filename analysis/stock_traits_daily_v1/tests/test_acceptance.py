"""Acceptance tests for stock_traits_daily_v1 (skill 验收 items), using controlled fixtures only.

Run (repo root, any python with pandas+pyarrow):  python3 -m unittest discover -s analysis/stock_traits_daily_v1/tests -v
"""
from __future__ import annotations

import json
import math
import os
import re
import subprocess
import sys
import time
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fixtures import CAL, Env, FakeOpenD, FakeR2, bars_5m, closes_for, universe_file, write_5m  # noqa: E402
import backfill  # noqa: E402
import refresh  # noqa: E402
import render  # noqa: E402
import traits as T  # noqa: E402
from common import REPO, read_json, read_jsonl, sha256_file  # noqa: E402
from sources import OpenDLayer  # noqa: E402

NOW_1007 = "2026-10-07T08:00:00+00:00"   # 16:00 Asia/Shanghai; ET still 10-07 04:00 -> cutoff 2026-10-06
NO_R2 = lambda: FakeR2()  # noqa: E731
NO_OPEND = FakeOpenD(ok=False)


def sessions(end: str, n: int) -> list[str]:
    return CAL.sessions_through(end, n)


def ds_of(manifest: dict, sid: str) -> dict:
    return next(d for d in manifest["datasets"] if d.get("security_id") == sid and d.get("kind") == "market_daily")


class BackfillAcceptance(unittest.TestCase):
    def setUp(self):
        self.env = Env()
        universe_file(self.env.universe, ["AAA"])

    def tearDown(self):
        self.env.cleanup()

    def run_bf(self, now=NOW_1007, extra=None, r2=None, opend=None):
        return backfill.run(self.env.backfill_args(now, extra), r2_factory=r2 or NO_R2, opend_runner=opend or NO_OPEND)

    def test_b1_internal_gap_detected_even_with_head_tail_present(self):
        days = sessions("2026-10-06", 161)
        gap = ["2026-08-10", "2026-08-11", "2026-08-12"]
        c = closes_for("AAA", [d for d in days if d not in gap])
        write_5m(self.env.local, "AAA", bars_5m("AAA", c))
        m = self.run_bf()["manifest"]
        d = ds_of(m, "US.AAA")
        self.assertEqual(d["actual_range"]["first"], days[0])        # head present
        self.assertEqual(d["actual_range"]["last"], days[-1])        # tail present
        self.assertEqual(d["status"], "missing_intervals")
        self.assertEqual([(i["start"], i["end"], i["sessions"]) for i in d["missing_intervals"]],
                         [("2026-08-10", "2026-08-12", 3)])
        self.assertEqual(d["complete_through"], "2026-08-07")

    def test_b2_holiday_not_reported_missing(self):
        self.assertFalse(CAL.is_session("2026-09-07"))  # Labor Day in official calendar
        days = sessions("2026-10-06", 161)
        c = closes_for("AAA", days)
        write_5m(self.env.local, "AAA", bars_5m("AAA", c, extra_days={"2026-09-07": 1.0}))  # stray bars on holiday
        d = ds_of(self.run_bf()["manifest"], "US.AAA")
        self.assertEqual(d["status"], "complete")
        self.assertEqual(d["missing_intervals"], [])
        self.assertEqual(d["complete_through"], "2026-10-06")
        self.assertIn({"session_date": "2026-09-07", "reason": "bars_on_non_session_day"}, d["anomalies"]["excluded_days"])

    def test_b3_stale_r2_hit_keeps_falling_back(self):
        days = sessions("2026-10-06", 161)
        c = closes_for("AAA", days)
        old = [d for d in days if d <= "2026-09-30"]
        write_5m(self.env.local, "AAA", bars_5m("AAA", {d: c[d] for d in old}))
        # Case A: R2 object last written 2026-09-27 -> cannot contain 10-01..10-06
        r2a = FakeR2({"us_5m/AAA/2026.parquet": (self.env.local / "AAA" / "2026.parquet", "Sun, 27 Sep 2026 10:04:16 GMT")})
        op = FakeOpenD(ok=True, kline={"US.AAA": c})
        d = ds_of(self.run_bf(r2=lambda: r2a, opend=op)["manifest"], "US.AAA")
        r2_layer = next(l for l in d["layers"] if l["layer"] == "r2")
        self.assertEqual(r2_layer["state"], "stale_by_last_modified")
        self.assertNotIn(("get", "us_5m/AAA/2026.parquet"), r2a.calls)
        self.assertTrue(any(a[0] == "kline" for a in op.calls))
        self.assertEqual(d["status"], "complete")
        self.assertEqual(d["complete_through"], "2026-10-06")
        # Case B (fresh env): R2 recently modified but content still stale -> downloaded, still_stale, OpenD tried
        env2 = Env(); universe_file(env2.universe, ["AAA"])
        try:
            write_5m(env2.local, "AAA", bars_5m("AAA", {d: c[d] for d in old if d <= "2026-09-25"}))
            stale_copy = env2.tmp / "r2obj.parquet"
            bars_5m("AAA", {d: c[d] for d in old}).to_parquet(stale_copy, index=False)
            r2b = FakeR2({"us_5m/AAA/2026.parquet": (stale_copy, "Wed, 07 Oct 2026 01:00:00 GMT")})
            op2 = FakeOpenD(ok=False)
            m = backfill.run(env2.backfill_args(NOW_1007), r2_factory=lambda: r2b, opend_runner=op2)["manifest"]
            d2 = ds_of(m, "US.AAA")
            r2l = next(l for l in d2["layers"] if l["layer"] == "r2")
            self.assertEqual(r2l["state"], "downloaded")
            self.assertTrue(r2l["still_stale"])
            opl = next(l for l in d2["layers"] if l["layer"] == "opend")
            self.assertEqual(opl["state"], "unavailable")
            self.assertEqual(d2["status"], "stale")
            self.assertEqual(d2["complete_through"], "2026-09-30")
        finally:
            env2.cleanup()

    def test_b4_budget_interrupt_and_failure_keep_complete_old_version(self):
        days = sessions("2026-10-06", 161)
        c = closes_for("AAA", days)
        write_5m(self.env.local, "AAA", bars_5m("AAA", c))
        r1 = self.run_bf()
        d1 = ds_of(r1["manifest"], "US.AAA")
        old_file = Path(d1["path"]) if Path(d1["path"]).is_absolute() else REPO / d1["path"]
        h_old = sha256_file(old_file)
        cur_before = read_json(self.env.data / "current.json")
        # change source, then run with zero budget -> nothing processed, old version carried forward
        c2 = dict(c); c2["2026-06-15"] = c2["2026-06-15"] * 1.1
        write_5m(self.env.local, "AAA", bars_5m("AAA", c2))
        r2 = self.run_bf(extra=["--budget-seconds", "0"])
        d2 = ds_of(r2["manifest"], "US.AAA")
        self.assertEqual(d2["change_type"], "carried_forward")
        self.assertEqual(d2["carried_reason"], "budget_exhausted")
        self.assertEqual(d2["version_id"], d1["version_id"])
        self.assertEqual(sha256_file(old_file), h_old)
        # a failing publish must not move current.json
        cur_mid = read_json(self.env.data / "current.json")
        orig = backfill.validate_manifest
        backfill.validate_manifest = lambda p: (_ for _ in ()).throw(RuntimeError("fixture: validation failure"))
        try:
            with self.assertRaises(RuntimeError):
                self.run_bf()
        finally:
            backfill.validate_manifest = orig
        self.assertEqual(read_json(self.env.data / "current.json"), cur_mid)
        self.assertEqual(sha256_file(old_file), h_old)
        self.assertEqual(read_jsonl(self.env.data / "ledger.jsonl")[-1]["status"], "failed")
        self.assertNotEqual(cur_before["data_run_id"], cur_mid["data_run_id"])

    def test_b5_revision_creates_new_version_old_kept(self):
        days = sessions("2026-10-06", 161)
        c = closes_for("AAA", days)
        write_5m(self.env.local, "AAA", bars_5m("AAA", c))
        d1 = ds_of(self.run_bf()["manifest"], "US.AAA")
        f1 = Path(d1["path"]); h1 = sha256_file(f1)
        c2 = dict(c); c2["2026-06-15"] = round(c2["2026-06-15"] * 1.05, 4)
        write_5m(self.env.local, "AAA", bars_5m("AAA", c2))
        m2 = self.run_bf()["manifest"]
        d2 = ds_of(m2, "US.AAA")
        self.assertNotEqual(d2["version_id"], d1["version_id"])
        self.assertEqual(d2["change_type"], "revised")
        self.assertIn("2026-06-15", d2["revised_sessions"])
        self.assertEqual(d2["supersedes"], d1["version_id"])
        self.assertTrue(f1.exists()); self.assertEqual(sha256_file(f1), h1)
        self.assertTrue(any(ch["change_type"] == "revised" for ch in m2["changes"]))

    def test_b6_repeat_is_idempotent(self):
        days = sessions("2026-10-06", 161)
        write_5m(self.env.local, "AAA", bars_5m("AAA", closes_for("AAA", days)))
        d1 = ds_of(self.run_bf()["manifest"], "US.AAA")
        caps_before = sorted(p.name for p in self.env.cap.iterdir())
        d2 = ds_of(self.run_bf(now="2026-10-07T09:00:00+00:00")["manifest"], "US.AAA")
        self.assertEqual(d2["change_type"], "unchanged")
        self.assertEqual(d2["version_id"], d1["version_id"])
        self.assertEqual(d2["received_at"], d1["received_at"])  # no new evidence time
        self.assertEqual(sorted(p.name for p in self.env.cap.iterdir()), caps_before)  # no new capture

    def test_b7_legacy_inputs_unchanged(self):
        days = sessions("2026-10-06", 161)
        p = write_5m(self.env.local, "AAA", bars_5m("AAA", closes_for("AAA", days)))
        h = sha256_file(p)
        m = self.run_bf()["manifest"]
        self.assertEqual(sha256_file(p), h)
        self.assertTrue(m["legacy_inputs"]["unchanged_after_run"])
        self.assertIn(h, m["legacy_inputs"]["sha256"].values())

    def test_b8_no_future_data(self):
        days = sessions("2026-10-06", 161)
        c = closes_for("AAA", days)
        write_5m(self.env.local, "AAA", bars_5m("AAA", c, extra_days={"2026-10-07": 99.0, "2026-10-08": 98.0}))
        m = self.run_bf()["manifest"]
        d = ds_of(m, "US.AAA")
        self.assertEqual(m["cutoff_session"], "2026-10-06")
        self.assertEqual(d["actual_range"]["last"], "2026-10-06")
        df = pd.read_parquet(d["path"])
        self.assertLessEqual(df["session_date"].max(), "2026-10-06")
        self.assertTrue((pd.to_datetime(df["available_at"], utc=True) <= pd.Timestamp(NOW_1007)).all())

    def test_b9_raw_data_and_credentials_not_in_git(self):
        for p in ["market_data/stock_data_v1/captures/cap-x/daily/US.AAA.parquet", ".cache/stock_data_v1/runs/x/manifest.json",
                  ".cache/stock_report_v1/snapshots/x/snapshot.json", "market_data/stock_data_v1/captures/cap-x/opend_5m/US.A.parquet"]:
            r = subprocess.run(["git", "-C", str(REPO), "check-ignore", "-q", p])
            self.assertEqual(r.returncode, 0, f"not ignored: {p}")
        tracked = subprocess.run(["git", "-C", str(REPO), "ls-files", "analysis/stock_traits_daily_v1"],
                                 capture_output=True, text=True).stdout.split()
        self.assertFalse([t for t in tracked if t.endswith((".parquet", ".json.gz"))])
        cfg = json.loads((REPO / "config" / "r2_storage.json").read_text())
        secrets_ = [cfg.get("secret_access_key"), cfg.get("access_key_id")]
        pkg = Path(__file__).resolve().parents[1]
        for f in pkg.rglob("*"):
            if f.is_file() and f.suffix in (".py", ".md", ".html", ".js", ".json"):
                txt = f.read_text(encoding="utf-8", errors="ignore")
                self.assertFalse(any(s and s in txt for s in secrets_), f"credential string found in {f.name}")

    def test_b10_opend_worker_is_killable(self):
        lay = OpenDLayer("127.0.0.1", 1, timeout=1.0)
        lay.worker_cmd = [sys.executable, "-c", "import time; time.sleep(30)"]
        t = time.monotonic()
        res = lay.probe()
        self.assertLess(time.monotonic() - t, 5)
        self.assertEqual(res["error_type"], "timeout")

    def test_b11_worker_result_survives_futu_log_noise(self):
        # futu-api prints its own log lines on stdout (e.g. on_disconnect after close); parse the marked result line
        lay = OpenDLayer("127.0.0.1", 1, timeout=5.0)
        code = ("print('2026-10-07 | x | [open_context_base.py] connect'); "
                "print('@@STDV1_RESULT@@ ' + '{\"ok\": true, \"state\": {\"qot_logined\": true}}'); "
                "print('2026-10-07 | x | on_disconnect: Disconnected: conn=0(1) reason=CallClose')")
        lay.worker_cmd = [sys.executable, "-c", code]
        self.assertTrue(lay.probe()["ok"])


class RefreshAcceptance(unittest.TestCase):
    def setUp(self):
        self.env = Env()
        universe_file(self.env.universe, ["AAA", "BBB"])
        self.all_days = sessions("2026-10-06", 170)
        self.cA = closes_for("AAA", self.all_days)
        self.cB = closes_for("BBB", self.all_days[-80:])  # short history: 80 closes

    def tearDown(self):
        self.env.cleanup()

    def publish(self, through: str, now: str, cA=None):
        cA = cA or self.cA
        write_5m(self.env.local, "AAA", bars_5m("AAA", {d: v for d, v in cA.items() if d <= through}))
        write_5m(self.env.local, "BBB", bars_5m("BBB", {d: v for d, v in self.cB.items() if d <= through}))
        return backfill.run(self.env.backfill_args(now), r2_factory=NO_R2, opend_runner=NO_OPEND)

    def refresh(self, now, extra=None):
        return refresh.run(self.env.refresh_args(now, extra))

    def snap(self, res):
        return read_json(Path(res["snapshot_path"]) if Path(res["snapshot_path"]).is_absolute()
                         else self.env.report / res["snapshot_path"])

    def res_of(self, snap, sid):
        return next(r for r in snap["results"] if r["security_id"] == sid)

    def test_r1_new_session_advances_window(self):
        self.publish("2026-09-30", "2026-10-01T08:00:00+00:00")
        s1 = self.snap(self.refresh("2026-10-01T08:30:00+00:00"))
        self.publish("2026-10-01", "2026-10-02T08:00:00+00:00")
        s2 = self.snap(self.refresh("2026-10-02T08:30:00+00:00"))
        a1, a2 = self.res_of(s1, "US.AAA"), self.res_of(s2, "US.AAA")
        self.assertEqual(a1["price"]["price_asof"], "2026-09-30")
        self.assertEqual(a2["price"]["price_asof"], "2026-10-01")
        i = CAL.dates.index
        self.assertEqual(i(a2["window_first_session"]) - i(a1["window_first_session"]), 1)
        self.assertNotEqual(s1["idempotency_key"], s2["idempotency_key"])
        page = render.build_data(self.env.report, "t")
        self.assertEqual(len(next(s for s in page["securities"] if s["id"] == "US.AAA")["points"]), 2)

    def test_r2_post_cutoff_data_excluded(self):
        self.publish("2026-09-30", "2026-10-01T08:00:00+00:00")
        s1 = self.snap(self.refresh("2026-10-01T08:30:00+00:00"))
        env_b = Env(); universe_file(env_b.universe, ["AAA", "BBB"])
        try:
            self.env, keep = env_b, self.env
            self.publish("2026-10-06", NOW_1007)
            s2 = self.snap(self.refresh(NOW_1007, ["--evaluation-cutoff", "2026-09-30"]))
            self.env = keep
            a1, a2 = self.res_of(s1, "US.AAA"), self.res_of(s2, "US.AAA")
            self.assertEqual(a2["price"]["price_asof"], "2026-09-30")
            for t in T.TRAITS:
                self.assertAlmostEqual(a1["traits"][t]["value"], a2["traits"][t]["value"], places=12)
        finally:
            env_b.cleanup()

    def test_r3_missing_traits_kept_independently(self):
        self.publish("2026-10-06", NOW_1007)
        s = self.snap(self.refresh(NOW_1007))
        b = self.res_of(s, "US.BBB")
        self.assertEqual(b["traits"]["G"]["status"], "unknown")
        self.assertEqual(b["traits"]["S"]["status"], "unknown")
        for t in ("V", "M", "J"):
            self.assertEqual(b["traits"][t]["status"], "ok")
        self.assertIsNone(b["map"])
        self.assertEqual(b["status"], "partial")
        hk = self.res_of(s, "HK.00001")
        self.assertEqual(hk["status"], "unknown")
        page = render.build_data(self.env.report, "t")
        self.assertIn("US.BBB", [x["id"] for x in page["securities"] if not x["map"]])

    def test_r4_rerun_does_not_duplicate_points(self):
        self.publish("2026-10-06", NOW_1007)
        r1 = self.refresh(NOW_1007)
        r2 = self.refresh("2026-10-07T08:10:00+00:00")
        self.assertTrue(r2["reused"])
        self.assertEqual(r1["report_run_id"], r2["report_run_id"])
        snaps = [p for p in (self.env.report / "snapshots").iterdir() if not p.name.startswith(".")]
        self.assertEqual(len(snaps), 1)
        led = read_jsonl(self.env.report / "ledger.jsonl")
        self.assertEqual([r["status"] for r in led if r["kind"] == "refresh"], ["computed", "reused"])
        page = render.build_data(self.env.report, "t")
        self.assertEqual(len(next(s for s in page["securities"] if s["id"] == "US.AAA")["points"]), 1)

    def test_r5_revision_new_version_old_snapshot_unchanged(self):
        self.publish("2026-10-06", NOW_1007)
        r1 = self.refresh(NOW_1007)
        p1 = Path(r1["snapshot_path"]); h1 = sha256_file(p1)
        cA2 = dict(self.cA); cA2["2026-09-15"] = round(cA2["2026-09-15"] * 1.07, 4)
        self.publish("2026-10-06", "2026-10-07T08:20:00+00:00", cA=cA2)
        r2 = self.refresh("2026-10-07T08:30:00+00:00")
        s2 = self.snap(r2)
        self.assertFalse(r2["reused"])
        self.assertEqual(s2["supersedes"], r1["report_run_id"])
        self.assertEqual(sha256_file(p1), h1)
        self.assertFalse(os.access(p1, os.W_OK))
        page = render.build_data(self.env.report, "t")
        a = next(s for s in page["securities"] if s["id"] == "US.AAA")
        self.assertEqual(len(a["points"]), 1)               # one point per (mode, cutoff): latest version
        self.assertEqual(a["points"][0]["report_run_id"], r2["report_run_id"])
        self.assertEqual(a["superseded_versions"][0]["report_run_id"], r1["report_run_id"])

    def test_r6_pure_render_adds_no_points(self):
        self.publish("2026-10-06", NOW_1007)
        self.refresh(NOW_1007)
        a = render.render(self.env.report)
        b = render.render(self.env.report)
        self.assertEqual(a["points"], b["points"])
        self.assertEqual(a["snapshots"], b["snapshots"])
        self.assertNotEqual(a["render_id"], b["render_id"])
        refreshes = [r for r in read_jsonl(self.env.report / "ledger.jsonl") if r["kind"] == "refresh"]
        self.assertEqual(len(refreshes), 1)
        self.assertTrue((self.env.report / "current").is_symlink())
        self.assertTrue(str(os.readlink(self.env.report / "current")).endswith(b["render_id"]))

    def test_r7_sparse_runs_s_uses_daily_series(self):
        self.publish("2026-09-01", "2026-09-02T08:00:00+00:00")
        self.refresh("2026-09-02T08:30:00+00:00")
        self.publish("2026-10-06", NOW_1007)          # 24 sessions later, no runs in between
        s = self.snap(self.refresh(NOW_1007))
        a = self.res_of(s, "US.AAA")
        closes = np.array([self.cA[d] for d in sessions("2026-10-06", 141)])
        self.assertAlmostEqual(a["traits"]["S"]["value"], T.s_reference(closes), places=12)
        led = [r for r in read_jsonl(self.env.report / "ledger.jsonl") if r["kind"] == "refresh"]
        self.assertEqual(len(led), 2)  # only 2 executions, yet S used 21 daily points

    def test_r8_ranges_only_crop_view(self):
        self.publish("2026-10-06", NOW_1007)
        r = self.refresh(NOW_1007)
        html = Path(r["render"]["page"]).read_text(encoding="utf-8")
        self.assertIn("[14, 30, 60]", html)
        js = Path(__file__).resolve().parents[1] / "view_logic.js"
        pts = [{"date": d, "traits": {"G": {"display": 50}}} for d in
               ["2026-08-01", "2026-08-08", "2026-09-07", "2026-09-22", "2026-09-23", "2026-09-24", "2026-10-06"]]
        code = ("const L=require(%r);const p=%s;const o={};[14,30,60].forEach(n=>o[n]=L.cropPoints(p,'2026-10-06',n).map(x=>x.date));"
                "o.seg=L.segments(L.cropPoints(p,'2026-10-06',60),'G').map(s=>s.gapDays);console.log(JSON.stringify(o));") % (str(js), json.dumps(pts))
        out = json.loads(subprocess.run(["node", "-e", code], capture_output=True, text=True, check=True).stdout)
        self.assertEqual(out["14"], ["2026-09-23", "2026-09-24", "2026-10-06"])  # 14 days incl. cutoff: 09-23..10-06
        self.assertEqual(out["30"], ["2026-09-07", "2026-09-22", "2026-09-23", "2026-09-24", "2026-10-06"])
        self.assertEqual(out["60"], ["2026-08-08", "2026-09-07", "2026-09-22", "2026-09-23", "2026-09-24", "2026-10-06"])
        self.assertEqual(out["seg"], [30, 15, 1, 1, 12])
        s = self.snap(r)
        self.assertEqual(s["model"]["G"]["window_returns"], 120)  # computation windows untouched by view range

    def test_r9_news_valuation_unconnected_stay_unknown(self):
        self.publish("2026-10-06", NOW_1007)
        r = self.refresh(NOW_1007)
        s = self.snap(r)
        for x in s["results"]:
            self.assertEqual(x["news"]["status"], "not_connected")
            self.assertIsNone(x["news"]["value"])
            self.assertEqual(x["valuation"]["status"], "not_connected")
            self.assertIsNone(x["composite_score"])
        html = Path(r["render"]["page"]).read_text(encoding="utf-8")
        self.assertIn("新闻 未接入", html)
        self.assertIn("无综合分数", html)


class FormulaChecks(unittest.TestCase):
    def test_formulas_against_literal_definitions(self):
        rng = np.random.default_rng(7)
        P = 30 * np.exp(np.cumsum(rng.normal(0, 0.03, 160)))
        r = np.diff(np.log(P))
        v = T.compute(r)
        self.assertAlmostEqual(v["G"][0], r[-120:].mean() * 252, places=12)
        self.assertAlmostEqual(v["V"][0], r[-60:].std(ddof=1) * math.sqrt(252), places=12)
        self.assertAlmostEqual(v["M"][0], np.corrcoef(r[-60:][:-1], r[-60:][1:])[0, 1], places=12)
        e = (r[-60:] - r[-60:].mean()) ** 2
        self.assertAlmostEqual(v["J"][0], np.sort(e)[-6:].sum() / e.sum(), places=12)
        self.assertAlmostEqual(v["S"][0], T.s_reference(P[-141:]), places=12)

    def test_zero_variance_is_unknown_not_zero(self):
        v = T.compute(np.zeros(150))
        self.assertIsNone(v["V"][0]); self.assertIsNone(v["M"][0]); self.assertIsNone(v["J"][0]); self.assertIsNone(v["S"][0])
        self.assertEqual(v["G"][0], 0.0)

    def test_bootstrap_reproducible_seed(self):
        r = np.random.default_rng(3).normal(0, 0.02, 140)
        a = T.bootstrap(r, "US.AAA", "2026-10-06", reps=32)
        b = T.bootstrap(r, "US.AAA", "2026-10-06", reps=32)
        self.assertEqual(a, b)
        import hashlib
        self.assertEqual(T.seed_for("US.AAA", "2026-10-06", 256),
                         int.from_bytes(hashlib.sha256(b"five_traits_v2|US.AAA|2026-10-06|256").digest()[:8], "little"))

    def test_gap_in_window_marks_unknown(self):
        days = sessions("2026-10-06", 150)
        c = closes_for("AAA", days)
        del c[days[-30]]
        res = T.evaluate(sorted(c.items()), CAL.dates, "2026-10-06", "US.AAA", reps=0)
        self.assertEqual(res["traits"]["G"]["status"], "unknown")
        self.assertIn("missing_session_in_window", res["traits"]["G"]["reason"])
        self.assertEqual(res["contiguous_closes"], 29)

    def test_corporate_action_break_not_bridged(self):
        days = sessions("2026-10-06", 150)
        c = closes_for("AAA", days)
        brk = days[-50]
        for d in days[-50:]:
            c[d] = c[d] / 3  # unadjusted 3:1 split
        res = T.evaluate(sorted(c.items()), CAL.dates, "2026-10-06", "US.AAA", reps=0, breaks=[brk])
        self.assertEqual(res["contiguous_closes"], 50)
        self.assertIn("suspected_corporate_action_break", res["traits"]["V"]["reason"])
        from sources import suspected_splits
        flagged = suspected_splits([{"session_date": d, "close": c[d]} for d in days])
        self.assertEqual([(x["session_date"], x["rule"]) for x in flagged], [(brk, "split_ratio")])


if __name__ == "__main__":
    unittest.main(verbosity=2)

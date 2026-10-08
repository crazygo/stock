"""Fixture tests for --price-basis qfq_daily (OpenD daily K, QFQ) and the ai_basket universe.

Covers: one source per security (no splicing), QFQ rebase detection vs reused files,
cutoff exclusion, quota budget cap (abort + cap), grade-filtered AI basket with tags.
"""
from __future__ import annotations

import json
import os
import re
import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fixtures import CAL, Env, FakeR2, closes_for  # noqa: E402
import backfill  # noqa: E402
import qfq_daily as Q  # noqa: E402
import refresh  # noqa: E402
import render  # noqa: E402
from common import read_json, read_jsonl  # noqa: E402

NOW_1007 = "2026-10-07T08:00:00+00:00"      # cutoff auto -> 2026-10-06
NOW_1008 = "2026-10-08T02:00:00+00:00"      # 10-07 already closed
NO_R2 = lambda: FakeR2()  # noqa: E731
DAYS = CAL.sessions_through("2026-10-06", 161)
DAYS_1007 = CAL.sessions_through("2026-10-07", 162)


def raw_dayk(closes: dict) -> pd.DataFrame:
    rows, prev = [], None
    for d in sorted(closes):
        c = closes[d]
        rows.append({"code": "X", "time_key": f"{d} 00:00:00", "open": round(c * 0.995, 3), "close": c,
                     "high": round(c * 1.01, 3), "low": round(c * 0.99, 3), "volume": 1000,
                     "last_close": prev if prev is not None else c})
        prev = c
    return pd.DataFrame(rows)


def write_reuse(d: Path, code: str, closes: dict, mtime: str | None = "2026-10-07T04:10:00+00:00") -> Path:
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{code}.parquet"
    raw_dayk(closes).to_parquet(p, index=False)
    if mtime:
        t = datetime.fromisoformat(mtime).timestamp()
        os.utime(p, (t, t))
    return p


class FakeQfqOpenD:
    def __init__(self, daily: dict, used=278, remain=722, codes=(), leak_future: dict | None = None, ok=True):
        self.daily, self.used, self.remain, self.codes = daily, used, remain, list(codes)
        self.leak_future = leak_future or {}
        self.ok, self.calls = ok, []

    def __call__(self, args, timeout):
        self.calls.append(list(args))
        if args[0] == "probe":
            if not self.ok:
                return {"ok": False, "error_type": "network", "error": "fixture: refused"}
            q = {"used": self.used, "remain": self.remain}
            if "--quota-detail" in args:
                q.update(codes=sorted(self.codes), request_times={c: "2026-10-06 10:00:00" for c in self.codes})
            return {"ok": True, "history_kl_quota": q}
        if args[0] == "kline_day":
            assert args[args.index("--autype") + 1] == "QFQ"
            code = args[args.index("--code") + 1]
            out = Path(args[args.index("--out") + 1])
            start, end = args[args.index("--start") + 1], args[args.index("--end") + 1]
            if code not in self.codes:
                self.codes.append(code)
                self.used += 1
                self.remain -= 1
            sel = {d: c for d, c in self.daily.get(code, {}).items() if start <= d <= end}
            sel.update(self.leak_future.get(code, {}))  # a misbehaving server returning rows after --end
            if not sel:
                return {"ok": True, "rows": 0, "out": None}
            raw_dayk(sel).to_parquet(out, index=False)
            return {"ok": True, "rows": len(sel), "out": str(out), "requests": 1}
        return {"ok": False, "error": f"fixture: unsupported {args[0]}"}

    def kline_codes(self):
        return [a[a.index("--code") + 1] for a in self.calls if a[0] == "kline_day"]


def basket_files(env: Env) -> tuple[Path, Path]:
    members = [("HK.00100", "MiniMax", 3, ["applications"], "A"), ("US.AAA", "Aaa", 3, ["compute", "connect"], "A"),
               ("US.BBB", "Bbb", 3, ["compute"], "A+"), ("US.CCC", "Ccc", 2, ["power"], "A"),
               ("US.DDD", "Ddd", 1, ["data"], "A-"), ("JP.6857", "Advantest", 2, ["design"], "A-")]
    b = env.tmp / "ai_basket_members.json"
    s = env.tmp / "screen_current.json"
    b.write_text(json.dumps({"built_at": "2026-10-06T03:39:15+00:00", "scope": "fixture",
                             "members": [{"code": c, "name": n, "tier": t, "groups": g} for c, n, t, g, _ in members]}))
    groups = {k: {"label": f"L-{k}", "order": i, "count": 1} for i, k in enumerate(
        ["design", "compute", "memory", "connect", "optics", "systems", "power", "energy", "cloud", "data",
         "applications", "health", "edge", "adjacent", "other", "pending"])}
    s.write_text(json.dumps({"as_of": "2026-10-07", "run_id": "screen_fixture", "rule_version": "fx",
                             "rating_type": "research_priority", "groups": groups,
                             "summary": {"grade_counts": {"A+": 1, "A": 3, "A-": 2}},
                             "records": {c: {"grade": gr, "tier": t, "value_groups": g, "primary_group": g[0],
                                             "verified_grade": "A+" if gr == "A+" else None, "status": "x"}
                                         for c, n, t, g, gr in members}}))
    return b, s


class QfqDaily(unittest.TestCase):
    def setUp(self):
        self.env = Env()
        self.basket, self.screen = basket_files(self.env)
        self.reuse = self.env.tmp / "reuse_tq"
        self.fresh = {f"US.{s}": closes_for(s, DAYS_1007, seed=7) for s in ("AAA", "BBB", "CCC")}

    def tearDown(self):
        self.env.cleanup()

    def args(self, now=NOW_1007, extra=None):
        return self.env.backfill_args(now, ["--universe", "ai_basket", "--ai-basket-file", str(self.basket),
                                            "--ai-screen-file", str(self.screen), "--grades", "A+", "A",
                                            "--price-basis", "qfq_daily", "--qfq-reuse-dir", str(self.reuse),
                                            *(extra or [])])

    def run_bf(self, fake, now=NOW_1007, extra=None):
        return backfill.run(self.args(now, extra), r2_factory=NO_R2, opend_runner=fake)

    @staticmethod
    def ds(m, sid):
        return next(d for d in m["datasets"] if d["security_id"] == sid and d["kind"] == "market_daily")

    @staticmethod
    def rows(ds):
        return pd.read_parquet(backfill.resolve_path(ds["path"]))

    def test_q1_one_source_per_security_no_splice(self):
        # AAA: verified reuse file, not free in quota window -> reused, no OpenD request.
        aaa = {d: self.fresh["US.AAA"][d] for d in DAYS}
        write_reuse(self.reuse, "US.AAA", aaa)
        # BBB: reuse file stops at 09-30 and is on another basis (x0.95) -> fresh fetch, nothing from the file.
        write_reuse(self.reuse, "US.BBB", {d: round(self.fresh["US.BBB"][d] * 0.95, 3) for d in DAYS if d <= "2026-09-30"},
                    mtime="2026-10-01T04:00:00+00:00")
        fake = FakeQfqOpenD(self.fresh)
        m = self.run_bf(fake)["manifest"]
        self.assertEqual(sorted(fake.kline_codes()), ["US.BBB", "US.CCC"])
        a, b, c = (self.ds(m, f"US.{s}") for s in ("AAA", "BBB", "CCC"))
        self.assertEqual(a["source"]["type"], "reused_file")
        self.assertEqual(b["source"]["type"], "opend_fresh")
        self.assertEqual(c["source"]["type"], "opend_fresh")
        for d in (a, b, c):
            df = self.rows(d)
            self.assertEqual(df["layer"].nunique(), 1, d["security_id"])
            self.assertEqual(d["status"], "complete")
            self.assertEqual(d["actual_range"]["last"], "2026-10-06")
            self.assertEqual(d["price_basis"], "opend_day_qfq")
        bdf = self.rows(b).set_index("session_date")
        for d in DAYS[:5]:  # the fresh value, never the x0.95 file value
            self.assertAlmostEqual(bdf.loc[d, "close"], self.fresh["US.BBB"][d], places=6)
        cand = b["layers"][0]["candidates"][0]
        self.assertIn("ends_before_cutoff", cand["suspicious"])
        self.assertIn("written_before_cutoff_close", cand["suspicious"])
        self.assertEqual(m["requirements"]["market_daily"]["price_basis_mode"], "qfq_daily")
        self.assertEqual(m["summary"]["market_daily_by_source"]["opend_fresh"], 2)
        self.assertEqual(sum(v for k, v in m["summary"]["market_daily_by_source"].items() if k.startswith("reused_file")), 1)
        self.assertEqual(m["resources"]["history_kl_quota"]["new_used"], 2)
        self.assertEqual(m["resources"]["history_kl_quota"]["after"]["used"], 280)

    def test_q2_rebase_detected_when_free_fresh_differs(self):
        old = {d: (round(self.fresh["US.AAA"][d] * 0.9, 3) if d < "2026-09-15" else self.fresh["US.AAA"][d]) for d in DAYS}
        write_reuse(self.reuse, "US.AAA", old)
        same = {d: self.fresh["US.BBB"][d] for d in DAYS}
        write_reuse(self.reuse, "US.BBB", same)
        fake = FakeQfqOpenD(self.fresh, codes=["US.AAA", "US.BBB"])  # both already in the 7-day window (free)
        m = self.run_bf(fake)["manifest"]
        self.assertIn("US.AAA", fake.kline_codes())
        a, b = self.ds(m, "US.AAA"), self.ds(m, "US.BBB")
        self.assertEqual(a["source"]["type"], "opend_fresh")
        self.assertTrue(a["source"]["rebase_detected_vs_reuse"])
        chk = a["source"]["rebase_checks"][0]
        self.assertLess(chk["first_differing"], "2026-09-15")
        self.assertEqual(chk["last_differing"] < "2026-09-15", True)
        self.assertIn("qfq_rebase_detected_vs_reuse", a["flags"])
        self.assertAlmostEqual(self.rows(a).set_index("session_date").loc[DAYS[0], "close"], self.fresh["US.AAA"][DAYS[0]])
        self.assertFalse(b["source"]["rebase_detected_vs_reuse"])  # identical overlap -> verified, no rebase
        self.assertEqual(m["summary"]["qfq_rebase_detected_vs_reuse"], ["US.AAA"])
        self.assertEqual(m["resources"]["history_kl_quota"]["free_fetches"], 2)

    def test_q2b_suspicious_reuse_is_not_used_without_fresh(self):
        jump = {d: (self.fresh["US.AAA"][d] * (4 if d < "2026-08-03" else 1)) for d in DAYS}  # unadjusted 4:1 split
        write_reuse(self.reuse, "US.AAA", jump)
        old = {d: self.fresh["US.BBB"][d] for d in DAYS}
        write_reuse(self.reuse, "US.BBB", old, mtime="2026-10-06T12:00:00+00:00")  # before the 10-06 close
        fake = FakeQfqOpenD(self.fresh, ok=False)  # OpenD down -> no fresh data
        m = self.run_bf(fake)["manifest"]
        a, b = self.ds(m, "US.AAA"), self.ds(m, "US.BBB")
        self.assertIn("split_like_jump_in_qfq_series", a["layers"][0]["candidates"][0]["suspicious"])
        self.assertIn("written_before_cutoff_close", b["layers"][0]["candidates"][0]["suspicious"])
        self.assertEqual(a["status"], "unavailable")
        self.assertEqual(b["status"], "unavailable")

    def test_q3_cutoff_excludes_later_rows(self):
        leak = {"US.AAA": {"2026-10-07": self.fresh["US.AAA"]["2026-10-07"]}}
        write_reuse(self.reuse, "US.BBB", {d: self.fresh["US.BBB"][d] for d in DAYS_1007},
                    mtime="2026-10-08T01:00:00+00:00")
        fake = FakeQfqOpenD(self.fresh, leak_future=leak)
        m = self.run_bf(fake, now=NOW_1008, extra=["--cutoff", "2026-10-06"])["manifest"]
        self.assertEqual(m["cutoff_session"], "2026-10-06")
        for s in ("AAA", "BBB", "CCC"):
            d = self.ds(m, f"US.{s}")
            self.assertEqual(d["actual_range"]["last"], "2026-10-06", s)
            self.assertNotIn("2026-10-07", set(self.rows(d)["session_date"]))
        self.assertEqual(self.ds(m, "US.AAA")["source"]["future_rows_dropped"], 1)
        self.assertEqual(self.ds(m, "US.BBB")["layers"][0]["candidates"][0]["future_rows_dropped"], 1)
        req = [a for a in fake.calls if a[0] == "kline_day"]
        self.assertTrue(all(a[a.index("--end") + 1] == "2026-10-06" for a in req))
        # auto cutoff at 10-07 08:00Z: 10-07 has not closed -> 10-06
        m2 = self.run_bf(FakeQfqOpenD(self.fresh, leak_future=leak))["manifest"]
        self.assertEqual(m2["cutoff_session"], "2026-10-06")
        self.assertEqual(self.ds(m2, "US.AAA")["actual_range"]["last"], "2026-10-06")

    def test_q4_quota_budget_abort_and_cap(self):
        fake = FakeQfqOpenD(self.fresh)
        with self.assertRaises(backfill.QuotaBudgetExceeded):
            self.run_bf(fake, extra=["--quota-max-new", "2"])
        self.assertEqual(fake.kline_codes(), [])  # stopped before any history request
        led = read_jsonl(self.env.data / "ledger.jsonl")
        self.assertEqual(led[-1]["status"], "failed")
        self.assertEqual(led[-1]["plan_quota"]["planned_new"], 3)
        self.assertFalse((self.env.data / "current.json").exists())
        # cap mode: fetch until the cap, then skip
        fake = FakeQfqOpenD(self.fresh)
        m = self.run_bf(fake, extra=["--quota-max-new", "2", "--quota-on-exceed", "cap"])["manifest"]
        self.assertEqual(len(fake.kline_codes()), 2)
        skipped = [d for d in m["datasets"] if d.get("kind") == "market_daily"
                   and any(l.get("state") == "skipped_quota_budget" for l in d.get("layers", []))]
        self.assertEqual(len(skipped), 1)
        self.assertEqual(skipped[0]["status"], "unavailable")
        self.assertTrue(any(e["type"] == "quota_budget" for e in m["errors"]))
        self.assertEqual(m["resources"]["history_kl_quota"]["new_used"], 2)
        # min remaining: remain 401, keep 400 -> only one new security allowed
        fake = FakeQfqOpenD(self.fresh, used=599, remain=401)
        m = self.run_bf(fake, extra=["--quota-on-exceed", "cap"])["manifest"]
        self.assertEqual(len(fake.kline_codes()), 1)
        self.assertEqual(m["resources"]["history_kl_quota"]["skipped_quota_budget"], 2)
        # free (already in window) securities never consume the budget
        fake = FakeQfqOpenD(self.fresh, used=600, remain=400, codes=["US.AAA", "US.BBB", "US.CCC"])
        m = self.run_bf(fake, extra=["--quota-max-new", "0"])["manifest"]
        self.assertEqual(len(fake.kline_codes()), 3)
        self.assertEqual(m["resources"]["history_kl_quota"]["new_used"], 0)

    def test_q5_ai_basket_universe_tags_refresh_render(self):
        fake = FakeQfqOpenD(self.fresh)
        m = self.run_bf(fake)["manifest"]
        u = m["universe"]
        self.assertEqual(sorted(x["security_id"] for x in u["members"]), ["HK.00100", "US.AAA", "US.BBB", "US.CCC"])
        self.assertEqual(u["source"]["type"], "ai_value_chain_screen")
        self.assertEqual(u["source"]["screen_run_id"], "screen_fixture")
        self.assertEqual(u["counts"]["by_grade"], {"A": 3, "A+": 1})
        hk = next(x for x in u["members"] if x["security_id"] == "HK.00100")
        self.assertFalse(hk["in_scope"])
        self.assertIn("HK exchange calendar", hk["out_of_scope_reason"])
        self.assertNotIn("HK.00100", fake.kline_codes())
        aaa = next(x for x in u["members"] if x["security_id"] == "US.AAA")
        self.assertEqual(aaa["tags"]["groups"], ["compute", "connect"])
        res = refresh.run(self.env.refresh_args(NOW_1007))
        snap = read_json(Path(res["snapshot_path"]) if Path(res["snapshot_path"]).is_absolute()
                         else self.env.report / res["snapshot_path"])
        r = next(x for x in snap["results"] if x["security_id"] == "US.AAA")
        self.assertEqual(r["tags"]["grade"], "A")
        self.assertEqual(r["price"]["price_basis"], refresh.BASIS_QFQ)
        self.assertEqual(r["data_source"]["type"], "opend_fresh")
        self.assertEqual(snap["intervals"]["data_granularity"], "1d OpenD daily K (QFQ)")
        self.assertEqual(snap["price_basis"]["mode"], "qfq_daily")
        self.assertEqual(snap["counts"]["evaluated_all_five"], 3)
        page = (self.env.report / "current" / "index.html").read_text()
        self.assertIn("AI 价值链 A+/A", page)
        self.assertNotIn("<title>特别关注", page)
        data = json.loads(re.search(r'<script type="application/json" id="data">(.*?)</script>', page, re.S)
                          .group(1).replace("<\\/", "</"))
        self.assertEqual(len(data["latest"]["universe"]["source"]["group_labels"]), 16)
        self.assertEqual(next(s for s in data["securities"] if s["id"] == "US.BBB")["tags"]["grade"], "A+")
        for marker in ("data-filter", "fsearch", "flabels"):
            self.assertIn(marker, page)

    def test_q6_splice_guard_and_compare(self):
        r1 = {d: {"session_date": d, "close": 10.0, "open": 10.0, "high": 10.0, "low": 10.0, "layer": "opend_day_qfq"}
              for d in DAYS[:3]}
        r1[DAYS[3]] = {**r1[DAYS[0]], "session_date": DAYS[3], "layer": "reused_qfq:x"}
        ctx = {"cal": CAL, "required": DAYS, "price_basis_mode": "qfq_daily"}
        with self.assertRaisesRegex(RuntimeError, "splice"):
            backfill.finalize_dataset({"security_id": "US.X", "ticker": "X"}, ctx, r1, {}, [], 0.0, None, {}, {})
        a = {d: {"open": 10.0, "high": 10.0, "low": 10.0, "close": 10.0} for d in DAYS[:5]}
        b = {d: dict(v) for d, v in a.items()}
        self.assertFalse(Q.compare_series(a, b)["rebase_detected"])
        b[DAYS[1]]["close"] = 10.001  # rounding-level difference is tolerated
        self.assertFalse(Q.compare_series(a, b)["rebase_detected"])
        b[DAYS[2]]["close"] = 9.8
        self.assertTrue(Q.compare_series(a, b)["rebase_detected"])


if __name__ == "__main__":
    unittest.main()

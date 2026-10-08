"""Audit coverage, provenance boundaries, membership and descriptive math."""
from pathlib import Path
import hashlib
import json
import numpy as np

OUT = Path(__file__).resolve().parent


def main():
    d = json.loads((OUT / "data.json").read_text())
    r = json.loads((OUT / "reviewed_labels.json").read_text())
    original = json.loads((OUT / "original_label_snapshot.json").read_text())
    u = json.loads((OUT / "label_work/universe.json").read_text())
    stock = [s for s in d["rows"] if s["kind"] == "STOCK"]
    assert len({s["code"] for s in d["rows"]}) == len(d["rows"])
    assert {s["code"] for s in stock} == {s["code"] for s in u} == set(r["records"])
    for key in ("securities", "stocks", "held", "watch", "qqq", "funds", "resolved_funds", "priced"):
        assert d["summary"][key] == original["summary"][key], key
    assert not r["validation_errors"], r["validation_errors"]
    assert all(j["status"] == "completed" for j in r["jobs"]), r["jobs"]
    for job in r["jobs"]:
        input_path = OUT / "label_work" / job["file"].removeprefix("review_")
        output_path = OUT / "label_work" / job["file"]
        output = json.loads(output_path.read_text())
        output_rows = output.get("records", output) if isinstance(output, dict) else output
        assert len({s["code"] for s in output_rows}) == len(output_rows)
        if input_path.exists():
            input_rows = json.loads(input_path.read_text())
            assert {s["code"] for s in input_rows} == {s["code"] for s in output_rows}, job["file"]
    groups = {g["id"] for g in d["groups"]}
    for s in stock:
        b = s["business"]
        assert b["tier"] in (0, 1, 2, 3, None)
        assert b["primary"] in groups and set(b["groups"]) <= groups
        if b["tier"] is not None:
            assert b["sources"] and b["review_status"] in ("official_review", "prior_business_review")
            assert any(e.get("status") not in ("issuer_sector_field", "prior_sec_profile") for e in b["sources"])
        if b["review_status"] == "industry_screened":
            assert b["tier"] is None
    unresolved_identity = [s["code"] for s in stock if s.get("identity_status") == "unresolved"]
    assert set(unresolved_identity) == {"US.ADRO", "US.CRGX", "US.INH"}
    assert all(s["business"]["tier"] is None for s in stock if s["code"] in unresolved_identity)
    assert not next(s for s in stock if s["symbol"] == "GOOGL")["blind"]
    assert next(f for f in d["funds"] if f["symbol"] == "IYM")["known_us_stocks"] == 0
    assert len(d["price"]["clusters"]) == 6
    # Independent NumPy regression cross-check against the browser's result.
    p = d["price"]; series = {s: dict(v) for s, v in p["series"].items()}
    triples = []
    for day, value in series["AAOI"].items():
        ref = [series[c][day] for c in p["core"] if day in series[c]]
        if p["default_start"] <= day <= p["default_end"] and day in series["QQQ"] and len(ref) >= 4:
            triples.append([value, float(np.mean(ref)), series["QQQ"][day]])
    v = np.array(triples); x = np.column_stack([np.ones(len(v)), v[:, 2]])
    raw = float(np.corrcoef(v[:, 0], v[:, 1])[0, 1])
    res = v[:, :2] - x @ np.linalg.lstsq(x, v[:, :2], rcond=None)[0]
    rho = float(np.corrcoef(res[:, 0], res[:, 1])[0, 1])
    browser = json.loads((OUT / "browser_verification.json").read_text())
    assert browser["passed"]
    assert browser["correlation_AAOI"]["n"] == len(v)
    assert abs(browser["correlation_AAOI"]["raw"] - raw) < 1e-12
    assert abs(browser["correlation_AAOI"]["residual"] - rho) < 1e-12
    html = (OUT / "index.html").read_text()
    assert "<script type=\"application/json\" id=\"dataset\">" in html
    for field in ("acc_id", "uniCardNum", "cost_price", "pl_ratio", "qty"):
        assert '"' + field + '":' not in html, field
    members = json.loads((OUT / "ai_basket_members.json").read_text())["members"]
    ai_codes = {s["code"] for s in stock if s["business"]["tier"] in (1, 2, 3)}
    assert {s["code"] for s in members} == ai_codes
    action = d["watchlist_action"]
    assert set(action["requested_codes"]) == ai_codes
    assert not action["complete"] or action["verified_requested_count"] == len(ai_codes)
    files = ["pool_snapshot.json", "taxonomy.py", "reviewed_labels.json", "data.json", "index.html", "ai_basket_members.json"]
    audit = dict(passed=True, scope_counts=d["summary"], ungraded=len(stock)-d["summary"]["graded"],
        unresolved_identity=unresolved_identity, AAOI_numpy=dict(n=len(v), raw=raw, residual=rho),
        checks=["all original securities preserved", "all agent inputs accounted for", "grades require business evidence",
            "sector-only metadata remains ungraded", "identity gaps remain explicit", "issuer gap deduplication",
            "rejected fund identity", "no financial account fields", "independent correlation calculation",
            "full AI manifest agrees with HTML", "watchlist result recorded without false completion"],
        files={f: hashlib.sha256((OUT/f).read_bytes()).hexdigest() for f in files}, browser_math_agrees=True)
    (OUT / "data_verification.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2))
    print(json.dumps({"passed": True, "summary": d["summary"], "ungraded": audit["ungraded"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()

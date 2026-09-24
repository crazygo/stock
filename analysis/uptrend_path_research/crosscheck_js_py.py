"""Cross-check the browser segmentation against the Python reference implementation.

The dashboard ships the JS port of segments.py.  If the two disagree, visual review
of the dashboard would not transfer to the pipeline, so this must hold exactly.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent / "analysis" / "growth_trigger"))

from gt_common import load_hourly, pinned_tickers
from segments import Bar, build_segments, qualifies

CASES = [
    ("NOW", "1h", 0.5, 1.0, 6, "global", 0.08, "amplitude"),
    ("NOW", "4h", 0.5, 1.0, 6, "global", 0.08, "window"),

    ("NVDA", "4h", 1.0, 2.0, 10, "global", 0.05, "amplitude"),
    ("NVDA", "4h", 0.5, 1.0, 6, "global", 0.08, "window"),
    ("MU", "1d", 0.75, 1.5, 8, "trailing", 0.10, "amplitude"),
    ("MU", "4h", 0.5, 1.0, 6, "trailing", 0.08, "window"),
    ("AMD", "4h", 0.25, 0.5, 4, "trailing", 0.08, "amplitude"),
    ("AMD", "1d", 0.5, 1.0, 6, "global", 0.15, "window"),
    ("TSLA", "1h", 1.5, 3.0, 12, "global", 0.08, "amplitude"),
    ("TSLA", "4h", 0.5, 1.0, 6, "global", 0.20, "window"),
    ("STX", "4h", 0.5, 1.0, 6, "global", 0.08, "amplitude"),
    ("MRVL", "1d", 0.5, 1.0, 6, "global", 0.08, "window"),
]
SKIP = {"SNOW", "SPCX", "TSM"}


def to_bars(ticker: str, scale: str) -> list[Bar]:
    s = load_hourly(ticker)
    raw = [Bar(s.ts[i], s.o[i], s.h[i], s.l[i], s.c[i], s.v[i]) for i in range(len(s.ts))]
    if scale == "1h":
        return raw
    n = 4 if scale == "4h" else 8
    out = []
    for i in range(0, len(raw), n):
        g = raw[i:i + n]
        out.append(Bar(g[0].ts, g[0].o, max(b.h for b in g),
                       min(b.l for b in g), g[-1].c, sum(b.v for b in g)))
    return out


def js_segments(bars, delta, min_amp, min_bars, atr_mode, target, qualmode):
    payload = [{"ts": b.ts.isoformat(sep=" "), "o": b.o, "h": b.h, "l": b.l, "c": b.c}
               for b in bars]
    template = (HERE / "debug_template.html").read_text(encoding="utf-8")
    start = template.index("function trueRanges")
    end = template.index("function readControls")
    js = template[start:end]
    script = (
        "const bars=" + json.dumps(payload) + ";\n" + js +
        f"\nconst r=segment(bars,{delta},{min_amp},{min_bars},'{atr_mode}');\n"
        f"r.segs.forEach(s=>{{s.hit=qualifies(bars,s,{target},'{qualmode}');}});\n"
        "console.log(JSON.stringify(r.segs.map(s=>[s.kind,s.i,s.j,+s.amplitude.toFixed(10),"
        "s.bars,s.censored?1:0,s.short?1:0,s.qualified?1:0,s.event_id,s.hit?1:0])));"
    )
    out = subprocess.run(["node", "-e", script], capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError(out.stderr[:800])
    return json.loads(out.stdout.strip().splitlines()[-1])


def main() -> None:
    bad = 0
    for ticker, scale, delta, min_amp, min_bars, atr_mode, target, qualmode in CASES:
        bars = to_bars(ticker, scale)
        py = build_segments(bars, delta=delta, min_amp=min_amp, min_bars=min_bars,
                            atr_mode=atr_mode)
        py_rows = [[s.kind, s.start_idx, s.end_idx, round(s.amplitude, 10), s.bars,
                    1 if s.censored else 0, 1 if s.short else 0,
                    1 if s.qualified else 0, s.event_id,
                    1 if qualifies(bars, s, target=target, mode=qualmode) else 0] for s in py]
        js_rows = js_segments(bars, delta, min_amp, min_bars, atr_mode, target, qualmode)
        same = py_rows == js_rows
        bad += 0 if same else 1
        print(f"{ticker:5s} {scale:3s} δ={delta:<4} A={min_amp:<4} T={min_bars:<3} {atr_mode:8s} "
              f"tgt={target:<5} {qualmode:9s} "
              f"bars={len(bars):5d} segs py={len(py_rows):4d} js={len(js_rows):4d} "
              f"{'MATCH' if same else 'MISMATCH'}")
        if not same:
            for a, b in zip(py_rows, js_rows):
                if a != b:
                    print("   py:", a, "\n   js:", b)
                    break
            if len(py_rows) != len(js_rows):
                print("   length differs")
    print("ALL MATCH" if bad == 0 else f"{bad} MISMATCHES")


if __name__ == "__main__":
    main()

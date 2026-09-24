"""Build the interactive segmentation debug dashboard (debug.html).

The segmentation runs in the browser so parameters can be tuned live; the Python
implementation in segments.py is the reference and is covered by test_segments.py.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent / "analysis" / "growth_trigger"))
from gt_common import load_hourly, pinned_tickers

SKIP = {"SNOW", "SPCX", "TSM"}      # no eligible stock-days in the earlier study


def main() -> None:
    bars_count = 0
    stocks = {}
    for t in pinned_tickers():
        if t in SKIP:
            continue
        series = load_hourly(t)
        if not series.ts:
            continue
        bars = [{"ts": series.ts[i].isoformat(sep=" "), "o": series.o[i],
                 "h": series.h[i], "l": series.l[i], "c": series.c[i]}
                for i in range(len(series.ts))]
        bars_count += len(bars)
        stocks[t] = {"bars": bars}
    payload = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "bar_count": bars_count,
        "baserate": {},
        "stocks": stocks,
    }
    html_text = (HERE / "debug_template.html").read_text(encoding="utf-8")
    html_text = html_text.replace("/*__DATA__*/", json.dumps(payload, ensure_ascii=False))
    out = HERE / "debug.html"
    out.write_text(html_text, encoding="utf-8")
    print(f"wrote {out} ({out.stat().st_size/1024:.0f} KB), "
          f"{len(stocks)} stocks, {bars_count:,} bars")


if __name__ == "__main__":
    main()

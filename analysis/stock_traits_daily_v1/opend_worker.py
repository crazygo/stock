#!/usr/bin/env python3
"""Killable, read-only OpenD worker. Prints one JSON object on stdout.

Actions: probe [--quota-detail] | groups | members --group NAME |
         kline --code US.X --start D --end D --out PATH            (5m, autype NONE)
         kline_day --code US.X --start D --end D --out PATH [--autype QFQ]   (daily K)
Only quote-context read calls are used (no trading context, no watchlist edits).
"""
from __future__ import annotations

import argparse
import json
import socket
import sys
import time

RESULT_PREFIX = "@@STDV1_RESULT@@ "  # futu-api writes its own log lines to stdout; the parent looks for this marker


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["probe", "groups", "members", "kline", "kline_day"])
    ap.add_argument("--host", required=True)
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--group")
    ap.add_argument("--code")
    ap.add_argument("--start")
    ap.add_argument("--end")
    ap.add_argument("--out")
    ap.add_argument("--autype", default="QFQ", choices=["QFQ", "NONE", "HFQ"])
    ap.add_argument("--quota-detail", action="store_true", help="probe: also list securities in the quota window")
    a = ap.parse_args()
    started = time.monotonic()
    out: dict = {"action": a.action, "host": a.host, "port": a.port}
    try:
        with socket.create_connection((a.host, a.port), timeout=3):
            pass
    except OSError as exc:
        out.update(ok=False, error_type="network", error=f"tcp connect failed: {type(exc).__name__}: {exc}")
        print(RESULT_PREFIX + json.dumps(out, ensure_ascii=False), flush=True)
        return 2
    try:
        import futu as ft  # noqa: WPS433
    except ImportError as exc:
        out.update(ok=False, error_type="dependency", error=f"futu-api not installed: {exc}")
        print(RESULT_PREFIX + json.dumps(out, ensure_ascii=False), flush=True)
        return 3
    ft.SysConfig.enable_proto_encrypt(False)
    q = ft.OpenQuoteContext(host=a.host, port=a.port)
    try:
        if a.action == "probe":
            ret, data = q.get_global_state()
            out.update(ok=ret == ft.RET_OK, state=data if ret == ft.RET_OK else None, error=None if ret == ft.RET_OK else str(data))
            try:  # read-only quota query (history kline symbols used/remaining in the rolling 7-day window (Futu docs))
                qret, qdata = q.get_history_kl_quota(get_detail=bool(a.quota_detail))
                if qret == ft.RET_OK:
                    out["history_kl_quota"] = {"used": qdata[0], "remain": qdata[1]}
                    if a.quota_detail:
                        det = qdata[2] if len(qdata) > 2 else []
                        out["history_kl_quota"]["codes"] = sorted({str(x.get("code")) for x in (det or []) if x.get("code")})
                        rt: dict = {}
                        for x in det or []:
                            c, t = str(x.get("code") or ""), str(x.get("request_time") or "")
                            if c and t and t > rt.get(c, ""):
                                rt[c] = t
                        out["history_kl_quota"]["request_times"] = rt
                        times = sorted(rt.values())
                        out["history_kl_quota"]["request_time_range"] = [times[0], times[-1]] if times else None
                else:
                    out["history_kl_quota"] = {"error": str(qdata)}
            except Exception as exc:  # noqa: BLE001
                out["history_kl_quota"] = {"error": f"{type(exc).__name__}: {exc}"}
        elif a.action == "groups":
            ret, data = q.get_user_security_group()
            out.update(ok=ret == ft.RET_OK, groups=data.to_dict("records") if ret == ft.RET_OK else None,
                       error=None if ret == ft.RET_OK else str(data))
        elif a.action == "members":
            ret, data = q.get_user_security(group_name=a.group)
            out.update(ok=ret == ft.RET_OK, members=data.to_dict("records") if ret == ft.RET_OK else None,
                       error=None if ret == ft.RET_OK else str(data))
        elif a.action in ("kline", "kline_day"):
            import pandas as pd
            ktype = ft.KLType.K_5M if a.action == "kline" else ft.KLType.K_DAY
            autype = ft.AuType.NONE if a.action == "kline" else getattr(ft.AuType, a.autype)
            # Regular session only (daily close needs no extended hours; ~2.5x fewer pages / quota-friendly).
            # OpenD limit: 60 request_history_kline calls per 30 s -> >= 0.6 s between pages, back off on limits.
            frames, page, requests, retries = [], None, 0, 0
            while True:
                ret, df, nxt = q.request_history_kline(code=a.code, start=a.start, end=a.end, ktype=ktype,
                                                       autype=autype, max_count=1000, extended_time=False,
                                                       page_req_key=page)
                requests += 1
                if ret != ft.RET_OK:
                    msg = str(df)
                    if retries < 3 and any(k in msg for k in ("频率", "frequen", "too fast", "limit")):
                        retries += 1
                        time.sleep(6.0 * retries)
                        continue
                    raise RuntimeError(msg)
                page = nxt
                if df is not None and len(df):
                    frames.append(df)
                if not page:
                    break
                time.sleep(0.6)
            raw = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
            if len(raw):
                raw.to_parquet(a.out, compression="zstd", compression_level=7)
            out.update(ok=True, rows=int(len(raw)), requests=requests, retries=retries, extended_time=False,
                       ktype=str(a.action), autype="NONE" if a.action == "kline" else a.autype,
                       first=str(raw["time_key"].min())[:10] if len(raw) else None,
                       last=str(raw["time_key"].max())[:10] if len(raw) else None,
                       out=a.out if len(raw) else None)
    except Exception as exc:  # report, never hide
        out.update(ok=False, error_type="opend", error=f"{type(exc).__name__}: {exc}")
    finally:
        q.close()
    out["elapsed_seconds"] = round(time.monotonic() - started, 3)
    print(RESULT_PREFIX + json.dumps(out, ensure_ascii=False, default=str), flush=True)
    return 0 if out.get("ok") else 4


if __name__ == "__main__":
    sys.exit(main())

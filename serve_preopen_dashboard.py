#!/usr/bin/env python3
"""Serve the local dashboard and persist its pinned ticker list to JSON."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PINS_FILE = ROOT / "analysis" / "preopen_three_week" / "pinned_stocks.json"
TICKER_RE = re.compile(r"^[A-Z0-9.:-]{1,16}$")


class DashboardHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def _json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path == "/api/pinned-stocks":
            if not PINS_FILE.exists():
                self._json(200, {"tickers": []})
                return
            try:
                payload = json.loads(PINS_FILE.read_text(encoding="utf-8"))
                tickers = payload.get("tickers", [])
                if not isinstance(tickers, list):
                    raise ValueError("tickers must be a list")
                self._json(200, {"tickers": tickers})
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                self._json(500, {"error": f"Cannot read pinned stocks: {exc}"})
            return
        super().do_GET()

    def do_PUT(self) -> None:
        if self.path != "/api/pinned-stocks":
            self._json(404, {"error": "Not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 65536:
                raise ValueError("Invalid request size")
            payload = json.loads(self.rfile.read(length))
            tickers = payload.get("tickers") if isinstance(payload, dict) else None
            if not isinstance(tickers, list) or len(tickers) > 500:
                raise ValueError("tickers must be a list of at most 500 symbols")
            if any(not isinstance(ticker, str) or not TICKER_RE.fullmatch(ticker) for ticker in tickers):
                raise ValueError("Invalid ticker symbol")
            normalized = list(dict.fromkeys(tickers))
            PINS_FILE.parent.mkdir(parents=True, exist_ok=True)
            record = {
                "version": 1,
                "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "tickers": normalized,
            }
            temporary = PINS_FILE.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            temporary.replace(PINS_FILE)
            self._json(200, {"tickers": normalized})
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            self._json(400, {"error": str(exc)})


if __name__ == "__main__":
    server = ThreadingHTTPServer(("127.0.0.1", 8768), DashboardHandler)
    print(f"Dashboard: http://127.0.0.1:8768/analysis/preopen_three_week/dashboard.html")
    print(f"Pinned stocks: {PINS_FILE}")
    server.serve_forever()

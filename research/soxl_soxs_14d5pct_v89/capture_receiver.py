"""Temporary loopback form for saving exact read-only browser document observations.

Run manually for acquisition, stop after use. Does not fetch, execute documents,
accept arbitrary paths, or modify existing evidence. Browser actions use cua_repl.
"""
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse
from datetime import datetime, timezone
import hashlib
import html
import json
import secrets

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / '.cache/soxl_soxs_14d5pct_v89/browser_captures'
ALLOWED = {'ir.amd.com', 'nvidianews.nvidia.com', 'investor.nvidia.com',
           'investors.broadcom.com', 'investors.micron.com', 'ir.appliedmaterials.com',
           'newsroom.lamresearch.com', 'investor.lamresearch.com', 'ir.kla.com', 'www.intc.com',
           'www.globenewswire.com'}
TOKEN = secrets.token_urlsafe(32)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_):pass

    def page(self, message='Ready'):
        body = ('<!doctype html><html><head><meta charset="utf-8"><title>Local source capture</title></head>'
                '<body><h1>Local source capture</h1><form method="post" action="/capture">'
                '<label for="payload">Observation JSON</label><textarea id="payload" name="payload" rows="8" cols="80"></textarea>'
                '<input type="hidden" name="token" value="' + TOKEN + '">'
                '<button type="submit">Save observation</button></form><output id="receipt">' + html.escape(message) + '</output></body></html>').encode()
        self.send_response(200);self.send_header('Content-Type','text/html; charset=utf-8')
        self.send_header('Content-Length',str(len(body)));self.send_header('Cache-Control','no-store')
        self.send_header('Content-Security-Policy',"default-src 'none'; form-action 'self'; frame-ancestors 'none'")
        self.end_headers();self.wfile.write(body)

    def do_GET(self):
        if self.path == '/queue':
            queue = ROOT / '.cache/soxl_soxs_14d5pct_v89/browser_queue.json'
            body = ('<!doctype html><html><head><meta charset="utf-8"><title>Official document queue</title></head><body><pre>' + html.escape(queue.read_text()) + '</pre></body></html>').encode()
            self.send_response(200);self.send_header('Content-Type','text/html; charset=utf-8')
            self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body);return
        if self.path != '/':self.send_error(404);return
        self.page()

    def do_POST(self):
        expected = 'http://127.0.0.1:' + str(self.server.server_port)
        if self.path != '/capture' or self.headers.get('Origin') != expected:
            self.send_error(403);return
        try:
            n = int(self.headers.get('Content-Length','0'))
            if not 0 < n <= 8_000_000:raise ValueError('Payload size')
            fields = parse_qs(self.rfile.read(n).decode())
            if fields.get('token') != [TOKEN]:raise ValueError('Token')
            data = json.loads(fields['payload'][0])
            url = urlparse(data['url'])
            if url.scheme != 'https' or url.hostname not in ALLOWED:raise ValueError('Unregistered source URL')
            if not data.get('text') or not data.get('observed_at'):raise ValueError('Missing document observation')
            data['capture_saved_at'] = datetime.now(timezone.utc).isoformat()
            data['capture_method'] = 'cua_repl_read_only_DOM_observation_saved_via_loopback_form'
            payload = (json.dumps(data, ensure_ascii=False, indent=2) + '\n').encode()
            digest = hashlib.sha256(payload).hexdigest()
            OUT.mkdir(parents=True,exist_ok=True)
            file = OUT / (digest + '.json')
            with file.open('xb') as f:f.write(payload)
            self.page('Saved ' + file.name)
        except (ValueError, KeyError, IndexError, FileExistsError):
            self.send_error(400,'Invalid capture')


if __name__ == '__main__':
    server = HTTPServer(('127.0.0.1',0),Handler)
    print('http://127.0.0.1:' + str(server.server_port) + '/',flush=True)
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:server.server_close()

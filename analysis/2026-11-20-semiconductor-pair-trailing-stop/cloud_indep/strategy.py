"""Independent re-implementation of ALGORITHM.zh-CN.md (2026-11-21). Written from the doc only.
Only shadow_freeze.json (coefficients/thresholds) is reused."""
import hashlib, json, math, sys
from datetime import timedelta
import numpy as np, pandas as pd

FREEZE = '/workspace/soxl-alert/ref/analysis/2026-11-20-semiconductor-pair-trailing-stop/entry_filters/shadow_freeze.json'
SYMS = ('SOXL', 'SOXS')
FIVE = pd.Timedelta(minutes=5)
FEE = 2.5

def load_model():
    raw = open(FREEZE, 'rb').read()
    h = hashlib.sha256(raw).hexdigest(); assert h.startswith('ed4c7604') and h.endswith('be34'), h
    m = json.loads(raw)
    def find(d, k):
        if isinstance(d, dict):
            if k in d: return d[k]
            for v in d.values():
                r = find(v, k)
                if r is not None: return r
        return None
    M = {k: m['model'][k] for k in ('mean', 'scale', 'weights', 'intercept')}
    M.update({k: m['thresholds'][k] for k in ('q50', 'q75', 'q90')})
    feats = m['features']
    assert feats == ['return5','return12','drawdown5','drawdown12','close_efficiency12','close_location12','inverse_return5']
    return M, feats, h

# ---------- data ----------
def frame_from_standard(path, sym):
    d = pd.read_parquet(path)
    d = d.assign(start=pd.to_datetime(d.start_at, utc=True))
    return d[['start', 'open', 'high', 'low', 'close', 'volume']].assign(symbol=sym)

def frame_from_futu(path, sym):
    d = pd.read_parquet(path)
    end = pd.to_datetime(d.time_key).dt.tz_localize('America/New_York')
    return pd.DataFrame(dict(start=(end - FIVE).dt.tz_convert('UTC'), open=d.open, high=d.high,
                             low=d.low, close=d.close, volume=d.volume, symbol=sym))

def trading_day(start_utc):
    et = start_utc.dt.tz_convert('America/New_York')
    return (et + pd.Timedelta(hours=4)).dt.date   # day starts 20:00 ET previous day

def window(frames, d0, d1):
    out = {}
    for s, f in frames.items():
        td = trading_day(f.start)
        out[s] = f[(td >= pd.Timestamp(d0).date()) & (td <= pd.Timestamp(d1).date())].sort_values('start').reset_index(drop=True)
    return out

# ---------- primitives ----------
def trend(bars):  # bars: list of (O,H,L,C)
    peak = 0.0; dd = 0.0; blocked = False
    for o, h, l, c in bars:
        peak = max(peak, h); dd = max(dd, 1 - l / peak)
        if l <= 0.97 * peak + 1e-10: blocked = True
    gain = bars[-1][3] / bars[0][0] - 1
    return gain, dd, (bars[-1][3] > bars[0][0] and not blocked)

class Pos:
    def __init__(s, sym, price, qty, entry_t, tag=None):
        s.sym, s.P, s.Q, s.entry_t, s.tag = sym, price, qty, entry_t, tag
        s.peak = price; s.target = 1.08 * price; s.exit = None
    def on_price(s, x, is_open):
        stop = 0.97 * s.peak
        if x <= stop + 1e-10: s.exit = (x if is_open else stop, 'stop'); return True
        if x >= s.target - 1e-10: s.exit = (x if is_open else s.target, 'target'); return True
        s.peak = max(s.peak, x); return False

def run(frames, path='OLHC', verbose=False):
    M, feats, h = load_model()
    mean, scale, w = map(np.array, (M['mean'], M['scale'], M['weights']))
    t = frames['SOXL'].start.values
    assert (frames['SOXS'].start.values == t).all(), 'ETF timestamps not aligned'
    n = len(t); ts = list(frames['SOXL'].start)
    B = {s: frames[s][['open', 'high', 'low', 'close']].to_numpy() for s in SYMS}
    contig = np.r_[False, np.diff(t) == np.timedelta64(5, 'm')]  # contig[i]: t[i]-t[i-1]==5m
    # run length of contiguous bars ending at i
    run_len = np.zeros(n, int)
    for i in range(n): run_len[i] = run_len[i-1] + 1 if i and contig[i] else 1
    def ok_span(a, b):  # bars a..b inclusive contiguous
        return a >= 0 and run_len[b] >= b - a + 1
    def W(s, i, k): return [tuple(x) for x in B[s][i-k:i]]  # ends i-1
    def score(s, i):
        if i < 12 or not ok_span(i-12, i-1): return None
        w5, w12 = W(s, i, 5), W(s, i, 12)
        g5, d5, _ = trend(w5); g12, d12, _ = trend(w12)
        cl = [b[3] for b in w12]
        den = sum(abs(cl[j+1]-cl[j]) for j in range(11)); eff = abs(cl[-1]-cl[0]) / den if den else 0.0
        hi = max(b[1] for b in w12); lo = min(b[2] for b in w12)
        loc = (cl[-1]-lo)/(hi-lo) if hi != lo else 0.5
        o = 'SOXS' if s == 'SOXL' else 'SOXL'; p5 = W(o, i, 5); inv = p5[-1][3]/p5[0][0]-1
        x = np.array([g5, g12, d5, d12, eff, loc, inv])
        z = (x-mean)/scale
        return 1/(1+math.exp(-(M['intercept']+float(z@w))))
    order = {'OLHC': (2, 1, 3), 'OHLC': (1, 2, 3)}[path]

    base = None; base_pending = None
    paper_cash = 1000.0; tranches = []  # list of dict(pos, stage, dec, fill)
    rnd = None; paper_pending = None; paper_log = []
    fin_cash = 1400.0; fin = None; fin_trades = []; fin_buys = 0
    for i in range(n):
        # ---- open of bar i: gap exits
        def open_exit(p, s): return p.on_price(B[s][i][0], True)
        if base and open_exit(base, base.sym): base = None
        for tr in list(tranches):
            if tr['pos'].exit is None and open_exit(tr['pos'], tr['pos'].sym):
                paper_cash += tr['pos'].Q*tr['pos'].exit[0]-FEE; tr['exit_t'] = ts[i]
        if fin and open_exit(fin, fin.sym):
            fin_cash += fin.Q*fin.exit[0]-FEE; fin_trades.append(close_trade(fin, ts[i])); fin = None
        # ---- execute bookings
        if base_pending is not None:
            s = base_pending; base = Pos(s, B[s][i][0], 1.0, ts[i]); base_pending = None
        third_fill = None
        if paper_pending is not None:
            pp = paper_pending; paper_pending = None; s = pp['sym']; px = B[s][i][0]
            first_alive = pp['stage'] == 1 or (rnd and rnd['first']['pos'].exit is None)
            amt = (100., 200., 300.)[pp['stage']-1]
            if first_alive and paper_cash >= amt + FEE - 1e-12:
                paper_cash -= amt + FEE
                tr = dict(pos=Pos(s, px, amt/px, ts[i]), stage=pp['stage'], dec=pp['dec'], fill=i, score=pp['score'], sigc=pp['sigc'])
                tranches.append(tr); paper_log.append(dict(stage=pp['stage'], sym=s, t=ts[i], price=px, score=pp['score']))
                if pp['stage'] == 1: rnd = dict(first=tr, sym=s, n=1, last=tr)
                else: rnd['n'] += 1; rnd['last'] = tr
                if pp['stage'] == 2: rnd['p2'] = px
                if pp['stage'] == 3: third_fill = (tr, pp)
            else:
                paper_log.append(dict(stage=pp['stage'], sym=s, t=ts[i], price=px, score=pp['score'], cancelled=True))
        if third_fill:
            tr, pp = third_fill
            if pp['gate'] and fin is None and fin_cash >= 1002.5 - 1e-12:
                fin = Pos(tr['pos'].sym, tr['pos'].P, 1000.0/tr['pos'].P, ts[i]); fin.tag = dict(score=pp['score'], p2=pp['p2'], sigc=pp['sigc'])
                fin_cash -= 1002.5; fin_buys += 1
        # ---- decision at t[i]+1s, inputs up to i-1, order at open i+1
        nxt = i+1 < n and contig[i+1]
        if nxt and i >= 5 and ok_span(i-5, i-1) and contig[i]:
            tests = {s: trend(W(s, i, 5)) for s in SYMS}
            base_pick = None
            if base is None:
                cands = [(-tests[s][0], s) for s in SYMS if tests[s][2]]
                if cands: base_pick = min(cands)[1]; base_pending = base_pick
            open_tr = [tr for tr in tranches if tr['pos'].exit is None]
            if not open_tr:
                rnd = None
                if base_pick:
                    sc = score(base_pick, i)
                    if sc is not None and sc >= M['q50']:
                        paper_pending = dict(sym=base_pick, stage=1, dec=i, score=sc, sigc=B[base_pick][i-1][3], gate=None, p2=None)
            elif rnd and rnd['first']['pos'].exit is None and rnd['n'] < 3:
                s = rnd['sym']; stage = rnd['n']+1; last = rnd['last']; first = rnd['first']
                sc = score(s, i)
                thr = M['q75'] if stage == 2 else M['q90']
                c_ok = (i - last['fill'] >= 1 and tests[s][2] and B[s][i-1][3] > B[s][last['dec']-1][3]
                        and ok_span(first['dec']-5, i-1) and trend([tuple(x) for x in B[s][first['dec']-5:i]])[2])
                if sc is not None and sc >= thr and c_ok:
                    gate = None; p2 = rnd.get('p2')
                    if stage == 3: gate = (B[s][i-1][3]-p2)*600 >= 5*p2 - 1e-9
                    paper_pending = dict(sym=s, stage=stage, dec=i, score=sc, sigc=B[s][i-1][3], gate=gate, p2=p2)
        # ---- intrabar path
        for k in order:
            if base and base.on_price(B[base.sym][i][k], False): base = None
            for tr in tranches:
                p = tr['pos']
                if p.exit is None and p.on_price(B[p.sym][i][k], False):
                    paper_cash += p.Q*p.exit[0]-FEE; tr['exit_t'] = ts[i]
            if fin and fin.on_price(B[fin.sym][i][k], False):
                fin_cash += fin.Q*fin.exit[0]-FEE; fin_trades.append(close_trade(fin, ts[i])); fin = None
        tranches = [tr for tr in tranches if tr['pos'].exit is None] + []  # keep only open for state
        if rnd and not any(tr['pos'].exit is None for tr in [rnd['first']] ) and not tranches: rnd = None
    stages = [sum(1 for x in paper_log if x['stage'] == k and not x.get('cancelled')) for k in (1, 2, 3)]
    opn = None
    if fin: opn = dict(sym=fin.sym, entry=str(fin.entry_t), P=fin.P, mark=B[fin.sym][-1][3])
    closed = fin_trades
    net = sum(x['net'] for x in closed)
    return dict(path=path, paper=stages, buys=fin_buys, wins=sum(x['net'] > 0 for x in closed),
                losses=sum(x['net'] <= 0 for x in closed), fees=5.0*len(closed)+(2.5 if fin else 0),
                net=net, trades=closed, open=opn, paper_log=paper_log)

def close_trade(p, t_exit):
    px, why = p.exit
    return dict(symbol=p.sym, entry=str(p.entry_t.tz_convert('America/New_York')), entry_px=p.P, score=p.tag['score'],
                p2=p.tag['p2'], sigc=p.tag['sigc'], exit=str(t_exit.tz_convert('America/New_York')), exit_px=px, reason=why,
                gross=p.Q*(px-p.P), net=p.Q*(px-p.P)-5.0)

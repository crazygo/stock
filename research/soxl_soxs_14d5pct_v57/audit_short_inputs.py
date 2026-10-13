"""Independent raw-array proxy payoff and original-target isolation audit."""
import argparse
from pathlib import Path
from short_learning import core, np, pd, read, dump, sha, restrict_market, VARIANTS, day_cutoff
from verify_continuation import near


def reference_label(market, s, at):
    day = pd.Timestamp(at.tz_convert(core.ET).date())
    future = [d for d in market.sessions if d > day]
    if day not in market.calendar or not future:
        return {'known': False, 'reason': 'pending_calendar', 'release': None, 'nominal_release': None}
    following = future[0]; close = pd.Timestamp(market.calendar[following]['close_at'])
    result = {'known': False, 'reason': 'missing_full_short_window', 'release': None,
              'nominal_release': close + pd.Timedelta(seconds=1)}
    f = market.bars[s]
    positions = np.flatnonzero((f.date.to_numpy() >= day.to_datetime64()) &
                              (f.date.to_numpy() <= following.to_datetime64()) & f.session_type.eq('regular').to_numpy())
    g = f.iloc[positions]
    expected = []
    for d in [day, following]:
        t = pd.Timestamp(market.calendar[d]['open_at'])
        end = pd.Timestamp(market.calendar[d]['close_at'])
        while t < end:
            expected.append(t + pd.Timedelta(minutes=5)); t += pd.Timedelta(minutes=5)
    valid = list(g.index) == expected and all(g.start_at == g.end_at - pd.Timedelta(minutes=5))
    valid = valid and g.available_at.notna().all() and all(g.available_at >= g.end_at)
    if not valid:
        return result
    possible = np.flatnonzero((g.start_at > at + pd.Timedelta(seconds=30)).to_numpy() & g.date.eq(day).to_numpy())
    if not len(possible) or (g.start_at.iloc[possible[0]] - at).total_seconds() > 360:
        result['reason'] = 'no_eligible_entry'; return result
    g = g.iloc[possible[0]:]
    factor = g.share_factor.to_numpy(float)
    entry = g.open.iloc[0] * 1.0005 * factor[0]
    target = np.ceil(g.open.iloc[0] * 1.0005 * 1.05 * 100 - 1e-10) / 100 * factor[0]
    lows = g.low.to_numpy(float) * factor
    highs = g.high.to_numpy(float) * factor
    opens = g.open.to_numpy(float) * factor
    stops = np.floor(entry * .94 / factor * 100 + 1e-10) / 100 * factor
    limits = np.ceil(target / factor * 100 - 1e-10) / 100 * factor
    hits = (lows <= stops) | (highs >= limits)
    i = int(np.flatnonzero(hits)[0]) if hits.any() else len(g) - 1
    if hits.any() and lows[i] <= stops[i]:
        reason = 'risk_stop'; fill = min(opens[i], stops[i]) * .9995
    elif hits.any():
        reason = 'target'; fill = limits[i]
    else:
        reason = 'short_close'; fill = g.close.iloc[-1] * factor[-1] * .9995
    dividend = 0.
    for a in market.actions[s]:
        ex = pd.Timestamp(a['date'])
        if a['type'] == 'CashDividends' and day < ex <= g.date.iloc[i]:
            dividend += float(a['value']) * g.loc[g.date == ex, 'share_factor'].iloc[0]
    result.update(known=True, reason=reason, release=max(result['nominal_release'], f.iloc[positions].available_at.max()),
                  entry_at=g.start_at.iloc[0], entry_fill_economic=entry, target_economic=target,
                  exit_at=g.end_at.iloc[i], exit_fill_economic=fill,
                  dividend_per_economic_unit=dividend, net_return=(fill * .9995 + dividend) / (entry * 1.0005) - 1,
                  same_bar_both_barriers=bool(hits.any() and lows[i] <= stops[i] and highs[i] >= limits[i]),
                  gap_below_stop=bool(reason == 'risk_stop' and opens[i] < stops[i]), required_rows=len(expected))
    return result


def audit(inputs):
    inputs = Path(inputs); m = read(inputs / 'input_manifest.json')
    for f, h in m['derived_files_sha256'].items(): assert sha(inputs / f) == h, f
    for f, h in m['original_inputs_sha256'].items(): assert sha(inputs / f) == h, f
    market = core.load_market(); assert market.manifest == m['full_market']
    market = restrict_market(market, pd.Timestamp(m['observation_cutoff_exclusive']))
    original = pd.read_parquet(inputs / 'activity_panel.parquet')
    short = pd.read_parquet(inputs / 'short_panel.parquet')
    unchanged = [c for c in original if c not in ['C', 'D', 'M', 'release', 'terminal_release']]
    pd.testing.assert_frame_equal(original[unchanged], short[unchanged])
    records = read(inputs / 'short_labels.json'); refs = {}; known = checks = 0
    assert len(records) == 2 * len(original)
    for r in records:
        at = pd.Timestamp(r['decision_at']); s = r['symbol']; ref = reference_label(market, s, at)
        assert (at, s) not in refs; refs[at, s] = ref
        for k, want in ref.items():
            if isinstance(want, pd.Timestamp): assert pd.Timestamp(r[k]) == want, (at, s, k)
            elif isinstance(want, (float, np.floating)): near(r[k], want)
            else: assert r[k] == want, (at, s, k)
        known += int(ref['known']); checks += 1
    for at in original.index:
        pair = [refs[at, s] for s in core.ETFS]
        if all(r['known'] for r in pair):
            l, s = [np.log1p(r['net_return']) for r in pair]
            np.testing.assert_allclose(short.loc[at, ['C', 'D', 'M']].to_numpy(float), [(l+s)/2, (l-s)/2, abs(l-s)/2], atol=1e-12)
            assert short.loc[at, 'terminal_release'] == short.loc[at, 'release'] == max(r['release'] for r in pair)
        else:
            assert short.loc[at, ['C', 'D', 'M']].isna().all() and pd.isna(short.loc[at, 'terminal_release'])
        assert short.loc[at, 'short_nominal_release'] == max(r['nominal_release'] for r in pair)
    out = {'passed': True, 'label_checks': checks, 'known_individual_labels': known,
           'pair_equations': len(short), 'original_fourteen_day_inputs_unchanged': True,
           'unchanged_feature_fields': len(unchanged), 'unknown_pairs_retained': int(short.terminal_release.isna().sum()),
           'input_manifest_sha256': sha(inputs / 'input_manifest.json'), 'new_capital_outcomes_seen': False}
    dump(inputs / 'short_input_audit.json', out); print(out, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--inputs', type=Path, required=True); audit(p.parse_args().inputs)

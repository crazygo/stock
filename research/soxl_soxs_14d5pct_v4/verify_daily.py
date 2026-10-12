"""Independently audit daily temporal membership, frozen forecasts and raw-bar cash flows."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd
from walkforward import core


def verify(run):
    run = Path(run)
    results = json.loads((run / 'results.json').read_text())
    protocol = json.loads((run / 'registration.json').read_text())['protocol']
    records = json.loads((run / 'daily_registry.json').read_text())
    features = pd.read_parquet(run / 'features.parquet')
    raw_labels = json.loads((run / 'labels.json').read_text())
    labels = pd.DataFrame(raw_labels)
    labels.decision_at = pd.to_datetime(labels.decision_at, utc=True)
    labels.label_available_at = pd.to_datetime(labels.label_available_at, utc=True)
    predictions = pd.read_csv(run / 'predictions.csv', index_col='decision_at')
    predictions.index = pd.to_datetime(predictions.index, utc=True)
    audited_models = 0
    for record in records:
        if 'model_id' not in record:
            continue
        cutoff = pd.Timestamp(record['training_cutoff'])
        train = pd.DatetimeIndex(pd.to_datetime(record['train_members'], utc=True))
        calibration = pd.DatetimeIndex(pd.to_datetime(record['calibration_members'], utc=True))
        eligible = labels[(labels.label_available_at < cutoff) & labels.complete_window.eq(True) &
                          np.isfinite(labels.net_return) & (labels.decision_at < cutoff)]
        pairs = eligible.groupby('decision_at').symbol.nunique()
        eligible_dates = pd.DatetimeIndex(pairs[pairs == 2].index).intersection(features.index)
        assert train.equals(eligible_dates[-protocol['training_window_dates']:])
        expected_calibration = eligible_dates.intersection(predictions.index)
        assert calibration.equals(expected_calibration[-protocol['calibration_window_dates']:])
        assert (train < cutoff).all() and (calibration < cutoff).all()
        for symbol in core.ETFS:
            t = labels[labels.symbol == symbol].set_index('decision_at')
            assert t.loc[train, 'label_available_at'].max() < cutoff
            if len(calibration):
                assert t.loc[calibration, 'label_available_at'].max() < cutoff
                assert (t.loc[calibration, 'label_available_at'] > calibration).all()
                assert (pd.to_datetime(predictions.loc[calibration, 'training_cutoff'], utc=True) <= calibration).all()
                bias = (t.loc[calibration, 'net_return'] - predictions.loc[calibration, symbol + '_raw_ev']).mean()
                assert np.isclose(bias, record['corrections'][symbol]['mean_residual'], rtol=0, atol=1e-14)
        assert hashlib.sha256((run / record['model_file']).read_bytes()).hexdigest() == record['model_sha256']
        saved = json.loads((run / 'models' / (record['date'] + '.forecast.json')).read_text())
        actual = predictions.loc[pd.Timestamp(saved['forecast_at'])]
        assert saved['model_id'] == actual.model_id == record['model_id']
        assert saved['action'] == actual.action
        for symbol in core.ETFS:
            assert np.isclose(saved[symbol + '_raw_ev'], actual[symbol + '_raw_ev'], rtol=0, atol=1e-14)
        audited_models += 1
    may = next(r for r in records if r['date'] == '2026-05-01')
    september = next(r for r in records if r['date'] == '2026-09-01')
    assert pd.Timestamp(may['train_last_label_available']) < pd.Timestamp('2026-05-01', tz=core.ET)
    assert may['model_id'] != september['model_id']
    assert any(pd.Timestamp(at).month in [5, 7, 8] and pd.Timestamp(at).year == 2026 for at in september['train_members'])
    frozen = json.loads((run / 'frozen_control.json').read_text())
    assert frozen['model_id'] == may['model_id']
    controls = pd.read_csv(run / 'frozen_control_predictions.csv')
    assert controls.model_id.nunique() == 1 and controls.model_id.iloc[0] == may['model_id']
    market = core.load_market()
    source = json.loads((run / 'input_manifest.json').read_text())
    assert all(market.manifest[s]['sha256'] == source['market'][s]['sha256'] for s in core.SYMBOLS)
    evaluation = features[(features.date >= protocol['evaluation_start']) & (features.date <= protocol['evaluation_end'])]
    columns = [c for c in features if c not in {'date', 'clock'}]
    for at, stored in evaluation.iterrows():
        row, audit = core.feature_row(market, at)
        assert row is not None, audit
        assert np.allclose([row[c] for c in columns], stored[columns].to_numpy(dtype=float), rtol=0, atol=1e-12)
    total=0; gaps=0; ambiguous=0
    for name,multiple in [('daily_updated',1),('frozen_april_control',1),('double_cost',2)]:
        trades=pd.read_csv(run/f'{name}_trades.csv')
        equity=pd.read_csv(run/f'{name}_equity.csv',parse_dates=['date']).set_index('date')
        assert (equity.cash_settled>=-1e-12).all()
        assert not trades.entry_date.duplicated().any()
        fee=.0005*multiple; slip=.0005*multiple
        closed_profit=0.; open_profit=0.
        last_exit=None
        for t in trades.itertuples():
            at=pd.Timestamp(t.decision_at); start=pd.Timestamp(t.entry_at); finish=pd.Timestamp(t.expiry_at)
            assert start>at+pd.Timedelta(seconds=30)
            assert last_exit is None or start>last_exit
            f=market.bars[t.symbol]
            first=f[f.start_at==start].iloc[0]
            entry=float(first.open)*(1+slip)*float(first.share_factor)
            assert np.isclose(entry,t.entry_fill_economic,rtol=0,atol=1e-9)
            assert np.isclose(t.units_economic*entry*(1+fee),t.capital,rtol=0,atol=1e-10)
            target=math.ceil(first.open*(1+slip)*1.05*100-1e-10)/100*first.share_factor
            stop=entry*(1-protocol['risk']['stop_fraction']); raised=False; pending=None
            path=f[(f.start_at>=start)&(f.end_at<=finish)&(f.session_type=='regular')]
            event=None
            for bar in path.itertuples():
                if pending is not None and bar.start_at > pending[1]:
                    stop=max(stop,pending[0]);raised=True;pending=None
                raw_stop=math.floor(stop/bar.share_factor*100+1e-10)/100
                raw_limit=math.ceil(target/bar.share_factor*100-1e-10)/100
                if bar.low<=raw_stop:
                    event=(bar.end_at,min(bar.open,raw_stop)*(1-slip)*bar.share_factor,'profit_protection' if raised else 'risk_stop')
                    gaps+=int(bar.open<raw_stop); ambiguous+=int(bar.high>=raw_limit)
                    break
                if bar.high>=raw_limit:
                    event=(bar.end_at,raw_limit*bar.share_factor,'target'); break
                trigger=protocol['risk']['breakeven_trigger']
                if trigger is not None and not raised and pending is None and bar.high*bar.share_factor>=entry*(1+trigger):
                    needed=entry*(1+fee)/((1-fee)*(1-slip))
                    pending=(math.ceil(needed/bar.share_factor*100-1e-10)/100*bar.share_factor,bar.available_at+pd.Timedelta(seconds=30))
            if event is None and len(path) and path.end_at.iloc[-1]==finish:
                event=(finish,path.econ_close.iloc[-1]*(1-slip),'expiry')
            if event is not None:
                assert event[0]==pd.Timestamp(t.exit_at),(name,t.symbol,t.entry_at)
                assert event[2]==t.reason
                assert np.isclose(event[1],t.exit_fill_economic,rtol=0,atol=1e-9)
                last_exit=event[0]
            else:
                assert pd.isna(t.exit_at)
            if pd.notna(t.exit_at) and pd.Timestamp(t.exit_at)<=pd.Timestamp('2026-09-30 16:00',tz=core.ET):
                independent_net=(t.exit_fill_economic*(1-fee)+t.dividend_per_economic_unit)/(entry*(1+fee))-1
                assert np.isclose(independent_net,t.net_return,rtol=0,atol=1e-10)
                closed_profit+=t.capital*independent_net
            else:
                mark=float(market.daily[t.symbol].loc[equity.index[-1],'close'])
                open_profit+=t.units_economic*mark+t.accrued_dividends-t.capital
            total+=1
        assert np.isclose(float(equity.equity.iloc[-1]),1+closed_profit+open_profit,rtol=0,atol=1e-9)
        ends=equity.equity.groupby(equity.index.to_period('M')).last()
        monthly=ends/pd.Series([1.,*ends.iloc[:-1].tolist()],index=ends.index)-1
        reported={m['month']:m['return'] for m in results[name]['monthly']}
        assert all(np.isclose(v,reported[str(m)],rtol=0,atol=1e-10) for m,v in monthly.items())
        mean=float(np.mean([monthly[pd.Period(m)] for m in ['2026-05','2026-07','2026-09']]))
        assert np.isclose(mean,results[name]['primary_average'],rtol=0,atol=1e-10)
    out = {'pass': True, 'daily_models_audited': audited_models,
           'all_training_and_calibration_labels_available_before_daily_cutoff': True,
           'all_calibration_forecasts_use_pre_outcome_model_cutoffs': True,
           'historical_replay_not_real_time_logs': True,
           'september_learns_matured_may_july_august_labels': True,
           'frozen_control_uses_may_first_model_only': True,
           'raw_bar_trade_checks': total, 'adverse_gap_exits': gaps,
           'causal_feature_rows_recomputed': len(evaluation),
           'ambiguous_stop_first_exits': ambiguous,
           'monthly_returns_and_terminal_equity_recomputed': True,
           'reported_acceptance': results['acceptance']['passed']}
    core.dump(run / 'verification.json', out)
    print(json.dumps(out, indent=2))
    return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    verify(parser.parse_args().run)

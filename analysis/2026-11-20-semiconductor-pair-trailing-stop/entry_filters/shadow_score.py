"""Veto original five-bar entries while retaining the original shadow position clock."""
import argparse
import json
import math
from pathlib import Path

import numpy as np
from sklearn import __version__ as sklearn_version
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import study
from study import base

HERE = Path(__file__).resolve().parent
PERIODS = {"train": ("2026-07-27", "2026-08-14"), "calibration": ("2026-08-17", "2026-08-28"),
           "september_check": ("2026-09-14", "2026-09-18"), "november": ("2026-11-02", "2026-11-19")}
SUPPLEMENT = {"october_check": ("2026-10-26", "2026-10-30"), "last_four": ("2026-11-16", "2026-11-19")}
study.BLOCKS.update(PERIODS)
study.BLOCKS.update(SUPPLEMENT)
FEATURES = ["return5", "return12", "drawdown5", "drawdown12", "close_efficiency12",
            "close_location12", "inverse_return5"]
QUANTILES = {"q50": .50, "q75": .75, "q90": .90}
PATHS = ("OLHC", "OHLC")


def feature_at(tape, times, i, s):
    if i < 12 or any(times[j+1]-times[j] != base.STEP for j in range(i-12, i)):
        return None
    bars = tape[s][i-12:i]
    fast, slow = base.signal(bars[-5:]), base.signal(bars)
    closes = [b[3] for b in bars]
    travel = sum(abs(r-l) for l,r in zip(closes,closes[1:]))
    high, low = max(b[1] for b in bars), min(b[2] for b in bars)
    other = "SOXS" if s == "SOXL" else "SOXL"
    inv = tape[other][i-5:i]
    return [fast["gain"], slow["gain"], fast["drawdown"], slow["drawdown"],
            abs(closes[-1]-closes[0])/travel if travel else 0.,
            (closes[-1]-low)/(high-low) if high > low else .5, inv[-1][3]/inv[0][0]-1]


def load(name):
    times, days, tape, sources = study.load_period(name)
    ref = base.signals(times, tape, 5)
    index = {ts.isoformat():i for i,ts in enumerate(times)}
    return dict(times=times, days=days, tape=tape, ref=ref, index=index, sources=sources)


def original(data,path):
    summary, trades, _ = base.replay(data["times"],data["days"],data["tape"],data["ref"],path)
    features = [feature_at(data["tape"],data["times"],data["index"][t["signal_end"]],t["symbol"]) for t in trades]
    return summary,trades,features


def portable_scores(rows, model):
    scores = []
    for x in rows:
        if x is None:
            scores.append(None)
        else:
            z = (np.asarray(x) - model["mean"]) / model["scale"]
            scores.append(float(1/(1+np.exp(-(z@model["weights"]+model["intercept"])))))
    return scores


def gated(data,path,shadow,scores,threshold):
    # The shadow clock runs independently of the real portfolio. Suppress only
    # existing shadow signals, fixing both symbol and decision time.
    allowed = {t["signal_end"]:t["symbol"] for t,score in zip(shadow,scores)
               if score is not None and score >= threshold}
    ref = [{s:dict(v,passed=(s==allowed.get(clock.isoformat()))) for s,v in old.items()}
           for clock,old in zip(data["times"],data["ref"])]
    stat,trades,_ = base.replay(data["times"],data["days"],data["tape"],ref,path)
    expected = [t for t in shadow if t["signal_end"] in allowed]
    assert trades == expected
    vetoes = [t for t in shadow if t["signal_end"] not in allowed]
    result = study.summarize(stat,trades)
    result.update(directly_vetoed=study.counted(vetoes),new_entries=0,
                  unscored=sum(x is None for x in scores))
    return result,trades


def fit():
    train = load("train")
    _,trades,features = original(train,"OLHC")
    known = [(x,t) for x,t in zip(features,trades) if x is not None and t["exit_reason"]!="open"]
    X = [x for x,_ in known]
    y = [int(t["net"]>0) for _,t in known]
    scale = StandardScaler().fit(X)
    clf = LogisticRegression(C=.3,solver="lbfgs",max_iter=1000).fit(scale.transform(X),y)
    model = dict(mean=scale.mean_.tolist(),scale=scale.scale_.tolist(),weights=clf.coef_[0].tolist(),
                 intercept=float(clf.intercept_[0]))
    assert np.allclose(portable_scores(X,model),clf.predict_proba(scale.transform(X))[:,1],atol=1e-12)
    cal = load("calibration")
    _,cal_trades,features = original(cal,"OLHC")
    scores = portable_scores(features,model)
    available = [s for s in scores if s is not None]
    thresholds = {key:float(np.quantile(available,level)) for key,level in QUANTILES.items()}
    ranks = {key:gated(cal,"OLHC",cal_trades,scores,t)[0] for key,t in thresholds.items()}
    eligible = [k for k,r in ranks.items() if r["closed"]>=8 and r["coverage"]>=3]
    selected = min(eligible,key=lambda k:(ranks[k]["loss_rate"],-ranks[k]["net"],QUANTILES[k])) if eligible else "q50"
    frozen = dict(periods=PERIODS, features=FEATURES, sklearn=sklearn_version, C=.3, model=model,
                  fit_path="OLHC", train_rows=len(y), train_wins=sum(y), train_buys=len(trades),
                  calibration_buys=len(cal_trades), calibration_scored=len(available),
                  quantiles=QUANTILES,thresholds=thresholds, selection=ranks, selected=selected,
                  selection_fallback=not eligible, train_sources=train["sources"], calibration_sources=cal["sources"])
    target = HERE/"shadow_freeze.json"
    if target.exists() and json.loads(target.read_text()) != json.loads(json.dumps(frozen)):
        raise ValueError("Existing freeze differs; create a new version before changing the protocol")
    study.write(target,frozen)
    print(selected, thresholds)


def evaluate():
    f = json.loads((HERE/"shadow_freeze.json").read_text())
    output = {"selected":f["selected"],"blocks":{}}
    for name in (*PERIODS, *SUPPLEMENT):
        data = load(name)
        tab = output["blocks"][name] = dict(sources=data["sources"], rules={})
        if name in ("train","calibration"):
            assert tab["sources"] == f[name+"_sources"]
        for path in PATHS:
            stat,shadow,features = original(data,path)
            scores = portable_scores(features,f["model"])
            if name == "november" and path == "OLHC":
                cutoff = f["thresholds"][f["selected"]]
                audit = [dict(symbol=t["symbol"], decision=t["decision"], entry=t["entry"],
                              inputs=dict(zip(FEATURES, x)) if x is not None else None,
                              score=score, allow=score is not None and score >= cutoff,
                              subsequent_exit=t["exit_reason"], subsequent_net=t["net"])
                         for t,x,score in zip(shadow,features,scores)]
            tab["rules"].setdefault("original",{})[path] = study.summarize(stat,shadow)
            for k,t in f["thresholds"].items():
                result,trades = gated(data,path,shadow,scores,t)
                tab["rules"].setdefault(k,{})[path] = result
                if name=="november" and k==f["selected"] and path=="OLHC":
                    ledger = dict(original="original",selected=k,trades={"original":shadow,k:trades})
                if name=="calibration" and path=="OLHC":
                    assert result==f["selection"][k]
        # Re-run on finite prefixes. The original eligibility/score at a given
        # decision cannot depend on later bar values or the later shadow state.
        for stop in (len(data["times"])//3,2*len(data["times"])//3):
            small_tape = {s:xs[:stop] for s,xs in data["tape"].items()}
            times,days = data["times"][:stop],data["days"][:stop]
            small = dict(times=times,days=days,tape=small_tape,index=data["index"],ref=base.signals(times,small_tape,5))
            summary, prefix_trades, xs = original(small,"OHLC")
            prefix_scores = portable_scores(xs,f["model"])
            key = lambda tr:(tr["symbol"],tr["entry"])
            assert [key(t) for t in prefix_trades] == [key(t) for t in shadow[:len(prefix_trades)]]
            assert prefix_scores == scores[:len(prefix_trades)]
    study.write(HERE/"shadow_results.json",output)
    study.write(HERE/"shadow_ledger.json",ledger)
    study.write(HERE/"shadow_decisions.json",dict(cutoff=cutoff, decisions=audit))
    for name,tab in output["blocks"].items():
        for k in tab["rules"]:
            s=tab["rules"][k]["OLHC"]
            print(name,k,s["buys"],s["wins"],s["losses"],s["net"],s["max_dd"],s["coverage"],s.get("directly_vetoed"))


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase",choices=["freeze","evaluate"])
    phase=parser.parse_args().phase
    fit() if phase=="freeze" else evaluate()

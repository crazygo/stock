"""Rebuild the November $1,000 strategy from 5-minute bars and check each fill."""
import hashlib
import json
from math import isclose
from pathlib import Path

from study_gates import Context, FEE, PATHS, VIRTUAL, late_account, observe

HERE = Path(__file__).resolve().parent
STUDY = HERE.parents[1]
MODEL_SHA256 = "ed4c7604202031a66a5c969b5e004467f9d11c100ef2784ded806441ff9abe34"


def check(ok, explanation):
    if not ok:
        raise ValueError(explanation)


def main():
    if not __debug__:
        raise RuntimeError("Run without -O so input and accounting checks are enabled.")
    model = STUDY/"entry_filters/shadow_freeze.json"
    check(hashlib.sha256(model.read_bytes()).hexdigest() == MODEL_SHA256, "Frozen model file differs")
    oracle = json.loads((HERE/"size_check.json").read_text())["results"]
    ctx = Context("november")
    expected_inputs = json.loads((STUDY/"results.json").read_text())["inputs"]
    check(ctx.data["sources"] == expected_inputs, "Input rows or their fingerprints differ")
    for path in PATHS:
        paper = ctx.execute(VIRTUAL, path, FEE)
        stages = [paper["summary"]["by_stage"][str(i)]["buys"] for i in (1,2,3)]
        check(stages == ([94,47,9] if path == "OLHC" else [99,47,9]), f"{path} paper stage counts differ")
        events = observe(ctx, paper["lots"])
        allowed = {event["entry_i"] for event in events if event["progress"]}
        routed = [lot for lot in paper["lots"] if lot["stage"] == 3 and lot["entry_i"] in allowed]
        account = late_account(dict(ctx.data, path=path), routed, 3, 1000., initial=1400.)
        expected = oracle[path]["november"]["original_gate_1000_1400_fixed"]
        for key in ("buys","wins","losses","open","cash_blocked","fees","gross","net"):
            check(isclose(account[key], expected[key], rel_tol=0, abs_tol=1e-7), f"{path} {key} differs")
        for trade, stored in zip(account["trades"], expected["trades"], strict=True):
            check((trade["symbol"], trade["start"], trade["reason"]) ==
                  (stored["symbol"], stored["entry"], stored["reason"]), f"{path} trade identity differs")
            for key in ("gross","net"):
                check(isclose(trade[key], stored[key], rel_tol=0, abs_tol=1e-7),
                      f"{path} {trade['start']} {key} differs")
        print(f"PASS {path} paper={stages} buys={account['buys']} "
              f"wins={account['wins']} losses={account['losses']} "
              f"fees={account['fees']:.2f} net={account['net']:.2f}")


if __name__ == "__main__":
    main()

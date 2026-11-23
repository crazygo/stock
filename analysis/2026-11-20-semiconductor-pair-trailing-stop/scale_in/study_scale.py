"""Compare fixed-dollar initial positions and successively confirmed additions."""
import argparse
from dataclasses import asdict, dataclass
from functools import lru_cache
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "entry_filters"))
import shadow_score as score
from engine import run
from study import base

PRIMARY = 2.50
FEES = (PRIMARY, 1.25)
DEVELOP = ("development_a","development_b")
PATHS = ("OLHC","OHLC")
MODEL = json.loads((HERE.parent/"entry_filters/shadow_freeze.json").read_text())
LEVELS = MODEL["thresholds"]


@dataclass(frozen=True)
class Config:
    amounts: tuple
    wait: tuple
    seed: str
    mode: str

    @property
    def label(self):
        return f"{self.seed}_{'-'.join(map(str,self.amounts))}_{'-'.join(map(str,self.wait))}_{self.mode}"

    def record(self):
        return dict(label=self.label, **asdict(self))


def configs():
    for seed in LEVELS:
        yield Config((600,),(),seed,"hold")
        for mode in ("hold","step"):
            for sizes in ((100,500),(200,400),(300,300)):
                for wait in (1,2,3,6):
                    yield Config(sizes,(wait,),seed,mode)
            for sizes in ((100,200,300),(100,150,350),(150,200,250),(200,200,200)):
                for waits in ((1,1),(2,2),(3,3),(6,6),(1,3),(3,6)):
                    yield Config(sizes,waits,seed,mode)


class Context:
    def __init__(self, block):
        d = self.data = score.load(block)
        self.scores = {s:score.portable_scores(
            [score.feature_at(d["tape"],d["times"],i,s) for i in range(len(d["times"]))], MODEL["model"])
            for s in base.SYMBOLS}
        self.seed = {}
        for path in PATHS:
            _, trades, _ = score.original(d,path)
            for level, cut in LEVELS.items():
                allowed = {}
                for t in trades:
                    i = d["index"][t["signal_end"]]
                    s = t["symbol"]
                    v = self.scores[s][i]
                    if v is not None and v >= cut:
                        allowed[i] = s
                self.seed[path,level] = allowed

        @lru_cache(maxsize=None)
        def anchored(s, first, i):
            start = first - 5
            if start < 0 or d["times"][i] - d["times"][start] != (i-start)*base.STEP:
                return False
            return base.signal(d["tape"][s][start:i])["passed"]
        self.anchored = anchored

    def confirm(self,s,first,last,i,stage,cuts):
        d = self.data
        v = self.scores[s][i]
        return (v is not None and v >= cuts[stage-2] and d["ref"][i][s]["passed"]
                and d["tape"][s][i-1][3] > d["tape"][s][last-1][3]
                and self.anchored(s,first,i))

    def execute(self,c,path,fee,whole=False):
        d = self.data
        cut = LEVELS[c.seed]
        cuts = (max(cut,LEVELS["q75"]), max(cut,LEVELS["q90"])) if c.mode == "step" else (cut,cut)
        return run(d["tape"],d["times"],d["days"],self.seed[path,c.seed],self.confirm,
                   c.amounts,c.wait,cuts,path,fee,whole=whole)


def compact(x):
    s=x["summary"]
    return dict(net=s["net_pnl"],gross=s["gross_pnl"],fees=s["fees"],buys=s["buys"],
                sells=s["sells"],open=s["open"],stage=[s["by_stage"][str(i)]["buys"] for i in (1,2,3)],
                days=len(x["daily"]),max_drawdown=s["max_drawdown"],cash_blocked=s["cash_blocked"])


def develop():
    all_cfg = list(configs())
    contexts = {b:Context(b) for b in DEVELOP}
    tables, chosen = {}, {}
    for fee in FEES:
        table = {}
        for c in all_cfg:
            tests = {f"{b}_{p}":compact(ctx.execute(c,p,fee)) for b,ctx in contexts.items() for p in PATHS}
            per_day = [x["net"]/x["days"] for x in tests.values()]
            qualified = all(v["stage"][0] >= 2 for v in tests.values())
            for p in PATHS:
                for stage, minimum in enumerate((0,3,2)[:len(c.amounts)]):
                    qualified &= sum(tests[f"{b}_{p}"]["stage"][stage] for b in DEVELOP) >= minimum
            table[c.label] = dict(config=c.record(),tests=tests,qualified=qualified,
                                  worst_net_day=min(per_day),avg_net_day=sum(per_day)/len(per_day))
        best={}
        for size in (1,2,3):
            pool=[v for v in table.values() if len(v["config"]["amounts"])==size]
            eligible=[v for v in pool if v["qualified"]]
            fallback=[v for v in pool if any(x["stage"][size-1] for x in v["tests"].values())] or pool
            winner=min(eligible or fallback,key=lambda x:(-x["worst_net_day"],-x["avg_net_day"],x["config"]["label"]))
            best[str(size)]={k:winner[k] for k in ("config","qualified","worst_net_day","avg_net_day")}
            best[str(size)]["eligible_count"]=len(eligible)
        tables[str(fee)],chosen[str(fee)]=table,best
    frozen=dict(per_fee=chosen,configs_per_fee=len(all_cfg),sources={b:ctx.data["sources"] for b,ctx in contexts.items()})
    file=HERE/"freeze.json"
    if file.exists() and json.loads(file.read_text()) != json.loads(json.dumps(frozen)):
        raise ValueError("Changed development selection: version the experiment")
    base_write(file,frozen)
    base_write(HERE/"development.json",tables)
    print(json.dumps(chosen,indent=2))


def base_write(path,obj):
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+"\n")


def config_of(record):
    return Config(tuple(record["amounts"]),tuple(record["wait"]),record["seed"],record["mode"])


def evaluated():
    f=json.loads((HERE/"freeze.json").read_text())
    summary,ledger={},{}
    for block in ("november","october_check","last_four"):
        ctx=Context(block)
        by_name={}
        for fee in FEES:
            prefix = "primary" if fee == PRIMARY else "alternative"
            winners={int(k):config_of(v["config"]) for k,v in f["per_fee"][str(fee)].items()}
            c=winners[3]
            cases={f"best{n}":v for n,v in winners.items()}
            cases.update(example=Config((100,200,300),(1,1),"q50","hold"),
                         rising_score_example=Config((100,200,300),(1,1),"q50","step"),
                         best3_equal=Config((200,200,200),c.wait,c.seed,c.mode),
                         example_at_best3=Config((100,200,300),c.wait,c.seed,c.mode),
                         best3_seed_single600=Config((600,),(),c.seed,"hold"),
                         best3_seed_single100=Config((100,),(),c.seed,"hold"),
                         q50_single600=Config((600,),(),"q50","hold"),
                         q75_single600=Config((600,),(),"q75","hold"),
                         q90_single600=Config((600,),(),"q90","hold"),
                         q50_single100=Config((100,),(),"q50","hold"),
                         primary_best3_other_fee=config_of(f["per_fee"][str(PRIMARY)]["3"]["config"]))
            for tag,v in cases.items():
                for path in PATHS:
                    for whole in ((False,True) if tag in ("best3","best3_seed_single600","example") and block=="november" else (False,)):
                        name=f"{prefix}_{tag}_{path}"+("_whole" if whole else "")
                        result=ctx.execute(v,path,fee,whole=whole)
                        record=dict(config=v.record(),path=path,fee=fee,whole=whole,
                                    summary=result["summary"],daily=result["daily"])
                        by_name[name]=record
                        if block=="november" and (tag in ("best3","example","rising_score_example","best3_seed_single600","best1","best2") and
                              (prefix=="primary" or tag=="best3")):
                            ledger[name]=dict(**record,lots=result["lots"])
        summary[block] = dict(sources=ctx.data["sources"],runs=by_name)
    base_write(HERE/"results.json",summary)
    base_write(HERE/"ledger.json",dict(block="november",initial=1000,runs=ledger))
    for name,r in summary["november"]["runs"].items():
        if name.startswith("primary") and r["path"]=="OLHC" and not r["whole"]:
            x=r["summary"]
            print(name,r["config"]["label"],x["buys"],x["sells"],round(x["gross_pnl"],2),round(x["fees"],2),round(x["net_pnl"],2))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase",choices=["develop","evaluate"])
    if parser.parse_args().phase=="develop":
        develop()
    else:
        evaluated()

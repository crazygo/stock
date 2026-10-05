import json
from pathlib import Path
from data import CACHE,build_dataset,load_panel,universe
from model import forward_backtest,evaluate
cfg=json.loads(Path(__file__).with_name('config.json').read_text())
u,meta,excluded=universe();p,m=load_panel(u);d=build_dataset(p,cfg)
out=Path(__file__).parent/'backtest_v1'
pred,a=forward_backtest(d,cfg,out);summary=evaluate(pred,cfg,out)
print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)

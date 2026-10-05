#!/usr/bin/env python3
import argparse,json
from pathlib import Path
from data import universe,refresh_archive,load_panel,build_dataset,CACHE
p=argparse.ArgumentParser();p.add_argument('--refresh',action='store_true');args=p.parse_args()
cfg=json.loads(Path(__file__).with_name('config.json').read_text())
print('archive',json.dumps(refresh_archive() if args.refresh else {'refresh':'skipped'}),flush=True)
u,meta,excluded=universe(args.refresh);print('universe',meta,flush=True)
panel,lineage=load_panel(u);print('panel', {k:v for k,v in lineage.items() if k!='lineage'},flush=True)
d=build_dataset(panel,cfg)
print('dataset',len(d),'eligible',int(d.eligible.sum()),'positive30',int(d.y30.sum()),'positive60',int(d.y60.sum()),flush=True)
(CACHE/'prepare_summary.json').write_text(json.dumps({'universe':meta,'panel':{k:v for k,v in lineage.items() if k!='lineage'},'rows':len(d),'eligible_rows':int(d.eligible.sum())},indent=2))

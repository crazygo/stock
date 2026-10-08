import json, time
from futu import *
import datetime as DT, os
pd_opts = None
q = OpenQuoteContext(host='127.0.0.1', port=11111)
out = {"fetched_utc": DT.datetime.utcnow().isoformat()+"Z", "results": {}}
for kw in ["Constellation Google", "Black Hills Google", "Fervo Google", "Fairfax Boots", "Accel Entertainment", "Golar LNG"]:
    try:
        ret, data = q.get_search_news(kw, 30, news_sub_type=NewsSubType.ALL)
        if ret == RET_OK:
            out["results"][kw] = data.astype(str).to_dict("records")
        else:
            out["results"][kw] = {"error": str(data)}
    except Exception as e:
        out["results"][kw] = {"exception": repr(e)}
    time.sleep(3.5)
q.close()
f=open("nsc_futu_news.json","w"); json.dump(out,f,ensure_ascii=False,indent=1); f.close()
for kw, v in out["results"].items():
    print("==", kw)
    if isinstance(v, dict): print(v); continue
    for r in v[:12]:
        print(r.get("publish_time"), "|", r.get("source"), "|", r.get("news_sub_type"), "|", r.get("title","")[:90], "|", r.get("url","")[:70])

import sys; sys.stdout.flush(); os._exit(0)

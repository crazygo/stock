import json, time, re, subprocess, os, sys
from futu import *
import datetime as DT
from email.utils import parsedate_to_datetime
q = OpenQuoteContext(host='127.0.0.1', port=11111)
track = {}   # title -> {pub, first_wire_seen, futu_first_seen, futu_row}
log = open("nsc_futu_lag.jsonl", "a")
def now(): return DT.datetime.now(DT.timezone.utc)
for it in range(int(sys.argv[1])):
    t0 = now()
    try:
        x = subprocess.run(["curl","-s","--max-time","20","-A","curl/8","https://www.prnewswire.com/rss/news-releases-list.rss"],capture_output=True,text=True).stdout
        items = re.findall(r"<item>.*?<title>(.*?)</title>.*?<pubDate>(.*?)</pubDate>", x, re.S)
    except Exception as e:
        items = []
    for title, pub in items[:6]:
        title = re.sub(r"<!\[CDATA\[|\]\]>","",title).strip()
        if title not in track:
            track[title] = {"pub": parsedate_to_datetime(pub).astimezone(DT.timezone.utc).isoformat(), "wire_first_seen": t0.isoformat(), "futu_first_seen": None}
    pending = [k for k,v in track.items() if v["futu_first_seen"] is None][-8:]
    for title in pending:
        kw = " ".join(re.sub(r"[^\w\s]"," ",title).split()[:6])
        try:
            ret, data = q.get_search_news(kw, 10, news_sub_type=NewsSubType.NEWS)
        except Exception as e:
            ret, data = -1, repr(e)
        hit = None
        if ret == RET_OK and len(data):
            words = set(w.lower() for w in kw.split() if len(w) > 3)
            for r in data.astype(str).to_dict("records"):
                if r["publish_time"] in (now().strftime("%-m/%-d"),) and ("PR Newswire" in r["source"] or "美通社" in r["source"] or sum(w in r["title"].lower() for w in words) >= 2):
                    hit = r; break
        if hit:
            track[title]["futu_first_seen"] = now().isoformat(); track[title]["futu_row"] = hit
        log.write(json.dumps({"poll": it, "t": now().isoformat(), "title": title, "kw": kw, "ret": int(ret) if isinstance(ret,int) else str(ret), "hit": hit}, ensure_ascii=False)+"\n"); log.flush()
        time.sleep(3.2)
    json.dump(track, open("nsc_futu_lag_track.json","w"), ensure_ascii=False, indent=1)
    if it < int(sys.argv[1]) - 1:
        time.sleep(max(0, 180 - (now()-t0).total_seconds()))
q.close(); sys.stdout.flush(); os._exit(0)

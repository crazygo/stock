import json, sys, collections, datetime as DT
def P(s): return DT.datetime.fromisoformat(s) if s else None
for d in sys.argv[1:]:
    rows=[json.loads(l) for l in open(f"{d}/first_seen.jsonl")]
    polls=collections.defaultdict(dict)
    for line in open(f"{d}/poller.log" if __import__("os").path.exists(f"{d}/poller.log") else f"{d}/poller.out"):
        p=line.split()
        if len(p)>=4 and p[0][:4]=='2026':
            k=polls[p[1]]; k[len(k)]=p[0]
    by=collections.defaultdict(list)
    for r in rows:
        if r['poll']>=1 and r['pub_utc']:
            prev=polls[r['feed']].get(r['poll']-1)
            fs=P(r['first_seen_utc']); pub=P(r['pub_utc'])
            ub=(fs-pub).total_seconds()/60; lb=(P(prev)-pub).total_seconds()/60 if prev else None
            by[r['feed']].append((ub,lb,r['title'][:70],r['pub_utc']))
    for f,v in sorted(by.items()):
        ubs=sorted(x[0] for x in v)
        late=[x for x in v if x[1] is not None and x[1]>0]
        print(f"{d}:{f} new_items={len(v)} upper_bound_min median={ubs[len(ubs)//2]:.1f} max={ubs[-1]:.1f} | appeared_after_a_poll_where_already_published(lb>0)={len(late)}" + (f" max_lb={max(x[1] for x in late):.1f}" if late else ""))
        for x in sorted(late,key=lambda x:-x[1])[:3]: print("   late:",f"lb={x[1]:.1f} ub={x[0]:.1f}",x[3],x[2])
    for f,p in polls.items(): print("  polls",f,len(p))

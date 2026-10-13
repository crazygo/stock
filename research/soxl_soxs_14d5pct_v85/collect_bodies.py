"""Fetch the preannouncements found in official indexes, keeping every failure."""
import re
from acquire_earnings import *
ANNOUNCEMENT=re.compile(r'to report|to announce|sets conference call|announces.*(?:earnings|financial conference call)',re.I)

def collect():
    records=[]
    for symbol in HOSTS:
        f=CACHE/(symbol+'_index.json')
        if not f.exists():continue
        for r in json.loads(f.read_text())['records']:
            if ANNOUNCEMENT.search(r['title']):records.append(r)
    def one(r):
        try:
            p=fetch(r['url'])
            return dict(**r,status='html_downloaded_body_review_pending',source={k:v for k,v in p.items() if k not in ['html','text','links']},text=p['text'],links=p['links'])
        except Exception as e:return dict(**r,status='failed',error=repr(e))
    with ThreadPoolExecutor(max_workers=3) as pool:
        for n,r in enumerate(pool.map(one,records)):
            target=CACHE/'bodies';target.mkdir(exist_ok=True)
            dump(target/(hashlib.sha256(r['url'].encode()).hexdigest()[:24]+'.json'),r)
            print(n,r['symbol'],r['published_date'],r['status'],flush=True)

if __name__=='__main__':collect()

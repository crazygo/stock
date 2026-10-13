"""Follow dated issuer press lists to the fixed lower boundary; no search-only coverage claim."""
import re
from datetime import datetime
from acquire_earnings import *
LOW='2022-10-01';HIGH='2026-04-30'
PATTERN=re.compile(r'(earnings|financial results|financial conference|fiscal.*results|results.*fiscal|webcast)',re.I)

def article_rows(html,base,symbol):
    out=[]
    blocks=re.findall(r'<article\b.*?</article>',html,re.S)
    if symbol=='LRCX':blocks=re.findall(r'<li class="wd_item">.*?</li>',html,re.S)
    for block in blocks:
        text,links=parse(block)
        if symbol=='NVDA':
            date=re.search(r'<span class="index-item-text-info-date">(.*?)</span>',block,re.S);assert date
            pub=datetime.strptime(date[1],'%B %d, %Y').strftime('%Y-%m-%d')
        elif symbol=='LRCX':
            date=re.search(r'<div class="wd_date">(.*?)</div>',block,re.S);assert date
            pub=datetime.strptime(date[1],'%b %d, %Y').strftime('%Y-%m-%d')
        else:
            date=re.search(r'<time datetime="([^"]+)"[^>]*>(.*?)</time>',block,re.S);assert date
            pub=date[1][:10]
        relevant=[(urljoin(base,u),t) for u,t in links if t.strip() and ('detail/' in u or symbol in ['LRCX','NVDA'])]
        assert relevant,(symbol,text)
        url,title=max(relevant,key=lambda x:len(x[1]));out.append(dict(symbol=symbol,url=url,title=title,published_date=pub,index_date_text=date[1] if symbol in ['LRCX','NVDA'] else date[2]))
    return out

def crawl(symbol,start_url=None):
    url=start_url or INDEX[symbol];pages=[];records={};seen=set();reached=False
    def checkpoint(error=None):
        out=dict(symbol=symbol,requested_start=LOW,requested_end=HIGH,index_lower_boundary_reached=reached,pages=pages,records=list(records.values()),selected_titles='earnings/financial results/conference/webcast; full index pages preserved',no_model_or_new_capital_outcomes=True,error=error)
        dump(CACHE/(symbol+'_index.json'),out)
        return out
    for i in range(60):
        assert url not in seen,url;seen.add(url)
        try:
            page=fetch(url);rows=article_rows(page['html'],page['final_url'],symbol);assert rows,(symbol,url)
        except Exception as e:
            checkpoint(dict(url=url,message=repr(e)));raise
        pages.append(dict(url=url,sha256=page['sha256'],received_at=page['received_at'],cache_file=page['cache_file'],rows=len(rows),first=rows[0]['published_date'],last=rows[-1]['published_date']))
        for r in rows:
            if LOW<=r['published_date']<=HIGH and PATTERN.search(r['title']):records[r['url']]={**r,'index_url':url}
        checkpoint()
        if min(r['published_date'] for r in rows)<LOW:reached=True;break
        nexts=[urljoin(page['final_url'],u) for u,t in page['links'] if t.startswith('Next') or (symbol in ['LRCX','NVDA'] and t=='»')]
        assert nexts,(symbol,url);url=nexts[0]
    out=checkpoint();print(symbol,len(pages),len(records),reached,flush=True)
    return out

if __name__=='__main__':
    with ThreadPoolExecutor(max_workers=3) as pool:list(pool.map(crawl,['AMD','KLAC','LRCX','INTC']))

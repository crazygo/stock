"""Join captured groups, existing AI screening and immutable daily data; never acquire market data."""
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter
import hashlib
import json
import math
import re
import pandas as pd

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
AI_ROOT=ROOT/'.cache/stock_traits_daily_v1/ai_basket_AplusA/data'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    prices=json.loads((OUT/'prices.json').read_text());base={r['code']:r for r in prices['records']}
    group_path=OUT/'group_capture.json';capture=json.loads(group_path.read_text())
    member_path=ROOT/'analysis/ai_value_chain_map_v1/ai_basket_members.json';members=json.loads(member_path.read_text())
    screen_path=ROOT/'analysis/ai_value_chain_map_v1/quality/screen_current.json';screen=json.loads(screen_path.read_text())
    rows={};excluded=[]
    def add(item,group):
        code=item['code'];kind=item.get('kind',item.get('stock_type','STOCK'))
        if kind not in ['STOCK','ETF'] or not re.fullmatch(r'(US\.[A-Z][A-Z0-9.\-]{0,9}|HK\.\d{5}|JP\.\d{4})',code):
            excluded.append({'code':code,'name':item.get('name'),'group':group,'reason':'非股票/基金或未识别证券代码'});return
        r=rows.setdefault(code,{'code':code,'name':item.get('name') or code,'kind':kind,'groups':[],
                                'listing_date':item.get('listing_date'),'quality':None})
        if item.get('kind') or item.get('stock_type'):r['kind']=kind
        if group not in r['groups']:r['groups'].append(group)
        if item.get('listing_date'):r['listing_date']=item['listing_date']
    for item in capture['holdings']:add(item,'holdings')
    for name,key in [('全部','watchlist'),('特别关注','special')]:
        for item in capture['watchlists'].get(name,[]):add(item,key)
    for item in members['members']:
        rating=screen['records'].get(item['code'],{});grade=rating.get('grade')
        if grade not in ['A+','A']:continue
        add(item,'ai_plus' if grade=='A+' else 'ai_a')
        if item['code'] in rows:
            rows[item['code']]['quality']={k:rating.get(k) for k in ['grade','verified_grade','status','as_of','formal_quality_as_of','expires_at']}
    # Preserve the original map members, even if a group was edited after its capture.
    for r in json.loads((OUT/'results.json').read_text())['records']:
        if r['code'] not in rows:rows[r['code']]={'code':r['code'],'name':r['name'],'kind':r['kind'],'groups':[],
                                               'listing_date':r.get('listing_date'),'quality':None,'original_map_only':True}
        else:rows[r['code']]['kind']=r['kind']
    manifests=[];candidates={}
    for pointer in [ROOT/'.cache/stock_data_v1/current.json',AI_ROOT/'current.json']:
        if not pointer.exists():continue
        ref=json.loads(pointer.read_text());rel=ref.get('manifest') or ref.get('manifest_path')
        if not rel:continue
        path=ROOT/rel
        if not path.exists():path=pointer.parent/rel
        if not path.exists():raise ValueError('Published manifest is missing: '+str(path))
        expected=ref.get('sha256') or ref.get('manifest_sha256')
        if expected and sha(path)!=expected:raise ValueError('Manifest hash mismatch: '+str(path))
        m=json.loads(path.read_text());manifests.append({'path':str(path.relative_to(ROOT)),'sha256':sha(path),
              'data_run_id':m['data_run_id'],'status':m.get('status'),'cutoff':m.get('requested_cutoff')})
        ds=m.get('datasets',{});ds=list(ds.values()) if isinstance(ds,dict) else ds
        for item in ds:
            if not isinstance(item,dict):continue
            source_path=item.get('path') or item.get('local_path')
            if not source_path or not source_path.endswith('.parquet') or 'daily/' not in source_path:continue
            code=item.get('security_id') or item.get('code') or Path(source_path).stem
            if code not in rows or code in base:continue
            basis=str(item.get('price_basis',''))+str(item.get('autype',''))
            if 'QFQ' not in basis and 'qfq' not in basis:continue
            candidates[code]={**item,'path':source_path,'data_run_id':m['data_run_id']}
    supplement_path=OUT/'comparison_acquisition.json'
    if supplement_path.exists():
        supplement=json.loads(supplement_path.read_text())
        manifests.append({'path':str(supplement_path.relative_to(ROOT)),'sha256':sha(supplement_path),
                          'data_run_id':supplement['data_run_id'],'status':supplement['status'],
                          'cutoff':supplement['requested_cutoff']})
        for item in supplement['datasets']:
            code=item['security_id']
            if code not in rows or code in base:continue
            if item.get('autype')!='QFQ':raise ValueError('Supplement must be daily QFQ: '+code)
            candidates[code]={**item,'data_run_id':supplement['data_run_id']}
            info=item.get('basic_info') or {}
            if info.get('stock_type') in ['STOCK','ETF']:rows[code]['kind']=info['stock_type']
            if re.fullmatch(r'\d{4}-\d{2}-\d{2}',info.get('listing_date','')) and info['listing_date']!='1970-01-01':
                rows[code]['listing_date']=info['listing_date']
    extra=[];provenance=[]
    for code,r in rows.items():
        if code in base:
            r['price_ref']='original';continue
        market=code.split('.')[0];dataset=candidates.get(code)
        projected={'code':code,'market':market,'currency':{'US':'USD','HK':'HKD','JP':'JPY'}[market],
                   'listing_date':r.get('listing_date'),'bars':[],'status':'unavailable','invalid_days':[],
                   'calendar_unverified_bars':0,'first_day':None,'last_day':None,'missing_sessions':[]}
        if dataset:
            path=ROOT/dataset['path'];digest=sha(path);expected=dataset.get('file_sha256') or dataset.get('sha256')
            if not expected or expected!=digest:raise ValueError('Dataset hash mismatch: '+code)
            frame=pd.read_parquet(path)
            if 'day' not in frame:frame['day']=frame['session_date'] if 'session_date' in frame else frame['time_key'].astype(str).str[:10]
            frame=frame.sort_values('day')
            if frame['day'].duplicated().any():raise ValueError('Duplicate daily dates: '+code)
            for b in frame.to_dict('records'):
                day=str(b['day'])[:10]
                if day<prices['mini_start'] or day>prices['as_of'] or (r.get('listing_date') and day<r['listing_date']):continue
                o,h,l,c=[float(b[k]) for k in ['open','high','low','close']]
                if not all(math.isfinite(v) and v>0 for v in [o,h,l,c]) or h<max(o,l,c) or l>min(o,h,c):
                    projected['invalid_days'].append(day);continue
                projected['bars'].append([day,o,h,l,c])
            projected.update(status='available' if projected['bars'] else 'empty',path=dataset['path'],sha256=digest,
                             source=dataset.get('source'),price_basis=dataset.get('price_basis'),data_run_id=dataset['data_run_id'])
            if projected['bars']:projected.update(first_day=projected['bars'][0][0],last_day=projected['bars'][-1][0])
            cal=prices['calendars'].get(market)
            projected['calendar_unverified_bars']=sum(not cal or b[0]<cal['start'] or b[0]>cal['end'] for b in projected['bars'])
            provenance.append({'code':code,'path':dataset['path'],'sha256':digest,'data_run_id':dataset['data_run_id']})
        extra.append(projected);r['price_ref']='extra'
    result={'version':'multi_stock_groups_v1','built_at':datetime.now(timezone.utc).isoformat(),'as_of':prices['as_of'],
            'group_observed_at':capture['observed_at'],'quality_as_of':screen['as_of'],'quality_run_id':screen['run_id'],
            'membership_not_PIT':True,'default_calendar_days':60,'group_errors':capture['errors'],
            'records':list(rows.values()),'extra_prices':extra,'excluded':excluded,
            'inputs':[{'path':str(p.relative_to(ROOT)),'sha256':sha(p)} for p in [group_path,member_path,screen_path,OUT/'prices.json']]+manifests,
            'provenance':provenance,'summary':{'members':len(rows),'group_counts':dict(Counter(g for r in rows.values() for g in r['groups'])),
            'with_prices':sum(bool(base[c]['bars']) if c in base else bool(next(p['bars'] for p in extra if p['code']==c)) for c in rows),
            'excluded_non_securities':len(excluded),'published_AI_manifest':(AI_ROOT/'current.json').exists()}}
    (OUT/'comparison.json').write_text(json.dumps(result,ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n')
    print(json.dumps(result['summary'],ensure_ascii=False))


if __name__=='__main__':main()

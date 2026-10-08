"""Build the offline review from existing immutable captures; never fit a predictor."""
from pathlib import Path
import hashlib,json
from datetime import datetime, timezone
OUT=Path(__file__).resolve().parent
SOURCE=OUT.parent/'operation_cadence_v1'
def read(name):return json.loads((SOURCE/name).read_text())
original=read('results.json')
members=read('membership.json')
records=[]
for row in original['records']:
    f=dict(row['fundamental'])
    for key in ['long_eligibility','training_included','grade','screen_grade','grade_type','status']:
        f.pop(key,None)
    records.append({k:row[k] for k in ['code','name','kind','listing_date','coverage']}|{'fundamental':f})
files=['prices.json','intraday.json','membership.json','acquisition.json','results.json','time_contract_probe.json']
manifest={name:{'path':str((SOURCE/name).relative_to(OUT.parents[1])),
    'sha256':hashlib.sha256((SOURCE/name).read_bytes()).hexdigest()} for name in files}
payload={'prices':read('prices.json'),'intraday':read('intraday.json'),'members':members,'records':records,'manifest':manifest,
    'version':'historical_path_description_v3','model_sha256':hashlib.sha256((OUT/'core.js').read_bytes()).hexdigest(),
    'generated_at':datetime.now(timezone.utc).isoformat()}
detail_data=json.dumps(payload,ensure_ascii=False,separators=(',',':'),allow_nan=False).replace('</','<\\/')
detail=(OUT/'detail_template.html').read_text().replace('__CORE__',(OUT/'core.js').read_text()).replace('__DATA__',detail_data)
(OUT/'detail.html').write_text(detail)
# The default comparison wall needs daily prices only. Hourly evidence stays in detail.html.
wall_payload={k:v for k,v in payload.items() if k!='intraday'}
wall_payload['view_version']='historical_comparison_wall_v3'
wall_payload['price_renderer_sha256']=hashlib.sha256((OUT/'price_overlay.js').read_bytes()).hexdigest()
text=json.dumps(wall_payload,ensure_ascii=False,separators=(',',':'),allow_nan=False).replace('</','<\\/')
page=(OUT/'comparison_template.html').read_text().replace('__CORE__',(OUT/'core.js').read_text()).replace('__PRICE_OVERLAY__',(OUT/'price_overlay.js').read_text()).replace('__SLOPE_OVERLAY__',(OUT/'slope_overlay.js').read_text()).replace('__DATA__',text)
(OUT/'index.html').write_text(page)
(OUT/'input_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'bytes':len(page.encode()),'detail_bytes':len(detail.encode()),'members':len(records),'version':payload['version'],'view':wall_payload['view_version']}))

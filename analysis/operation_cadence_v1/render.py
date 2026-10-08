"""Self-contained offline artifact. No network dependency at render/view time."""
from pathlib import Path
import json
OUT=Path(__file__).resolve().parent
payload={k:json.loads((OUT/(fn+'.json')).read_text()) for k,fn in [('results','results'),('prices','prices'),('evaluation','evaluation')]}
ip=OUT/'intraday.json';payload['intraday']=json.loads(ip.read_text()) if ip.exists() else {'records':{}}
data=json.dumps(payload,ensure_ascii=False,separators=(',',':'),allow_nan=False).replace('</','<\\/')
page=(OUT/'template.html').read_text().replace('__DATA__',data)
(OUT/'index.html').write_text(page)
print(json.dumps({'bytes':len(page.encode()),'securities':len(payload['results']['records']),'hourly_securities':len(payload['intraday']['records'])}))

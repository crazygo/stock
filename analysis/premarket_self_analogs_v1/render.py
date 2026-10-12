from pathlib import Path
import json
p=Path(__file__).resolve().parent
s=(p/'report.template.html').read_text();obj=json.loads((p/'results.json').read_text())
shape=json.loads((p/'shape_match_v2.json').read_text())
obj['shape_match_v2']={'protocol':shape['protocol'],'stocks':{}}
for stock in obj['stocks']:
 if stock['code'] in shape['stocks']:
  scored=shape['stocks'][stock['code']]
  obj['shape_match_v2']['stocks'][stock['code']]={key:value for key,value in scored.items() if key!='samples'}
 for sample in stock.get('samples',[]):
  if stock['code'] in shape['stocks']:
   sample['shape']=shape['stocks'][stock['code']]['samples'][sample['date']]
  sample.pop('feature_keys',None);sample.pop('future_24h_comparable',None)
  if sample.get('forward_3_sessions'):
   sample['forward_3_sessions'].pop('keys',None);sample['forward_3_sessions'].pop('missing_keys',None)
  if sample.get('future_rth'):
   for key in ['curve','labels','ohlc']:sample['future_rth'].pop(key,None)
data=json.dumps(obj,ensure_ascii=False,separators=(',',':')).replace('</','<\\/')
(p/'index.html').write_text(s.replace('__DATA__',data))
print('rendered',p/'index.html',(p/'index.html').stat().st_size)

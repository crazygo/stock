"""Render a saved snapshot offline; no acquisition or calculation."""
from pathlib import Path
import hashlib,json
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    pointer=json.loads((HERE/'REPORT_POINTER.json').read_text());p=ROOT/pointer['render_data'];assert sha(p)==pointer['sha256'];data=json.loads(p.read_text());html=(HERE/'template.html').read_text().replace('__DATA__',json.dumps(data,ensure_ascii=False,separators=(',',':'),allow_nan=False).replace('</','<\\/')).replace('__CHART__',(HERE/'chart.js').read_text());tmp=HERE/'index.html.tmp';tmp.write_text(html);tmp.replace(HERE/'index.html');d={'report_run_id':pointer['report_run_id'],'data_run_id':pointer['data_run_id'],'html_sha256':sha(HERE/'index.html'),'render_data_sha256':sha(p),'template_sha256':sha(HERE/'template.html'),'chart_sha256':sha(HERE/'chart.js'),'bytes':len(html.encode())};(HERE/'HTML_MANIFEST.json').write_text(json.dumps(d,indent=2)+'\n');print(json.dumps(d,indent=2))
if __name__=='__main__':main()

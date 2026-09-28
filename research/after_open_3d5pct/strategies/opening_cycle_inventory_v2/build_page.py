"""Export a single offline HTML evidence explorer from the frozen inventory."""
import argparse,base64,gzip,hashlib,json
from pathlib import Path

HERE=Path(__file__).resolve().parent

def build(run, replace_html=False):
    raw=(run/'evidence.json').read_bytes()
    packed=base64.b64encode(gzip.compress(raw,compresslevel=9,mtime=0)).decode()
    template=(HERE/'page_template.html').read_text()
    html=template.replace('__EVIDENCE_GZIP__',packed).replace('__SOURCE_SHA__',hashlib.sha256(raw).hexdigest())
    path=run/'index.html'
    if path.exists() and not replace_html:raise FileExistsError(path)
    path.write_text(html)
    print(json.dumps({'html':str(path),'html_bytes':path.stat().st_size,'evidence_bytes':len(raw)}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--replace-html',action='store_true');a=p.parse_args();build(a.run.resolve(),a.replace_html)

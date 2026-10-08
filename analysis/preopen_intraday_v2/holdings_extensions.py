"""Supplement a captured universe with full issuer XBI and GRID holdings."""
from urllib.request import Request,urlopen
from html.parser import HTMLParser
from pathlib import Path
import io,json,hashlib,re,zipfile,xml.etree.ElementTree as ET
OUT=Path(__file__).resolve().parent
class Rows(HTMLParser):
    def __init__(self):super().__init__();self.rows=[];self.row=[];self.cell=None
    def handle_starttag(self,t,a):
        if t=='tr':self.row=[]
        if t in ('td','th'):self.cell=''
    def handle_data(self,s):
        if self.cell is not None:self.cell+=s
    def handle_endtag(self,t):
        if t in ('td','th') and self.cell is not None:self.row.append(' '.join(self.cell.split()));self.cell=None
        if t=='tr' and self.row:self.rows.append(self.row)
def xlsx_rows(raw):
    z=zipfile.ZipFile(io.BytesIO(raw));ns={'m':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'};shared=[]
    if 'xl/sharedStrings.xml' in z.namelist():
        for si in ET.fromstring(z.read('xl/sharedStrings.xml')).findall('m:si',ns):shared.append(''.join(t.text or '' for t in si.findall('.//m:t',ns)))
    rows=[]
    for row in ET.fromstring(z.read('xl/worksheets/sheet1.xml')).findall('.//m:row',ns):
        vals=[]
        for c in row.findall('m:c',ns):
            col=re.sub(r'\d','',c.attrib['r']);idx=0
            for l in col:idx=idx*26+ord(l)-64
            while len(vals)<idx:vals.append('')
            v=c.find('m:v',ns);v=v.text if v is not None else ''.join(t.text or '' for t in c.findall('.//m:t',ns))
            vals[idx-1]=shared[int(v)] if c.attrib.get('t')=='s' else v
        rows.append(vals)
    return rows
def main():
    p=OUT/'universe.json';u=json.loads(p.read_text())
    for fund,url in [('XBI','https://www.ssga.com/library-content/products/fund-data/etfs/us/holdings-daily-us-en-xbi.xlsx'),('GRID','https://www.ftportfolios.com/Retail/Etf/EtfHoldings.aspx?Ticker=GRID')]:
        try:
            with urlopen(Request(url,headers={'User-Agent':'Mozilla/5.0'}),timeout=15) as r:raw=r.read()
            if fund=='XBI':
                rows=xlsx_rows(raw);i=next(i for i,row in enumerate(rows) if 'Ticker' in row);idx=rows[i].index('Ticker');tickers=[r[idx] for r in rows[i+1:] if len(r)>idx];header=rows[:i]
            else:
                parser=Rows();parser.feed(raw.decode('utf8','replace'));i=next(i for i,r in enumerate(parser.rows) if r[:2]==['Security Name','Identifier']);rows=[r for r in parser.rows[i+1:] if len(r)==7 and re.fullmatch(r'[A-Z0-9]{9}',r[2])]
                plain=' '.join(re.sub(r'<[^>]+>',' ',raw.decode('utf8','replace')).split());expected=int(re.search(r'Total Number of Holdings \(excluding cash\):\s*(\d+)',plain).group(1))
                assert len(rows)>=expected;tickers=[r[1] for r in rows];header=['full issuer HTML table',len(rows)]
            members=sorted({t for t in tickers if re.fullmatch(r'[A-Z]{1,6}',t) and t not in ('USD','CASH','SXX')})
            assert len(members)>10
            u['funds'][fund]={'status':'full_issuer_table_us_equities' if fund=='GRID' else 'full_issuer_file_us_equities','source_url':url,'raw_sha256':hashlib.sha256(raw).hexdigest(),'source_header':header,'members':members}
            for s in members:
                m=u['members'].setdefault(s,{'symbol':s,'groups':[],'name':s})
                if 'ETF成分:'+fund not in m['groups']:m['groups'].append('ETF成分:'+fund)
            print(json.dumps({'fund':fund,'members':len(members)}),flush=True)
        except Exception as e:
            u['funds'][fund].setdefault('extension_errors',[]).append(type(e).__name__+': '+str(e)[:180]);print(fund,type(e).__name__,flush=True)
    p.write_text(json.dumps(u,ensure_ascii=False,indent=2))
if __name__=='__main__':main()

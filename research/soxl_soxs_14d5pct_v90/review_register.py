"""Values transcribed after reading all current-quarter table/guidance packets.

The literal ledger below is checked against both v89 extraction and separate raw
HTML cell parsing. It is not generated from the values it is meant to check.
"""
from prepare_review import *
from datetime import date,timedelta,time
from zoneinfo import ZoneInfo

# USD million. Each sequence follows the issuer's dated regular releases only.
ACTUALS={
 'NVDA':[6704,5931,6051,7192,13507,18120,22103,26044,30040,35082,39331,44062,46743,57006,68127],
 'AMD':[6550,5565,5599,5353,5359,5800,6168,5473,5835,6819,7658,7438,7685,9246,10270],
 'AVGO':[8464,8930,8915,8733,8876,9295,11961,12487,13072,14054,14916,15004,15952,18015,19311],
 'MU':[6643,4085,3693,3752,4010,4726,5824,6811,7750,8709,8053,9301,11315,13643,23860],
 'AMAT':[6520,6749,6739,6630,6425,6723,6707,6646,6778,7045,7166,7100,7302,6800,7012],
 'LRCX':[4635.554,5074.121,5277.569,3869.569,3207.257,3482.062,3758.259,3793.558,3871.507,4167.976,4376.047,4720.175,5171.393,5324.173,5344.791,5841.488],
 'KLAC':[2486.739,2724.424,2983.887,2432.608,2355.137,2396.956,2486.726,2359.830,2568.735,2841.541,3076.851,3063.029,3174.741,3209.696,3297.146,3415.078],
 'INTC':[15338,14042,11715,12949,14158,15406,12724,12833,13284,14260,12667,12859,13653,13674,13577],
}
CENTERS={
 'NVDA':[5900,6000,6500,11000,16000,20000,24000,28000,32500,37500,43000,45000,54000,65000,78000],
 'AMD':[6700,5500,5300,5300,5700,6100,5400,5700,6700,7500,7100,7400,8700,9600,9800],
 'AVGO':[8900,8900,8700,8850,9270,50000,50000,51000,14000,14600,14900,15800,17400,19100,22000],
 'MU':[4250,3800,3700,3900,4400,5300,6600,7600,8700,7900,8800,10700,12500,18700,33500],
 'AMAT':[6650,6700,6400,6150,6510,6470,6500,6650,6930,7150,7100,7200,6700,6850,7650],
 'LRCX':[4900,5100,3800,3100,3400,3700,3700,3800,4050,4300,4650,5000,5200,5200,5700,6600],
 'KLAC':[2600,2800,2350,2250,2350,2450,2300,2500,2750,2950,3000,3075,3150,3225,3350,3575],
 'INTC':[14500,11000,12000,13400,15100,12700,13000,13000,13800,12200,11800,13100,13300,12200,14300],
}
WIDTHS={
 'NVDA':[v*.02 for v in CENTERS['NVDA']], 'AMD':[200]+[300]*14,
 'AVGO':[0]*15, 'MU':[250]+[200]*10+[300,300,400,750],
 'AMAT':[400]*11+[500]*4, 'LRCX':[300]*15+[400],
 'KLAC':[125,150,150,125,125,125,125,125]+[150]*7+[200],
 'INTC':[500,500,500,500,500,500,500,500,500,500,600,500,500,500,500],
}
FIRST_PERIOD={'NVDA':'2023Q2','AMD':'2022Q2','AVGO':'2022Q3','MU':'2022Q4',
              'AMAT':'2022Q3','LRCX':'2022Q4','KLAC':'2022Q4','INTC':'2022Q3'}
# These are conservative unknowns identified in the result release, without
# looking at strategy returns or applying knowledge learned in later filings.
SCOPE_UNRESOLVED={
 ('AMD','2023Q3'):'Newly acquired AI software businesses in release; timing/contribution relative to the scheduled outlook not reconciled.',
 ('AMD','2024Q3'):'Silo AI acquisition reported; no same-scope revenue bridge against scheduled outlook in this release.',
 ('AMD','2025Q2'):'ZT manufacturing is held for sale/discontinued; prior release does not explicitly establish the same guidance revenue boundary.',
 ('INTC','2025Q3'):'Altera deconsolidated during quarter; prior outlook does not specify a comparable revenue bridge.',
}


def advance(period,n):
    y,q=map(int,period.split('Q'));v=4*y+q-1+n
    return f'{v//4}Q{v%4+1}'


def selected_table(packet):
    s=packet['symbol'];labels={'NVDA':['revenue'],'AMD':['net revenue'],
        'AVGO':['net revenue'],'MU':['revenue'],'AMAT':['net sales','net revenue','revenue'],
        'LRCX':['revenue'],'KLAC':['total revenues'],'INTC':['net revenue']}[s]
    for table in packet['revenue_tables']:
        if table['cells'][0].lower() not in labels:continue
        # KLA's summary is rounded to millions. Read the statements in thousands.
        if s=='KLAC' and table['table']!=2:continue
        for index,cell in enumerate(table['cells'][1:],1):
            m=re.fullmatch(r'\$?\s*([\d,]+(?:\.\d+)?)',cell)
            if m:return table,index,float(m[1].replace(',',''))/(1000 if s in ['LRCX','KLAC'] else 1)
    raise ValueError(('No reviewed financial table',packet['id']))


def build():
    old=read(V89/'evidence/revenue_candidates.json');rows=[];versions=[];inputs={}
    for symbol in ACTUALS:
        source=sorted([r for r in old['records'] if r['symbol']==symbol and r['candidate_kind']=='quarterly_results'],key=lambda r:r['published_date'])
        assert len(source)==len(ACTUALS[symbol])==len(CENTERS[symbol])==len(WIDTHS[symbol])
        for i,r in enumerate(source):
            packet_path=CACHE/'review_packets'/(r['id']+'.json');packet=read(packet_path)
            doc=read(ROOT/r['document_file']);table,column,value=selected_table(packet)
            expected=ACTUALS[symbol][i];period=advance(FIRST_PERIOD[symbol],i)
            assert value==expected==r['actual_candidate']['revenue_million']
            assert period==r['actual_candidate']['fiscal_period']
            center= CENTERS[symbol][i];width=WIDTHS[symbol][i];g=r['guidance_candidate']
            annual=symbol=='AVGO' and i in [5,6,7]
            assert g['scope']==('annual' if annual else 'quarter')
            assert g['fiscal_period']==('2024' if annual else advance(period,1))
            assert abs(g['range_million']['lower']-(center-width))<1e-9
            assert abs(g['range_million']['upper']-(center+width))<1e-9
            a=r['actual_candidate'];prefix=doc['text'][max(0,a['evidence_start']-1500):a['evidence_start']]
            assert re.search(a['unit_assumption'],prefix,re.I) and packet['publication_matches']
            row=dict(id=r['id'],symbol=symbol,url=r['url'],published_date=r['published_date'],
                source_sha256=r['source_sha256'],source_file=r['source_file'],received_at=doc['received_at'],
                actual_period=period,actual_revenue_million=expected,
                guidance_period='2024' if annual else advance(period,1),guidance_scope='annual' if annual else 'quarter',
                guidance_lower_million=center-width,guidance_upper_million=center+width,guidance_midpoint_million=center,
                actual_table=table['table'],actual_table_row=table['row'],actual_value_cell=column,
                table_cell_value=value,raw_unit=a['unit_assumption'],currency='USD',
                date_evidence_start=packet['publication_matches'][0]['start'],
                numeric_review='all_current_quarter_headers_cells_units_and_guidance_read',
                scope_status='unresolved' if (symbol,period) in SCOPE_UNRESOLVED else 'reported_issuer_revenue_proxy',
                scope_note=SCOPE_UNRESOLVED.get((symbol,period),'As-reported company revenue versus scheduled company outlook; no organic-growth or pure-semiconductor interpretation.'),
                historical_archive_only=True,complete_revision_history=False)
            row['available_at']=datetime.combine(date.fromisoformat(r['published_date'])+timedelta(days=2),time(),ZoneInfo('America/New_York')).isoformat()
            rows.append(row)
            inputs[str(packet_path.relative_to(ROOT))]=sha(packet_path)
            for f,h in [(r['source_file'],r['source_sha256']),(r['document_file'],r['document_sha256'])]:
                assert sha(ROOT/f)==h;inputs[f]=h
    for event in read(V89/'review_cases.json')['updates']:
        r=next(r for r in old['records'] if r['id']==event['id']);packet=read(CACHE/'review_packets'/(r['id']+'.json'))
        assert packet['publication_matches'] and sha(ROOT/r['source_file'])==r['source_sha256']
        d=date.fromisoformat(event['published_date'])+timedelta(days=2)
        versions.append(dict(event,available_at=datetime.combine(d,time(),ZoneInfo('America/New_York')).isoformat(),
            semantic_use='Known special release marker; original v89 no-training flag retained; not a final actual or automatic numeric replacement.'))
        for f,h in [(r['source_file'],r['source_sha256']),(r['document_file'],r['document_sha256'])]:inputs[f]=h
    result=dict(stage='reviewed_historical_archive_proxy_records',records=rows,special_events=versions,
        review_scope='All122quarterly numerical tables, fiscal periods, revenue guidance,128visible publication dates;4known scope questions retained as unknown. Fixed issuer reported-revenue proxy, not consensus or pure-sector demand.',
        input_sha256=inputs,models_trained=False,complete_revision_history=False,historical_real_receipt=False)
    dump(SOURCE/'evidence/reviewed_records.json',result)
    print({'quarter_records':len(rows),'special_events':len(versions),'scope_unknown':sum(r['scope_status']=='unresolved' for r in rows),'raw_input_hashes':len(inputs)})


if __name__=='__main__':build()

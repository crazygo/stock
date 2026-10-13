"""Small provider sample quality audit; no complete-history acceptance claim."""
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2];CACHE=ROOT/'.cache/soxl_soxs_14d5pct_v75'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    receipt=json.loads((CACHE/'older_probe_receipt.json').read_text());provider=receipt['OpenD'];assert provider['ok'] and provider['pagination_complete']
    frames=[]
    for row in provider['pages']:
        p=ROOT/row['path'];assert sha(p)==row['sha256'];frames.append(pd.read_parquet(p))
    bars=pd.concat(frames,ignore_index=True);t=pd.to_datetime(bars.time_key);assert not t.duplicated().any()
    assert t.is_monotonic_increasing and t.dt.strftime('%Y-%m-%d').between('2023-04-03','2023-04-05').all()
    values=bars[['open','high','low','close','volume','turnover']].to_numpy(float);assert np.isfinite(values).all()
    assert (bars[['open','high','low','close']]>0).all().all() and (bars[['volume','turnover']]>=0).all().all()
    assert bars.high.ge(bars[['open','close','low']].max(axis=1)).all() and bars.low.le(bars[['open','close','high']].min(axis=1)).all()
    counts={}
    for day in ['2023-04-03','2023-04-04','2023-04-05']:
        expected=pd.date_range(day+' 09:30',day+' 15:55',freq='5min');got=pd.DatetimeIndex(t[(t.dt.strftime('%Y-%m-%d')==day)&t.dt.strftime('%H:%M').between('09:30','15:55')])
        assert got.equals(expected);counts[day]=len(got)
    clock=t.dt.strftime('%H:%M');session=np.select([clock.lt('04:00')|clock.ge('20:00'),clock.lt('09:30'),clock.lt('16:00')],['overnight','pre_market','regular'],default='post_market')
    detail={}
    for name in sorted(set(session)):
        b=bars.loc[session==name];detail[name]={'bars':len(b),'zero_volume':int(b.volume.eq(0).sum()),'flat_ohlc':int(b.high.eq(b.low).sum())}
    output={'passed':True,'scope':'Three-date structural and regular timestamp sample only; authenticity of full older archive, corporate actions, session availability and broader coverage remain unverified',
            'rows':len(bars),'regular_exact_grid':counts,'sessions':detail,'receipt_sha256':sha(CACHE/'older_probe_receipt.json'),'source_sha256':sha(Path(__file__)),'new_model_trained':False}
    (CACHE/'older_probe_audit.json').write_text(json.dumps(output,indent=2)+'\n');print(output,flush=True)


if __name__=='__main__':main()

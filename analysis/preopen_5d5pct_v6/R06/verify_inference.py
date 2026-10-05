"""Exercise the unified R06 command with pickle loading forbidden."""
import sys,json,io,pickle
from pathlib import Path
from contextlib import redirect_stdout
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import OUT,write,now
import recommend

def main():
    original=pickle.loads
    def reject(*args,**kwargs):raise AssertionError('Inference must use verified native models')
    pickle.loads=reject;sys.argv=['recommend.py','--round','R06','--top','3','--format','json'];stream=io.StringIO()
    try:
        with redirect_stdout(stream):recommend.main()
    finally:pickle.loads=original
    r=json.loads(stream.getvalue());assert not r['current_probability'] and not r['issued_signal'] and not r['orders_sent'];assert len(r['routes'])==8
    for route in r['routes']:
        assert route['artifact_format']=='data_only_native';assert len(route['options'])==3
        for option in route['options']:assert option['effective_threshold'] is None and not option['issued_signal'] and option['window_end'] and option['target_price']>option['evaluation_price']
    write(OUT/'R06/reference_latest.json',r);write(OUT/'R06/inference_verification.json',dict(at=now(),status='passed',routes=8,options=24,pickle_loads_used=0,current_probability=False,issued_signal=False,note='Historical reference and portability only, no effective recommendation admission.'))
    print(json.dumps(dict(status='passed',routes=8,options=24)))

if __name__=='__main__':main()

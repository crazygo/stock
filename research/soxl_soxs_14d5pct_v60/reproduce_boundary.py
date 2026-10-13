"""Reproduce the first audit disagreement using decimal arithmetic and old scalar exits."""
import argparse,gzip,pickle
from decimal import Decimal,localcontext
from path_learning import *
from risk import first_exit


def reproduce(run):
    run=Path(run);b=read(SOURCE/'evidence/first_mismatch.json');at=pd.Timestamp(b['at']);rec=read(run/'training'/(tag(at)+'.json'));info=rec['fits'][b['kind']];bank=dict(np.load(run/'path_bank.npz'));manifest=read(run/'input_manifest.json');assert sha(run/'path_bank.npz')==manifest['derived_files_sha256']['path_bank.npz']
    with gzip.open(run/'models'/(b['kind']+'_'+tag(at)+'.pkl.gz'),'rb') as f:m=pickle.load(f)
    sample=b['path'];symbol=b['symbol'];selected=m['selected_blocks'][sample];c=CLOCKS.index(rec['clock']);p=q=info['query_quote'][symbol]['raw_close'];frames=[];rows=[];factor=1.
    with localcontext() as context:
        context.prec=60;exact=Decimal(str(p))
        for day,length in enumerate(info['plan']['bar_counts']):
            i=bank['blocks'][selected[day//2],day%2];start=info['plan']['first_offset'] if day==0 else 0;den=bank['quotes'][i,c,symbol] if day==0 else bank['previous_close'][i,symbol];src=bank['ohlc'][i,start:length,symbol];canonical=src*(p/den);alternative=(src/den)*q;high=Decimal(str(src[0,1]))*exact/Decimal(str(den))
            if day:factor*=bank['split_ratio'][i,symbol]
            frame=pd.DataFrame(canonical,columns=['econ_open','econ_high','econ_low','econ_close']);frame['share_factor']=factor;frames.append(frame)
            if day==int(b['production']['exit_day']):rows.append({'day':day,'canonical_high':float(canonical[0,1]),'alternative_high':float(alternative[0,1]),'decimal_high':str(high),'target':b['production']['fill']})
            exact=Decimal(str(src[-1,3]))*exact/Decimal(str(den));p=canonical[-1,3];q=alternative[-1,3]
    entry=m['result']['entry'][sample,symbol];v=first_exit(pd.concat(frames),entry,np.ceil(entry*1.05*100-1e-10)/100,.06,None,.0005,.0005)
    assert v['reason']=='target' and v['fill']==b['production']['fill']==26.33
    assert rows[0]['canonical_high']==rows[0]['target'] and rows[0]['alternative_high']<rows[0]['target'] and Decimal(rows[0]['decimal_high'])==Decimal('26.33')
    out={'passed':True,'boundary':rows,'scalar_result':v,'source_sha256':sha(Path(__file__)),'bank_sha256':sha(run/'path_bank.npz'),'strategy_changes':False,'tolerance_changes':False};dump(SOURCE/'evidence/boundary_reproduction.json',out);print(out)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);reproduce(p.parse_args().run)

"""Eight preregistered input arms, one fixed algorithm and original cohorts."""
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import OUT
import research

def configure():
    research.OUT=OUT/'R06';research.VERSION='preopen_5d5pct_v6_R06';research.REPORT_TITLE='富途历史信息增量 R06 结果'
    research.ARMS=['T0','TC','TF','TO','TV','TS','TA','TL'];research.ALG={a:'LightGBM' for a in research.ARMS}
    research.NAMES={'T0':'原H3重训控制','TC':'共享数据覆盖控制','TF':'日级资金流','TO':'期权成交与持仓','TV':'历史IV与HV','TS':'每日卖空成交','TA':'四族信息联合','TL':'四族信息五交易日延迟'}
    research.GLOBAL_TOP_CALIBRATION=True;research.TRAINING_ONLY_VOL_EDGES=True
    def columns(arm):
        c=json.loads((research.OUT/'coverage.json').read_text());base=c['base_features']
        if arm=='T0':return base
        base=base+c['control_features']
        family={'TF':'capital','TO':'options','TV':'volatility','TS':'short_volume'}
        if arm in family:return base+c['family_features'][family[arm]]
        if arm=='TA':return base+sum(c['family_features'].values(),[])
        if arm=='TL':return base+c['delayed_features']
        return base
    research.columns=columns

if __name__ in ['__main__','__mp_main__']:configure()
if __name__=='__main__':research.main()

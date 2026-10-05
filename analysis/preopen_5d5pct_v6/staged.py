"""Reuse the frozen temporal evaluator with four preregistered R01 inputs."""
import research
from common import OUT

BASE_OUT=OUT
research.OUT=BASE_OUT/'R01'
research.VERSION='preopen_5d5pct_v6_R01'
research.REPORT_TITLE='五日 +5% 分阶段 R01 结果'
research.ARMS=['S0','S1','S2','S3']
research.NAMES={'S0':'基础分钟共同样本','S1':'日线与分钟联合','S2':'日线机会OOF + 分钟入场','S3':'联合输入 + 日线机会OOF'}
research.ALG={a:'LightGBM' for a in research.ARMS}
def columns(arm):
    excluded={'_row_id','minute','entry_minute','reference','feature_available','prior_volume','last_volume','liquidity_dollars','stock_id','ex_action','entry','target','y','label_end','future_status','opp_oof_raw'}
    panel=research.PANEL
    base=[c for c in panel.select_dtypes('number') if c not in excluded and not c.startswith(('qqq_','peer_','relative_','d_','b_','e_'))]
    if arm in ['S1','S3']:base+=[c for c in panel if c.startswith('d_')]
    if arm in ['S2','S3']:base+=['opp_oof_raw']
    return base
research.columns=columns
if __name__=='__main__':research.main()

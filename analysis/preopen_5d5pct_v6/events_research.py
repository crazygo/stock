"""R02 frozen input comparison using the same temporal evaluator."""
import research
from common import OUT
research.OUT=OUT/'R02';research.VERSION='preopen_5d5pct_v6_R02';research.REPORT_TITLE='五日 +5% 精确板块与公开事件 R02 结果'
research.ARMS=['P0','P1','E0','E1'];research.ALG={a:'LightGBM' for a in research.ARMS}
research.NAMES={'P0':'相关股/QQQ共同输入','P1':'精确30分钟板块持续性','E0':'分钟 + SEC覆盖控制','E1':'已公布原始申报 + 量价确认'}
def columns(arm):
 excluded={'_row_id','minute','entry_minute','reference','feature_available','prior_volume','last_volume','liquidity_dollars','stock_id','ex_action','entry','target','y','label_end','future_status','sec_source_covered'}
 p=research.PANEL;cols=[c for c in p.select_dtypes('number') if c not in excluded and not c.startswith(('qqq_','peer_','relative_','d_','b_','e_'))]
 if arm in ['P0','P1']:cols += [c for c in p if c.startswith(('qqq_','peer_','relative_'))]
 if arm=='P1':cols += [c for c in p if c.startswith('b_')]
 if arm in ['E0','E1']:cols += ['sec_source_covered']
 if arm=='E1':cols += [c for c in p if c.startswith('e_')]
 return cols
research.columns=columns
if __name__=='__main__':research.main()

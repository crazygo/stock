import research
from common import OUT
research.OUT=OUT/'R04';research.VERSION='preopen_5d5pct_v6_R04';research.REPORT_TITLE='五日 +5% 季度已披露实际值 R04 结果'
research.ARMS=['F0','F1'];research.ALG={a:'LightGBM' for a in research.ARMS};research.NAMES={'F0':'公开申报延续共同输入','F1':'原季度实际同比 + 申报延续'}
def columns(arm):
 excluded={'_row_id','minute','entry_minute','reference','feature_available','prior_volume','last_volume','liquidity_dollars','stock_id','ex_action','entry','target','y','label_end','future_status','sec_source_covered'}
 p=research.PANEL;cols=[c for c in p.select_dtypes('number') if c not in excluded and not c.startswith(('qqq_','peer_','relative_','d_','b_','e_','f_'))]
 cols+=['sec_source_covered']+[c for c in p if c.startswith('e_')]
 if arm=='F1':cols+=[c for c in p if c.startswith('f_')]
 return cols
research.columns=columns
if __name__=='__main__':research.main()

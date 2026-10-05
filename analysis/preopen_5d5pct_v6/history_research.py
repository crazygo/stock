"""R03 preregistered same-algorithm history/mature-information comparison."""
import json
import pandas as pd
import research
from common import OUT,sha
research.OUT=OUT/'R03';research.VERSION='preopen_5d5pct_v6_R03';research.REPORT_TITLE='五日 +5% 历史覆盖与成熟状态 R03 结果'
research.ARMS=['H0','H1','H2','H3'];research.ALG={a:'LightGBM' for a in research.ARMS}
research.NAMES={'H0':'旧历史 + 新行动审计','H1':'扩展历史相同分钟输入','H2':'扩展历史 + 已成熟五日状态','H3':'成熟状态 + 前日日线'}
research.GLOBAL_TOP_CALIBRATION=True;research.TRAINING_ONLY_VOL_EDGES=True
FULL=None;OLD_PANEL=None;COVERAGE=None
original_init=research.init
def init():
    global FULL,OLD_PANEL,COVERAGE
    original_init();FULL=research.PANEL;COVERAGE=json.loads((research.OUT/'coverage.json').read_text())
    path=research.OUT/'cache/H0_panel.parquet';assert sha(path)==COVERAGE['H0_panel_sha256']
    OLD_PANEL=pd.read_parquet(path);OLD_PANEL.symbol=OLD_PANEL.symbol.astype(str);OLD_PANEL.day=OLD_PANEL.day.astype(str)
def columns(arm):
    cols=COVERAGE['base_features'].copy()
    if arm in ['H2','H3']:cols+=[c for c in FULL.select_dtypes('number') if c.startswith('m5_')]
    if arm=='H3':cols+=[c for c in FULL if c.startswith('d_')]
    return cols
original_fit=research.fit
def fit(job):
    arm,_=job;research.PANEL=OLD_PANEL if arm=='H0' else FULL
    research.PANEL_PATH=research.OUT/'cache'/('H0_panel.parquet' if arm=='H0' else 'panel.parquet')
    research.PANEL_SHA=COVERAGE['H0_panel_sha256' if arm=='H0' else 'panel_sha256']
    return original_fit(job)
research.init=init;research.columns=columns;research.fit=fit
if __name__=='__main__':research.main()

"""Permit newly matured outcomes, while preserving causal inputs and known results."""
from path_learning import pd,read,np,sha
OUTCOMES=['C','D','M','release','terminal_release']


def compare_panel(old,new,cutoff):
    pd.testing.assert_frame_equal(old.drop(columns=OUTCOMES),new.loc[old.index].drop(columns=OUTCOMES))
    for f in OUTCOMES:
        known=old[f].notna();pd.testing.assert_series_equal(old.loc[known,f],new.loc[old.index[known],f].astype(old[f].dtype))
    newly=old.terminal_release.isna()&new.loc[old.index,'terminal_release'].notna()
    assert (new.loc[old.index[newly],'terminal_release']>=cutoff).all()
    newly_values=old.C.isna()&new.loc[old.index,'C'].notna()
    assert new.loc[old.index[newly_values],'terminal_release'].notna().all() and (new.loc[old.index[newly_values],'terminal_release']>=cutoff).all()
    return int(newly.sum())


def audit_prefix(dev,run):
    cutoff=pd.Timestamp(read(dev/'input_manifest.json')['observation_cutoff_exclusive'])
    counts={}
    for f in ['activity_panel.parquet','financial_panel.parquet']:
        counts[f]=compare_panel(pd.read_parquet(dev/f),pd.read_parquet(run/f),cutoff)
    old=pd.read_parquet(dev/'features.parquet');new=pd.read_parquet(run/'features.parquet');pd.testing.assert_frame_equal(old,new.loc[old.index]);return counts

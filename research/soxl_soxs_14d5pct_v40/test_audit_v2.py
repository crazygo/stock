"""Regression for a real frozen sample exactly on a tree split boundary."""
import unittest,gzip,pickle
from threadpoolctl import threadpool_limits
from nonlinear_direction import *
from audit_nonlinear_models_v2 import model_equations

class TestBoundary(unittest.TestCase):
    def test_frozen_discontinuous_preprocessing_boundary(self):
        with threadpool_limits(1):
            root=core.ROOT/'.cache/soxl_soxs_14d5pct_v40/runs/nonlinear_20261013';source=Path(read(root/'input_manifest.json')['source'])
            panel=pd.read_parquet(source/'financial_panel.parquet');record=read(root/'training/2025-08-12_1035.json');at=pd.Timestamp(record['at'])
            for kind in ['tree','stump','linear']:
                info=record['fits'][kind]['fit'];model=pickle.load(gzip.open(root/'models'/(kind+'_2025-08-12_1035.pkl.gz'),'rb'))
                values,n=model_equations(panel,list(pd.to_datetime(info['members'],utc=True)),at,model,info,info['columns'],kind)
                self.assertGreater(n,2);np.testing.assert_allclose([values[s] for s in core.ETFS],record['raws'][kind],atol=1e-12)

if __name__=='__main__':unittest.main()

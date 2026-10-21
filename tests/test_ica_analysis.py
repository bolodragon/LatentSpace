import sys,tempfile,unittest
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from ica_analysis import register_ica_input,fit_ica,save_ica


class ICATests(unittest.TestCase):
    def setUp(self):
        rng=np.random.default_rng(13)
        self.sources=np.column_stack([rng.laplace(size=2000),rng.uniform(-2,2,2000),rng.exponential(size=2000)])
        self.x=self.sources@np.array([[1,.3,.5],[.4,1,.1],[.3,.2,1]])
        self.table=pd.DataFrame(dict(source=['particle']*2000,image_name=[f'id{i}' for i in range(2000)],source_row_1based=np.arange(2000)+1))
        self.bundle=register_ica_input({},'test',self.x,self.table,input_source='own',input_directory='/tmp/source')

    def test_recover_sources_preserve_ids_and_replay(self):
        result=fit_ica(self.bundle,columns=[3,1,2],seed=0)
        self.assertTrue(result['settings']['converged'])
        corr=np.abs(np.corrcoef(result['scores'].T,self.sources.T)[:3,3:])
        self.assertTrue(np.all(corr.max(axis=0)>.97))
        self.assertEqual(result['table'].image_name.tolist(),self.table.image_name.tolist())
        with tempfile.TemporaryDirectory() as tmp:
            folder=save_ica(tmp,result,'test')
            with np.load(folder/'transform.npz') as t:
                replay=(self.x[:,t['input_columns']-1]-t['mean'])@t['components'].T
            np.testing.assert_allclose(replay,result['scores'],atol=1e-10)

    def test_invalid_and_nonconvergence(self):
        for columns in ([],[0],[1,1],[4]):
            with self.assertRaises(ValueError):fit_ica(self.bundle,columns=columns)
        rank_def=register_ica_input({},'test',np.c_[self.x[:,0],self.x[:,0]],self.table,input_source='own',input_directory='/tmp')
        with self.assertRaises(ValueError):fit_ica(rank_def)
        self.assertFalse(fit_ica(self.bundle,max_iter=1,tol=1e-15)['settings']['converged'])

    def test_registry_snapshots(self):
        original=self.bundle['values'].copy()
        self.x[:]=0;self.table.loc[0,'image_name']='changed'
        np.testing.assert_array_equal(self.bundle['values'],original)
        self.assertEqual(self.bundle['table'].iloc[0].image_name,'id0')

if __name__=='__main__':unittest.main()

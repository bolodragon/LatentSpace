import sys,tempfile,unittest
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from analysis_storage import save_pca,restore_pca,save_clustering,load_clustering
from clustering_results import register_clustering_result
from ica_analysis import register_ica_input,fit_ica,save_ica,load_ica


class StorageTests(unittest.TestCase):
    def test_restart_pca_and_named_clustering(self):
        rng=np.random.default_rng(7);x=rng.normal(size=(30,9))
        model=PCA(n_components=4).fit(x);scores=model.transform(x)
        table=pd.DataFrame(dict(source=['particle']*30,image_name=[f'{i:06d}' for i in range(30)],source_row_1based=np.arange(30)+1))
        ns=dict(cwf_pca=model,cwf_pca_embedding=scores,cwf_pca_table=table,cwf_circle_mask=np.ones((3,3),bool),
            cwf_display_pixels=dict(denoised=x,noise_denoised=np.empty((0,9))),cwf_pca_noise_indices=np.array([],int),
            cwf_pca_source='own',cwf_loaded_directory=Path('/tmp/source'),cwf_pca_input_shape=x.shape,
            cwf_pca_seed=7,cwf_display_description='test')
        with tempfile.TemporaryDirectory() as tmp:
            folder=save_pca(tmp,ns);restored={};restore_pca(restored,tmp)
            np.testing.assert_array_equal(restored['cwf_pca_embedding'],scores)
            np.testing.assert_allclose(restored['cwf_pca'].transform(x),scores)
            self.assertEqual(restored['cwf_pca_table'].image_name.tolist(),table.image_name.tolist())
            table['cluster']=np.r_[[-1]*10,[1]*20]
            result=register_clustering_result({},'example',table,input_source='own',input_directory='/tmp/source',parameters={'algorithm':'DBSCAN'})
            first=save_clustering(tmp,'example',result,arrays={'core_indices':np.arange(10)},parent_pca=folder)
            second=save_clustering(tmp,'example',result,parent_pca=folder)
            self.assertNotEqual(first,second);self.assertTrue(first.exists())
            fresh=load_clustering(tmp,'example',first)
            pd.testing.assert_frame_equal(fresh['table'],table)
            np.testing.assert_array_equal(fresh['arrays']['core_indices'],np.arange(10))
            self.assertEqual(fresh['parent_pca'],str(folder))
            with self.assertRaises(ValueError):load_clustering(tmp,'wrong',first)

    def test_ica_reload_transform(self):
        rng=np.random.default_rng(1);x=rng.laplace(size=(200,3))
        table=pd.DataFrame(dict(source=['particle']*200,image_name=[f'{i:06d}' for i in range(200)],source_row_1based=np.arange(200)+1))
        bundle=register_ica_input({},'pca',x,table,input_source='own',input_directory='/tmp/source')
        result=fit_ica(bundle)
        with tempfile.TemporaryDirectory() as tmp:
            save_ica(tmp,result,'pca');fresh=load_ica(tmp)
            np.testing.assert_allclose(fresh['model'].transform(x),result['scores'])
            self.assertEqual(fresh['table'].image_name.tolist(),table.image_name.tolist())

if __name__=='__main__':unittest.main()

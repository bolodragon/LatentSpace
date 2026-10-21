import sys,tempfile,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from particle_aggregation import aggregate_polar
from particle_paired import crop_pairs,align_pairs,aggregate_pairs,fft_pairs,combine_pairs,transform_combined,loading_maps


class FeatureTests(unittest.TestCase):
    def test_radial_geometry_and_occupancy(self):
        images=np.ones((2,32,32),np.float32);images[1]*=-2
        r=aggregate_polar(images,n_radial_bands=3,n_sectors=8,radius_fraction=.35)
        self.assertEqual(r['vectors'].shape,(2,48))
        np.testing.assert_array_equal(r['vectors'][0,:24],r['pixel_counts'])
        np.testing.assert_array_equal(r['vectors'][0,24:],1)
        np.testing.assert_array_equal(r['vectors'][1,24:],0)
        self.assertTrue(np.all(r['sector_map'][~r['support_mask']]==-1))
        # Within the same angular direction, outward pixels move to later rings.
        self.assertLess(r['sector_map'][14,16],r['sector_map'][6,16])
        counts=aggregate_polar(images,n_radial_bands=3,n_sectors=8,count_mode='count')
        np.testing.assert_array_equal(counts['vectors'][0,24:],r['pixel_counts'])

    def test_binary_edges_fft_and_balanced_combination(self):
        rng=np.random.default_rng(2);images=rng.normal(size=(12,32,32)).astype('f4')
        names=[f'{i:06d}' for i in range(12)]
        with tempfile.TemporaryDirectory() as tmp:
            crop=crop_pairs(tmp,images,names,tmp,'test',preserve_edge_strength=False)
            edge=np.load(crop['settings']['files']['cropped_derivative'])
            self.assertEqual(set(np.unique(edge)),{0,1})
            empty=crop_pairs(tmp,np.zeros((1,32,32),np.float32),['zero'],tmp,'test',preserve_edge_strength=False)
            self.assertFalse(np.load(empty['settings']['files']['cropped_derivative']).any())
            aligned=align_pairs(tmp,crop);agg=aggregate_pairs(tmp,aligned);fft=fft_pairs(tmp,aligned)
            circle=np.load(crop['settings']['files']['circle_mask'])
            for channel in ('original','derivative'):
                arr=np.load(aligned['settings']['files']['aligned_cropped_'+channel])
                spectra=np.load(fft['settings']['files']['fft_complex_'+channel])
                np.testing.assert_allclose(spectra[:,16,16],0,atol=1e-6)
                restored=np.fft.ifft2(np.fft.ifftshift(spectra,axes=(-2,-1)),norm='ortho').real
                np.testing.assert_allclose(restored[:,~circle],0,atol=1e-6)
                np.testing.assert_allclose(restored[:,circle],arr[:,circle]-arr[:,circle].mean(axis=1)[:,None],atol=1e-6)
            raw=combine_pairs(tmp,agg,fft)
            balanced=combine_pairs(tmp,agg,fft,scaling='block_variance',block_weights={'fft_original':2})
            self.assertEqual(balanced['vectors'].shape,(12,4*32*32+4*24))
            np.testing.assert_allclose(transform_combined(raw['vectors'],balanced),balanced['vectors'],atol=1e-7)
            np.testing.assert_allclose(balanced['vectors'].mean(axis=0),0,atol=1e-7)
            for block in balanced['settings']['blocks']['combined_cropped']:
                v=balanced['vectors'][:,block['start']:block['stop']].var(axis=0).sum()
                self.assertAlmostEqual(float(v),block['weight']**2,places=5)
            maps,_=loading_maps(balanced['vectors'][:1],balanced,'combined_cropped')
            for i in [2,3,4,5]:np.testing.assert_array_equal(maps[:,i,~circle],0)
            with self.assertRaises(ValueError):combine_pairs(tmp,agg,fft,scaling='oops')
            with self.assertRaises(ValueError):combine_pairs(tmp,agg,fft,block_weights={'fft_original':-1})
            # Constant blocks remain finite and zero rather than amplifying numerical noise.
            const=align_pairs(tmp,empty);ca=aggregate_pairs(tmp,const);cf=fft_pairs(tmp,const)
            cb=combine_pairs(tmp,ca,cf,scaling='block_variance')
            self.assertTrue(np.isfinite(cb['vectors']).all());self.assertFalse(cb['vectors'].any())

if __name__=='__main__':unittest.main()

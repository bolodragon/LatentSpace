import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from particle_paired import (crop_pairs,align_pairs,aggregate_pairs,fft_pairs,combine_pairs,
    load_paired_dataset,loading_maps,_alignment)
from particle_alignment import apply_saved_alignment
from cluster_examples import load_example_images
import pandas as pd


class PairedTests(unittest.TestCase):
    def test_pair_features_and_replay(self):
        with tempfile.TemporaryDirectory() as tmp:
            y,x=np.indices((32,32))
            base=np.exp(-((y-15)**2/45+(x-14)**2/5))+ .7*np.exp(-((y-10)**2+(x-20)**2)/8)
            images=np.stack([base,np.fliplr(base),base*3-0.2]).astype('f4')
            names=np.array(['a','b','c'])
            crop=crop_pairs(tmp,images,names,tmp,'test',sigma=1,edge_percentile=70,radius_fraction=.35)
            m=crop['settings']; circle=np.load(m['files']['circle_mask'])
            np.testing.assert_array_equal(crop['images'],np.where(circle,images,0))
            derivatives=np.load(m['files']['cropped_derivative'])
            for i,image in enumerate(images):
                edge=np.hypot(gaussian_filter(image,1,order=(1,0)),gaussian_filter(image,1,order=(0,1)))
                np.testing.assert_allclose(derivatives[i],np.where(circle & (edge>=np.percentile(edge,70)),edge,0))
            aligned=align_pairs(tmp,crop)
            for key,source in [('aligned_cropped_original',crop['images']),('aligned_cropped_derivative',derivatives)]:
                saved=np.load(aligned['settings']['files'][key])
                replay,_=apply_saved_alignment(source[::-1],names[::-1],aligned['directory'])
                np.testing.assert_array_equal(replay,saved[::-1])
            agg=aggregate_pairs(tmp,aligned,n_sectors=8,start_angle_degrees=45,n_radial_bands=1,count_mode="count")
            original=np.load(aligned['settings']['files']['aligned_cropped_original'])
            np.testing.assert_allclose(agg['vectors'][:,:8].sum(axis=1),original[:,circle].sum(axis=1),rtol=1e-5)
            np.testing.assert_array_equal(agg['vectors'][:,8:].sum(axis=1),(original[:,circle]>0).sum(axis=1))
            fft=fft_pairs(tmp,aligned)
            complex_values=np.load(fft['settings']['files']['fft_complex_original'])
            np.testing.assert_allclose(np.fft.ifft2(np.fft.ifftshift(complex_values,axes=(-2,-1)),norm='ortho').real,np.where(circle, original-original[:,circle].mean(axis=1)[:,None,None],0),atol=1e-6)
            combined=combine_pairs(tmp,agg,fft)
            expected=np.concatenate([np.load(combined['settings']['files'][key]).reshape(3,-1) for key in ['aligned_cropped_original','aligned_cropped_derivative','aggregated_original','aggregated_derivative','fft_original','fft_derivative']],axis=1)
            np.testing.assert_array_equal(combined['vectors'],expected)
            self.assertEqual(expected.shape,(3,4*32*32+32))
            maps,labels=loading_maps(expected[:1],combined,'combined_cropped')
            self.assertEqual(maps.shape,(1,8,32,32));self.assertEqual(len(labels),8)
            subset=combine_pairs(tmp,agg,fft,selected_blocks=['fft_derivative','original_positive_pixel_count','aligned_cropped_original'])
            expected_subset=np.concatenate([np.load(fft['settings']['files']['fft_derivative']).reshape(3,-1),agg['vectors'][:,8:],original.reshape(3,-1)],axis=1)
            np.testing.assert_array_equal(subset['vectors'],expected_subset)
            subset_maps,subset_labels=loading_maps(expected_subset[:1],subset,'combined_cropped')
            self.assertEqual(subset_maps.shape,(1,3,32,32))
            self.assertEqual([b['start'] for b in subset['settings']['blocks']['combined_cropped']],[0,1024,1032])
            single=combine_pairs(tmp,agg,fft,selected_blocks=['derivative_positive_pixel_count'])
            self.assertEqual(single['vectors'].shape,(3,8))
            np.testing.assert_array_equal(single['vectors'],np.load(agg['settings']['files']['aggregated_derivative'])[:,8:])
            for bad in ([],['unknown'],['fft_original','fft_original']):
                with self.assertRaises(ValueError): combine_pairs(tmp,agg,fft,selected_blocks=bad)
            table=pd.DataFrame({'source':['particle','particle'],'image_name':['c','a']})
            selected,_,_=load_example_images(tmp,table,'aligned_cropped_original',fitted_source='combined_cropped',fitted_directory=combined['directory'],fitted_bundle=combined)
            np.testing.assert_array_equal(selected,original[[2,0]])
            other=align_pairs(tmp,crop)
            with self.assertRaises(ValueError): combine_pairs(tmp,agg,fft_pairs(tmp,other))
            with self.assertRaises(ValueError): crop_pairs(tmp,images,['a','a','b'],tmp,'test')

    def test_axis_and_upper_right_secondary(self):
        y,x=np.indices((64,64))
        image=np.exp(-((y-32)**2/90+(x-32)**2/3))+.5*np.exp(-((y-23)**2+(x-42)**2)/10)
        from scipy.ndimage import rotate
        for source in (image,np.fliplr(image),np.flipud(image),rotate(image,35,reshape=False)):
            rec=_alignment(source)
            theta=np.radians(rec['angle_degrees'])
            matrix=np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
            axis=matrix@np.array([rec['axis_y'],rec['axis_x']])
            self.assertAlmostEqual(axis[1],0,places=6)
            secondary=matrix@np.array([rec['secondary_y'],rec['secondary_x']])
            secondary*=np.array([-1 if rec['flip_vertical'] else 1,-1 if rec['flip_horizontal'] else 1])
            self.assertLessEqual(secondary[0],1e-8);self.assertGreaterEqual(secondary[1],-1e-8)
        self.assertEqual(_alignment(np.zeros((32,32)))['status'],'no_positive_signal')

if __name__=='__main__': unittest.main()

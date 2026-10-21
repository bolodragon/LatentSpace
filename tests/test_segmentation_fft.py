"""Regression checks for seeded continuity and ID-preserving Fourier features."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from scipy import ndimage as ndi
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from particle_segmentation import segment_center_particle
from particle_fft import circle_fft, save_circle_fft, load_fft_results, concatenate_by_image_id


class ContinuityTests(unittest.TestCase):
    def segment(self, image, **kwargs):
        return segment_center_particle(image, sigma=0, seed_radius_fraction=.1,
            growth_radius_fraction=.48, thresholds=[.2], threshold_mode='relative',
            seed_threshold=.1, min_area=1, max_hole_area=0, **kwargs)['masks'][0]

    def test_detached_island_and_narrow_bridge(self):
        image = np.zeros((41,41))
        image[17:24,17:24]=8
        image[18:23,31:36]=8
        image[20,20]=10
        self.assertFalse(self.segment(image)[20,33])
        image[20,24:31]=8
        self.assertTrue(self.segment(image, bridge_radius_pixels=0)[20,33])
        mask=self.segment(image)
        self.assertTrue(mask[20,20])
        self.assertFalse(mask[20,33])
        self.assertEqual(ndi.label(mask)[1],1)

    def test_broad_connection_survives(self):
        image=np.zeros((41,41))
        image[17:24,17:35]=8
        image[20,20]=10
        self.assertTrue(self.segment(image)[20,33])

    def test_diagonal_and_weak_seed(self):
        image=np.zeros((41,41));image[20,20]=10;image[21,21]=8
        self.assertFalse(self.segment(image,bridge_radius_pixels=0,connectivity=4)[21,21])
        self.assertTrue(self.segment(image,bridge_radius_pixels=0,connectivity=8)[21,21])
        self.assertFalse(self.segment(np.zeros((41,41))).any())
        self.assertFalse(self.segment(image).any())  # Opening removed isolated seed.


class FourierTests(unittest.TestCase):
    def test_roi_dc_and_fft_convention(self):
        rng=np.random.default_rng(2)
        images=rng.normal(size=(3,16,16))
        result=circle_fft(images,.35)
        changed=images.copy();changed[:,~result['circle_mask']]=1e6
        np.testing.assert_allclose(circle_fft(changed,.35)['vectors'],result['vectors'])
        recovered=np.fft.ifft2(np.fft.ifftshift(result['spectra'],axes=(-2,-1)),norm='ortho').real
        np.testing.assert_allclose(recovered,result['windowed_images'],atol=1e-6)
        np.testing.assert_allclose(result['spectra'][:,8,8],0,atol=1e-6)
        np.testing.assert_allclose(result['vectors'],np.log1p(np.abs(result['spectra'])).reshape(3,-1),atol=1e-6)
        np.testing.assert_allclose(circle_fft(np.ones((2,16,16)),.35)['vectors'],0,atol=1e-6)
        self.assertEqual(result['vectors'].shape,(3,256))

    def test_checkpoint_ids_concatenation_and_legacy_source(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);source=root/'source';seg=root/'seg'
            source.mkdir();seg.mkdir()
            images=np.random.default_rng(1).normal(size=(3,16,16))
            np.savez(source/'cwf_result.npz',denoised=images,image_names=['a','b','c'])
            np.savez(seg/'masks.npz',masks=np.ones((2,1,16,16),bool),thresholds=[.1],image_names=['c','a'])
            (seg/'settings.json').write_text(json.dumps(dict(source_checkpoint=str(source),source='normalized_camille',growth_radius_fraction=.35)))
            saved=save_circle_fft(root,seg)
            loaded=load_fft_results(root)
            self.assertEqual(loaded['image_names'].tolist(),['c','a'])
            np.testing.assert_allclose(loaded['vectors'],circle_fft(images[[2,0]],.35)['vectors'])
            combined=concatenate_by_image_id(np.array([[1],[2]]),['a','c'],loaded,source)
            np.testing.assert_allclose(combined[:,1:],loaded['vectors'][[1,0]])
            with self.assertRaises(ValueError):concatenate_by_image_id(np.ones((1,1)),['b'],loaded,source)
            with self.assertRaises(ValueError):concatenate_by_image_id(np.ones((1,1)),['a'],loaded,root/'other')
            # Cluster displays trace back to the exact CWF source using IDs.
            import pandas as pd
            from cluster_examples import load_example_images
            table=pd.DataFrame({'image_name':['a','c'],'source':['particle']*2,'source_row_1based':[1,3]})
            displayed,_,_=load_example_images(root,table,dataset='fft_circle',fitted_source='fft_circle',fitted_directory=saved['directory'],fitted_bundle=saved)
            np.testing.assert_allclose(displayed,saved['images'][[1,0]])
            # Outside-in diagnostics follow the selected edge percentile, not argmin.
            masks=np.zeros((2,2,16,16),bool)
            masks[:,1,4:12,4:12]=True
            np.savez(seg/'masks.npz',masks=masks,thresholds=[50,70],
                     selected_threshold_index=1,image_names=['c','a'])
            settings=json.loads((seg/'settings.json').read_text())
            settings['method']='outside_in'
            (seg/'settings.json').write_text(json.dumps(settings))
            exterior_fft=save_circle_fft(root,seg)
            self.assertFalse(exterior_fft['empty_masks'].any())


class EdgeWatershedTests(unittest.TestCase):
    def setUp(self):
        y,x=np.indices((41,41))
        self.image=(10*np.exp(-((x-20)**2+(y-20)**2)/18)
                    + 5*np.exp(-((x-29)**2+(y-20)**2)/16))
        self.settings=dict(method='edge_watershed', sigma=1, thresholds=[-.1,.1],
                           threshold_mode='relative', seed_threshold=.1, edge_threshold=.2)

    def test_two_controls_and_connected_output(self):
        r=segment_center_particle(self.image,**self.settings)
        stronger=segment_center_particle(self.image,**dict(self.settings,edge_threshold=0))
        self.assertFalse(np.array_equal(r['masks'],stronger['masks']))
        self.assertFalse(np.array_equal(r['masks'][0],r['masks'][1]))
        for mask in r['masks']:
            self.assertEqual(ndi.label(mask)[1],1)
            self.assertTrue(mask[r['seed']])
            self.assertFalse((mask & ~r['growth_region']).any())
        # A narrow branch is not erased by a hidden opening setting.
        opened=segment_center_particle(self.image,**dict(self.settings,bridge_radius_pixels=5))
        np.testing.assert_array_equal(opened['masks'],r['masks'])

    def test_derivatives_and_contrast_invariance(self):
        r=segment_center_particle(self.image,**self.settings)
        expected=np.hypot(ndi.gaussian_filter(self.image,1,order=(1,0)),
                          ndi.gaussian_filter(self.image,1,order=(0,1)))
        np.testing.assert_allclose(r['gradient'],expected)
        scaled=segment_center_particle(3*self.image+7,**self.settings)
        np.testing.assert_allclose(r['gradient_relative'],scaled['gradient_relative'],atol=1e-12)
        np.testing.assert_array_equal(r['masks'],scaled['masks'])
        weak=segment_center_particle(np.ones_like(self.image),**self.settings)
        self.assertFalse(weak['masks'].any())

    def test_invalid_controls(self):
        for overrides in ({'edge_threshold':-1}, {'edge_threshold':np.nan},
                          {'foreground_core_fraction':0}, {'thresholds':[-1.1,.1]}):
            with self.assertRaises(ValueError):
                segment_center_particle(self.image,**dict(self.settings,**overrides))


if __name__=='__main__':unittest.main()

import sys,unittest
from pathlib import Path
import numpy as np
from scipy import ndimage as ndi
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from particle_exterior import segment_exterior, exterior_partition

class ExteriorTests(unittest.TestCase):
    def test_closed_boundary_protects_dark_center_and_rejects_exterior(self):
        y,x=np.indices((41,41));radius=np.hypot(y-20,x-20)
        domain=radius<=18;central=radius<=5
        ring=(radius>=7)&(radius<=9)
        r=exterior_partition(ring,domain,central,min_area=4)
        self.assertTrue(r['mask'][20,20])
        self.assertTrue(r['exterior'][20,35])
        self.assertFalse(r['mask'][20,35])
        self.assertFalse((r['mask']&r['exterior']).any())
        # A gap admits exterior flood; the center is not protected by its location.
        ring[19:22,20:]=False
        r=exterior_partition(ring,domain,central,min_area=4)
        self.assertFalse(r['mask'].any())

    def test_percentiles_and_intensity_is_diagnostic(self):
        y,x=np.indices((41,41));image=np.exp(-((x-20)**2+(y-20)**2)/30)
        a=segment_exterior(image,intensity_percentile=20,edge_percentiles=[40,70],barrier_dilation_pixels=1)
        b=segment_exterior(image,intensity_percentile=80,edge_percentiles=[40,70],barrier_dilation_pixels=1)
        np.testing.assert_array_equal(a['masks'],b['masks'])
        self.assertEqual(a['intensity_cutoff'],np.percentile(image,20))
        for q,d in zip([40,70],a['details']):
            cutoff=np.percentile(a['gradient'],q)
            np.testing.assert_allclose(d['thresholded_edge'],np.where(a['gradient']>=cutoff,a['gradient'],0))
            np.testing.assert_array_equal(d['dark_exterior']|d['bright_exterior'],d['exterior'])
            self.assertFalse((d['dark_exterior']&d['bright_exterior']).any())
        np.testing.assert_allclose(a['gradient'],np.hypot(ndi.gaussian_filter(image,1,order=(1,0)),ndi.gaussian_filter(image,1,order=(0,1))))

    def test_multiple_interiors_no_forced_largest_selection(self):
        y,x=np.indices((41,41));domain=np.hypot(y-20,x-20)<=18
        r1=np.hypot(y-20,x-14);r2=np.hypot(y-20,x-26)
        barrier=((r1>=3)&(r1<=5))|((r2>=3)&(r2<=5))
        r=exterior_partition(barrier,domain,np.hypot(y-20,x-20)<=10,min_area=4)
        self.assertEqual(r['component_count'],2)
        self.assertTrue(r['mask'][20,14]);self.assertTrue(r['mask'][20,26])

    def test_empty_blocked_and_invalid(self):
        for q in (0,50,100):
            r=segment_exterior(np.ones((32,32)),edge_percentiles=[q])
            self.assertFalse(r['masks'].any())
        full=np.ones((21,21),bool)
        r=exterior_partition(full,full,full)
        self.assertEqual(r['status'],'no_exterior_seeds')
        self.assertFalse(r['mask'].any())
        for kw in ({'intensity_percentile':-1},{'edge_percentiles':[101]},{'sigma':-1}):
            with self.assertRaises(ValueError):segment_exterior(np.ones((32,32)),**kw)

if __name__=='__main__':unittest.main()

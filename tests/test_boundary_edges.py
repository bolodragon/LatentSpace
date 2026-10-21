import unittest,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from particle_exterior import remove_boundary_edges,segment_boundary_edges,segment_exterior

class BoundaryCleanupTests(unittest.TestCase):
    def test_reject_edge_loop_preserve_other_interior(self):
        domain=np.zeros((41,41),bool);domain[2:39,2:39]=True
        edges=np.zeros_like(domain)
        edges[7:16,3]=True;edges[7:16,11]=True;edges[7,3:12]=True;edges[15,3:12]=True
        edges[18:25,18]=True;edges[18:25,24]=True;edges[18,18:25]=True;edges[24,18:25]=True
        r=remove_boundary_edges(edges,domain,boundary_band_pixels=2,edge_expansion_pixels=0)
        self.assertFalse(r['mask'][10,7]) # enclosed by rejected peripheral loop
        self.assertFalse(r['mask'][10,3]) # rejected edge
        self.assertTrue(r['mask'][20,20]) # enclosed center survives
        self.assertTrue(r['mask'][30,30]) # even ordinary interior background survives
        self.assertTrue(r['kept_edges'][18,20])
        self.assertFalse(r['mask'][0,0])
        self.assertEqual(r['rejected_component_count'],1)
        unfilled=remove_boundary_edges(edges,domain,edge_expansion_pixels=0,fill_removed_loops=False)
        self.assertTrue(unfilled['mask'][10,7])

    def test_open_boundary_arc_does_not_enclose_whole_circle(self):
        domain=np.zeros((31,31),bool);domain[2:29,2:29]=True
        edges=np.zeros_like(domain);edges[2:20,4]=True;edges[19,4:12]=True
        r=remove_boundary_edges(edges,domain,edge_expansion_pixels=0,boundary_band_pixels=3)
        self.assertFalse(r['removed_enclosures'].any())
        self.assertTrue(r['mask'][15,15])
        self.assertEqual(r['removed'].sum(),edges.sum())

    def test_no_edges_means_keep_circle_and_no_brightness_gate(self):
        a=segment_boundary_edges(np.zeros((32,32)),edge_percentiles=[70])
        np.testing.assert_array_equal(a['masks'][0],a['growth_region'])
        y,x=np.indices((32,32));im=np.sin(x/3)+np.cos(y/5)
        a=segment_boundary_edges(im,intensity_percentile=20,edge_percentiles=[70])
        b=segment_boundary_edges(im,intensity_percentile=80,edge_percentiles=[70])
        old=segment_exterior(im,edge_percentiles=[70])
        np.testing.assert_array_equal(a['masks'],b['masks'])
        np.testing.assert_array_equal(a['gradient'],old['gradient'])
        np.testing.assert_array_equal(a['details'][0]['thresholded_edge'],old['details'][0]['thresholded_edge'])

    def test_components_classified_before_expansion(self):
        domain=np.zeros((31,31),bool);domain[2:29,2:29]=True
        edges=np.zeros_like(domain);edges[3:10,3]=True;edges[10:20,5]=True
        r=remove_boundary_edges(edges,domain,edge_expansion_pixels=1)
        self.assertEqual(r['rejected_component_count'],1)
        self.assertTrue(r['kept_edges'][18,5])
        self.assertFalse(r['mask'][6,3])

if __name__=='__main__':unittest.main()

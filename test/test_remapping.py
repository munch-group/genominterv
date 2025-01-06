import unittest
import numpy as np
from genominterv.remapping import remap

class TestRemapping(unittest.TestCase):

    def test_remap_nearest_left(self):
        self.assertEqual(
            remap( (550, 600),
                   [(200, 500), (800, 900)]),
            [(50, 100)]
            )

    def test_remap_nearest_right(self):
        self.assertEqual(
            remap( (700, 750),
                   [(200, 500), (800, 900)]),
            [(-50, -100)]
            )

    def test_remap_span_midpoint(self):
        self.assertEqual(
            remap( (600, 700),
                   [(200, 500), (800, 900)]),
            [(100, 150), (-100, -150)]
            )

    def test_remap_overlaps_left(self):
        self.assertEqual(
            remap( (700, 750),
                   [(200, 500), (800, 900)]),
            [(-50, -100)]
            )

    def test_remap_overlaps_left_overlap_as_zero(self):
        self.assertEqual(
            remap( (700, 850),
                   [(200, 500), (800, 900)], overlap_as_zero=True),
            [(0, -100)]
            )

    # def test_remap_overlaps_right(self):
    #     self.assertEqual(
    #         remap( (400, 600),
    #                [(200, 500), (800, 900)]),
    #         [(np.nan, np.nan), (np.nan, np.nan)]
    #         )

    def test_remap_overlaps_right_overlap_as_zero(self):
        self.assertEqual(
            remap( (400, 600),
                   [(200, 500), (800, 900)], overlap_as_zero=True),
            [(0, 100)]
            )

    # def test_remap_overlaps_first_left(self):
    #     self.assertEqual(
    #         remap( (100, 300),
    #                [(200, 500), (800, 900)]),
    #         [(np.nan, np.nan), (np.nan, np.nan)]
    #         )        

    def test_remap_overlaps_first_left_overlap_as_zero(self):
        self.assertEqual(
            remap( (100, 300),
                   [(200, 500), (800, 900)], overlap_as_zero=True),
            [(0, -100)]
            )        

    # def test_remap_overlaps_left_and_right(self):
    #     self.assertEqual(
    #         remap( (400, 850),
    #                [(200, 500), (800, 900)]),
    #         [(np.nan, np.nan), (np.an, np.nan)]
    #         )
        
    def test_remap_span_one(self):
        self.assertEqual(
            remap( (100, 600),
                   [(200, 500), (800, 900)]),
            []
            )
        
    def test_remap_span_one_span_as_zero(self):
        self.assertEqual(
            remap( (100, 600),
                   [(200, 500), (800, 900)], span_as_zero=True),
            [(0, 0)]
            )

    def test_remap_spanning_two(self):
        self.assertEqual(
            remap( (100, 1000),
                   [(200, 500), (800, 900)]),
            []
            )

    def test_remap_at_0(self):
        self.assertEqual(
            remap( (0, 100),
                   [(200, 500), (800, 900)] ),
            [(-100, -200)]
            )

    def test_remap_before_annot(self):
        self.assertEqual(
            remap( (10, 100),
                   [(200, 500), (800, 900)]  ),
            [(-100, -190)]
            )

    def test_remap_after_annot(self):
        self.assertEqual(
            remap( (800, 900),
                   [(0, 100), (500, 700), (10000, 11000)] ),
            [(100, 200)]
            )



if __name__ == '__main__':
    unittest.main()

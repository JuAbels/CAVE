import sys
import os
import unittest
import numpy as np

import cave.plot.mds


class TestMdsMethods(unittest.TestCase):

    def test_angle(self):
        """Function to test, if the angle of two points is calculated correctly"""
        angle = mds.calculate_angle(np.array([[1, 2], [2, 3]]))
        self.assertEqual(angle, np.array([7.125]))

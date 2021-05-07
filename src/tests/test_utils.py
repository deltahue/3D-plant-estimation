import unittest
import numpy as np

import sys
path_to_lib ='/home/soley/3D-plant-estimation/src'
sys.path.insert(0, path_to_lib)
from utils import compute_rotation_matrix


class TestUtils(unittest.TestCase):
    def test_compute_rotation_matrix(self):
        
        self.assertTrue(np.allclose(compute_rotation_matrix(30, 0, 0), 
                                       np.array([[1,0,0],[0,np.sqrt(3)/2,-0.5],[0,0.5,np.sqrt(3)/2]]), 
                                       rtol=1e-05, atol=1e-08))
        
        self.assertTrue(np.allclose(compute_rotation_matrix(60, 0, 0), 
                                       np.array([[1,0,0],[0,0.5, -np.sqrt(3)/2],[0,np.sqrt(3)/2,0.5]]), 
                                       rtol=1e-05, atol=1e-08))
        
        self.assertTrue(np.allclose(compute_rotation_matrix(45, 45, 45), 
                                       np.array([[0.5,-0.1464,0.8536],[0.5,0.8536, -0.1464],[-0.7071,0.5,0.5]]), 
                                       rtol=1e-03, atol=1e-08))
        
        
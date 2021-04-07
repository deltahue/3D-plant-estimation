#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  1 13:38:55 2021

@author: soley
"""
import numpy as np

test_transformation_matrices = []
'''
# Case 1, only translations, get transformation 3,3,3 as expecte
test_transformation_matrices.append(np.array([[1,0,0,1],[0,1,0,2],[0,0,1,3],[0,0,0,1]]))
test_transformation_matrices.append(np.array([[1,0,0,4],[0,1,0,5],[0,0,1,6],[0,0,0,1]]))

# Case 2, one translation, one rotation 90 degrees about x-axis, must translate back
test_transformation_matrices.append(np.array([[1,0,0,1],[0,1,0,2],[0,0,1,3],[0,0,0,1]]))
test_transformation_matrices.append(np.array([[1,0,0,0],[0,0,-1,0],[0,1,0,0],[0,0,0,1]]))
'''
# Case 3: rotate first and then translate (must nown roatate back)
test_transformation_matrices.append(np.array([[1,0,0,0],[0,0,-1,0],[0,1,0,0],[0,0,0,1]]))
test_transformation_matrices.append(np.array([[1,0,0,1],[0,1,0,2],[0,0,1,3],[0,0,0,1]]))

#test_transformation_matrices.append(np.array([[],[],[],[]]))
test_transformation_matrices = np.array(test_transformation_matrices)

# Convert the world coordinate matrices into trajectories
test_trajectories = []
test_trajectories.append(np.eye(4)) # First frame is the reference
for i in range((len(test_transformation_matrices)-1)):
    trans = np.matmul(test_transformation_matrices[i+1],np.linalg.inv(test_transformation_matrices[i]))
    test_trajectories.append(trans)

test_trajectories = np.array(test_trajectories)
print(test_trajectories)


#%% Test quaternion
from utils import quaternion_rotation_matrix

from scipy.spatial.transform import Rotation as R
r = R.from_quat([0, 0, np.sin(np.pi/4), np.cos(np.pi/4)])
print(r.as_matrix())

print(quaternion_rotation_matrix(r.as_quat()))



import sys, os
import cv2
import numpy as np
import matplotlib.pyplot as plt
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.params import PARAMS
im = cv2.imread('../camera_cal/calibration1.jpg')
params = PARAMS['camera']['undistort']
im_undistort = cv2.undistort(im, params['matrix_intrinsic'], 
            params['distortion_parameters'], 
            None, 
            params['matrix_intrinsic'])

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 8))
ax[0].imshow(im)
ax[0].set_title("Original image")
ax[1].imshow(im_undistort)
ax[1].set_title("Undistort image")
plt.show()

im = cv2.imread('../test_images/straight_lines1.jpg')
params = PARAMS['camera']['undistort']
im_undistort = cv2.undistort(im, params['matrix_intrinsic'], 
            params['distortion_parameters'], 
            None, 
            params['matrix_intrinsic'])

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 8))
ax[0].imshow(im[..., ::-1])
ax[0].set_title("Original image")
ax[1].imshow(im_undistort[..., ::-1])
ax[1].set_title("Undistort image")
plt.show()
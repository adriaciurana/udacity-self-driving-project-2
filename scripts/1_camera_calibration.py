import cv2
import os, sys
import glob
import numpy as np
import pickle
curr_path = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.dirname(curr_path)
config_path = os.path.join(project_path, '.config')
os.makedirs(config_path, exist_ok=True)

def create_chessboard_3d_points(grid_specs):
    objp = np.zeros((np.prod(grid_specs), 3), dtype=np.float32)
    objp[:, :2] = np.mgrid[:grid_specs[0], :grid_specs[1]].T.reshape(-1, 2)
    return objp

CALIBRATION_FOLDER = os.path.join(project_path, 'camera_cal')
CHESSBOARD_GRID_SPECS = (9, 6)
CHESSBOARD_3D_POINTS = create_chessboard_3d_points(CHESSBOARD_GRID_SPECS)

chessboard_2d = []
chessboard_3d = []

list_images = glob.glob(os.path.join(glob.escape(CALIBRATION_FOLDER), '*.jpg'))
total = len(list_images)
for i, im_path in enumerate(list_images):
    im = cv2.imread(im_path)
    im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    
    if i == 0:
        width = im_gray.shape[1]
        height = im_gray.shape[0]
    ret, corners = cv2.findChessboardCorners(im_gray, CHESSBOARD_GRID_SPECS, None)
    
    if ret:
        corners_fixed = cv2.cornerSubPix(im_gray, corners, (11, 11), (-1, -1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        chessboard_2d.append(corners_fixed)
        chessboard_3d.append(CHESSBOARD_3D_POINTS)

    print(i, total)

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(chessboard_3d, chessboard_2d, (width, height), None, None)
if ret:
    print('OWWW NICE! You have an amazing camera matrix and parameters distortion')
    with open(os.path.join(config_path, 'camera_params.pickle'), 'wb') as f:
        pickle.dump({
            'matrix_intrinsic': mtx,
            'distortion_parameters': dist
        }, f, protocol=pickle.HIGHEST_PROTOCOL)



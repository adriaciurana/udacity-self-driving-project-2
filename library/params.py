import pickle
import numpy as np
import cv2
import os
project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

with open(os.path.join(project_path, '.config/camera_params.pickle'), 'rb') as handle:
    camera_params = pickle.load(handle)

with open(os.path.join(project_path, '.config/perspective_transform.pickle'), 'rb') as handle:
    perspective_transform = pickle.load(handle)

PARAMS = {
	'camera': {
		'undistort': {
			'matrix_intrinsic': camera_params['matrix_intrinsic'],
			'distortion_parameters': camera_params['distortion_parameters'],
		},

		'cenital': {
			'matrix_transform': perspective_transform['matrix'],
			'inv_matrix_transform': perspective_transform['inv_matrix'],
			'roi_points': perspective_transform['roi_points']
		}
	},

	'apply_threshold': {
		'blur': (11, 11),

		'yellow': (np.array([15, 93, 0], dtype="uint8"), np.array([75, 255, 255], dtype="uint8")),
		'white': (50, 190),

		'morphology': {
			'close_vertical': (5, 25),
			'close': (15, 15)
		}
	}
}
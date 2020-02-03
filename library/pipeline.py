import numpy as np
import cv2
from .lanes_estimator import LanesEstimator
class LanePipeline(object):
    def __init__(self, **kwargs):
        self.params = {} # default
        self.params.update(kwargs)
        self.lanes_estimator = LanesEstimator() #self.params['lanes_estimator']
        self.cache = {}

    """
        STEPS
    """
    def __camera_undistort(self, frame):
        params = self.params['camera']['undistort']
        return cv2.undistort(frame, params['matrix_intrinsic'], 
            params['distortion_parameters'], 
            None, 
            params['matrix_intrinsic'])

    def __apply_threshold(self, frame):
        params = self.params['apply_threshold']

        # Gaussian
        frame_blur = cv2.GaussianBlur(frame, params['blur'], 0)

        # BGR -> HSV
        frame_hsv = cv2.cvtColor(frame_blur, cv2.COLOR_BGR2HSV)

        # Threshold
        #frame_thres = cv2.inRange(frame_hsv, params['hsv_threshold']['lower'], params['hsv_threshold']['higher'])
        #frame_thres = frame_thres & (np.std(frame_blur, axis=-1) < 80)
        # (np.std(frame_blur[..., 1:], axis=-1) < 40) & (frame_blur[..., 2] > 140) & (frame_blur[..., 0] < 110)
        frame_yellow_cand = cv2.inRange(frame_hsv, *params['yellow']) > 0 #
        frame_white_cand = (np.std(frame_blur, axis=-1) < params['white'][0]) & (frame_blur[..., 0] > params['white'][1]) & (frame_blur[..., 1] > params['white'][1]) & (frame_blur[..., 2] > params['white'][1])
        frame_mix = frame_yellow_cand | frame_white_cand

        # Vertical morphology: 
        frame_thres = cv2.morphologyEx(np.uint8(frame_mix), cv2.MORPH_CLOSE, np.ones(params['morphology']['close_vertical'], dtype='uint8'))
        frame_thres = cv2.morphologyEx(np.uint8(frame_thres), cv2.MORPH_CLOSE, np.ones(params['morphology']['close'], dtype='uint8'))

        #aux = np.uint8(255 * (frame_thres > 0))
        #aux = np.tile(np.expand_dims(aux, axis=-1), [1, 1, 3])
        #cv2.imshow('image', np.concatenate((aux, frame), axis=1))
        #cv2.waitKey(2)#

        return frame_thres

    def __crop_roi(self, frame):
        if 'roi' not in self.cache:
            params = self.params['camera']['cenital']
            points = np.int32(params['roi_points'])
            points = np.concatenate((points, points[0].reshape(-1, 2)), axis=0)
            self.cache['roi'] = cv2.fillPoly(np.zeros_like(frame, 'uint8'), [points], (255, 255, 255)) > 0
        return frame * self.cache['roi']


    def __cenital_perspective(self, frame):
        params = self.params['camera']['cenital']
        return cv2.warpPerspective(frame, params['matrix_transform'], (frame.shape[1], frame.shape[0]), flags=cv2.INTER_LINEAR)

    def __lanes_estimation(self, frame):
        return self.lanes_estimator.run(frame)

    def __recover_perspective_lane(self, lane):
        params = self.params['camera']['cenital']
        return cv2.perspectiveTransform(lane.reshape(-1, 1, 2), params['inv_matrix_transform']).reshape(-1, 2)

    def __draw_output(self, frame, rec_left, rec_right, curvature, dist_from_center):
        """
            DRAW POLY
        """
        poly = np.concatenate((rec_left, rec_right[::-1]), axis=0)
        draw_poly = np.zeros_like(frame, 'uint8')
        cv2.fillPoly(draw_poly, np.int32([poly]), (0, 255, 0))
        frame = cv2.addWeighted(frame, 1, draw_poly, 0.3, 0)
        cv2.polylines(frame, [np.int32(rec_left)], False, (255, 0, 0), 5)
        cv2.polylines(frame, [np.int32(rec_right)], False, (255, 0, 0), 5)
        cv2.polylines(frame, [np.int32((rec_right + rec_left) / 2)], False, (0, 0, 255), 3)
        cv2.line(frame, (frame.shape[1] // 2, frame.shape[0]), (frame.shape[1] // 2, frame.shape[0] - 50), (0, 0, 255), 5)


        center_txt = "{0:.2f}m".format(round(dist_from_center, 2))
        center_size, _ = cv2.getTextSize(center_txt, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
        center_pos = (frame.shape[1] // 2 - center_size[0] // 2, frame.shape[0] - 100)
        
        draw_poly = np.zeros_like(frame, 'uint8')
        cv2.rectangle(draw_poly, (center_pos[0] - 10, center_pos[1] + 10), (center_pos[0] + center_size[0] + 10, center_pos[1] - center_size[1] - 10), (0, 255, 255), -1)
        frame = cv2.addWeighted(frame, 1, draw_poly, 0.5, 0)

        top_txt = "Curvature {0:d}m".format(int(curvature))
        top_size, _ = cv2.getTextSize(top_txt, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
        cv2.putText(frame, center_txt, center_pos, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, top_txt, (frame.shape[1]//2 - top_size[0]//2, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        return frame

    def run(self, frame):
        # 1. Camera undistort
        frame_orig = self.__camera_undistort(frame)

        # 2. Roi crop
        frame = self.__crop_roi(frame_orig)

        # 3. Cenital Homography
        frame = self.__cenital_perspective(frame)


        # 4. Postprocessing
        frame = self.__apply_threshold(frame)

        # 5. Lane estimation
        left, right, curvature, dist_from_center = self.__lanes_estimation(frame)
        rec_left = self.__recover_perspective_lane(left)
        rec_right = self.__recover_perspective_lane(right)

        # 6. Draw output
        frame_output = self.__draw_output(frame_orig.copy(), rec_left, rec_right, curvature, dist_from_center)
        
        
        return {
            'undistort': frame_orig,
            'output': frame_output,
            'curvature': curvature,
            'center': dist_from_center,
            'lanes': {
                'left': rec_left,
                'right': rec_right
            },

        }


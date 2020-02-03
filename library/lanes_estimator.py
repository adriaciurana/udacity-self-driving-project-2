import numpy as np
import cv2
import math
from collections import deque
class LanesEstimator(object):
    class Accum(object):
        def __init__(self, length, length_diff=None):
            self.__queue = deque(maxlen=length)
            self.value = None

            if length_diff is not None:
                self.diff = 0
                self.__queue_diff = deque(maxlen=length_diff)

        def set(self, value):
            if self.value is not None and hasattr(self, 'diff'):
                self.__diff = (value - self.value)
                self.__queue_diff.append(self.__diff)

            self.value = value
            self.__queue.append(self.value)

        def get_mean(self):
            return np.median(self.__queue, axis=0)

        def get_last(self):
            return self.value

        def get_mean_diff(self):
            if not hasattr(self, 'diff'):
                print('Error not has diff')
                exit()

            if len(self.__queue_diff) == 0:
                return 0.
            return np.median(self.__queue_diff, axis=0)

        def get_last_diff(self):
            if not hasattr(self, 'diff'):
                print('Error not has diff')
                exit()

            return self.__diff
    class LineAccum(object):
        def __init__(self, im_size):
            self.im_size = im_size
            self.detection = LanesEstimator.Accum(15)
            self.points = LanesEstimator.Accum(15)
            self.coefs = LanesEstimator.Accum(15, 15)
            self.curvature = LanesEstimator.Accum(15, 15)

        def __compute_poly_approx(self, points):
            coefs = np.polyfit(points[:, 1], points[:, 0], 2)
            return coefs

        def __compute_curvature(self, coefs):
            ym_per_pix = 30/720 # meters per pixel in y dimension
            xm_per_pix = 3.7/700 # meteres per pixel in x dimension
            points = self.points_from_poly(coefs)
            points[:, 0] *= xm_per_pix
            points[:, 1] *= ym_per_pix
            scaled_coefs = self.__compute_poly_approx(points)
            r = ((1 + (2*scaled_coefs[0]*self.im_size[1]*ym_per_pix + scaled_coefs[1])**2)**1.5) / np.absolute(2*scaled_coefs[0])
            return r
    
        def points_from_poly(self, coefs, y_values=None):
            if y_values is None:
                y_values = np.linspace(0, self.im_size[1] - 1, self.im_size[1])
            return np.stack((coefs[0]*(y_values**2) + coefs[1]*y_values + coefs[2], y_values), axis=1)


        def set(self, detection, points=None):
            self.detection.set(detection)

            if detection:
                self.points.set(points)
                coefs = self.__compute_poly_approx(points)
                self.coefs.set(coefs)
                self.curvature.set(self.__compute_curvature(coefs))

    def __init__(self, **kwargs):
        self.params = {} # default
        self.params.update(kwargs)

        self.frames_counter = 0
        self.left_lane = None
        self.right_lane = None
        self.dist_from_center = LanesEstimator.Accum(15, 15)

    def compute_from_hist(self, frame):
        middle_x = int(frame.shape[1] // 2)
        hist_full = np.sum(frame[frame.shape[0] - 300:frame.shape[0]], axis=0)
        
        step_y = 50
        margin_y = 60
        margin_x = 150

        weight_fusion_lanes = 0.7
        weight_from_estimation = 0.6
        speed_update_beta = 0.7

        peak_left = np.argmax(hist_full[:middle_x])
        prev_peak_left = peak_left
        peak_right = middle_x + np.argmax(hist_full[middle_x:])
        prev_peak_right = peak_right
        speed_approximator = None

        left_lane_points = []
        right_lane_points = []
        for offset_y in range(frame.shape[0], 0, -step_y):
            start_y = max(offset_y - margin_y, 0)
            end_y = offset_y

            prev_peak_left = peak_left
            prev_peak_right = peak_right

            # Estimation
            if speed_approximator is None:
                peak_left_estim = prev_peak_left
                peak_right_estim = prev_peak_right
            else:
                peak_left_estim = prev_peak_left + speed_approximator
                peak_right_estim = prev_peak_right + speed_approximator

            # Left hist
            margin_left = (math.floor(peak_left_estim - margin_x), math.ceil(peak_left_estim + margin_x))
            nonzero_left = np.where(frame[start_y:end_y, margin_left[0]:margin_left[1]])[1]
            if len(nonzero_left) / ((end_y - start_y)*(2*margin_x)) > 0.001:
                tmp_peak_left = margin_left[0] + np.mean(nonzero_left)
            else:
                tmp_peak_left = peak_left_estim

            # Right hist
            margin_right = (math.floor(peak_right_estim - margin_x), math.ceil(peak_right_estim + margin_x))
            nonzero_right = np.where(frame[start_y:end_y, margin_right[0]:margin_right[1]])[1]
            if len(nonzero_right) / ((end_y - start_y)*(2*margin_x)) > 0.001:
                tmp_peak_right = margin_right[0] + np.mean(nonzero_right)
            else:
                tmp_peak_right = peak_right_estim

            # Fusion increments
            tmp_peak_left, tmp_peak_right = weight_fusion_lanes * (tmp_peak_left - prev_peak_left) + (1 - weight_fusion_lanes) * (tmp_peak_right - prev_peak_right) + prev_peak_left, \
                weight_fusion_lanes * (tmp_peak_right - prev_peak_right) + (1 - weight_fusion_lanes) * (tmp_peak_left - prev_peak_left) + prev_peak_right

            # Increment approximator
            if speed_approximator is None:
                peak_left = tmp_peak_left
                peak_right = tmp_peak_right
                
                speed_approximator = ((peak_left - prev_peak_left) + (peak_right - prev_peak_right)) / 2
            
            else:
                peak_left = weight_from_estimation * tmp_peak_left + (1 - weight_from_estimation) * peak_left_estim
                peak_right = weight_from_estimation * tmp_peak_right + (1 - weight_from_estimation) * peak_right_estim
                
                mean_speed = ((peak_left - prev_peak_left) + (peak_right - prev_peak_right)) / 2
                speed_approximator = (1 - speed_update_beta)*speed_approximator + speed_update_beta*mean_speed

            # Points accumulator+
            middle_y = (end_y + start_y) / 2
            left_lane_points.append([peak_left, middle_y])
            right_lane_points.append([peak_right, middle_y])

        return np.array(left_lane_points), np.array(right_lane_points)
    
    def compute_from_poly(self, frame):
        step_y = 50
        margin_y = 60
        margin_x = 50

        frame_rgb = 255 * np.tile(np.expand_dims(frame, axis=-1), [1,1,3])
        for x, y in self.right_lane.points_from_poly(self.right_lane.coefs.get_last()):
            cv2.circle(frame_rgb, (int(x), int(y)),  5, (0, 255, 0), -1)

        nonzeroy, nonzerox = frame.nonzero()
        nonzerox = np.array(nonzerox)
        nonzeroy = np.array(nonzeroy)

        left_coefs = self.left_lane.coefs.get_mean() + self.left_lane.coefs.get_mean_diff()
        poly_left_lane = left_coefs[0]*(nonzeroy**2) + left_coefs[1]*nonzeroy + left_coefs[2]
        left_idx = (nonzerox > poly_left_lane - margin_x) & (nonzerox < poly_left_lane + margin_x)

        right_coefs = self.right_lane.coefs.get_mean() + self.right_lane.coefs.get_mean_diff()
        poly_right_lane = right_coefs[0]*(nonzeroy**2) + right_coefs[1]*nonzeroy + right_coefs[2]
        right_idx = (nonzerox > poly_right_lane - margin_x) & (nonzerox < poly_right_lane + margin_x)

        tmp_left_x = nonzerox[left_idx]
        tmp_left_y = nonzeroy[left_idx] 
        tmp_right_x = nonzerox[right_idx]
        tmp_right_y = nonzeroy[right_idx]

        # Rasterize bottom to top, to avoid inconsistences
        left_points = []
        right_points = []
        
        left_x_prev = None
        right_x_prev = None
        left_x = None
        right_x = None
        speed_update_beta = 0.9
        weight_fusion_lanes = 0.6
        
        speed_approximator = None
        
        for i in range(frame.shape[0] - 1, -1, -1):
            left_x_prev = left_x
            right_x_prev = right_x

            aux_left = tmp_left_x[tmp_left_y == i]
            if len(aux_left) > 0:
                left_x = np.median(aux_left)
            # elif left_x_prev is not None and speed_approximator is not None:
            #     left_x = left_x_prev + speed_approximator
            else:
                left_x = None

            aux_right = tmp_right_x[tmp_right_y == i]
            if len(aux_right) > 0:
                right_x = np.median(aux_right)
            # elif right_x_prev is not None and speed_approximator is not None:
             #    right_x = right_x_prev + speed_approximator
            else:
                right_x = None
            
            # Fusion increments
            if left_x is not None and right_x is not None and left_x_prev is not None and right_x_prev is not None:
                 left_x, right_x = weight_fusion_lanes * (left_x - left_x_prev) + (1 - weight_fusion_lanes) * (right_x - right_x_prev) + left_x_prev, \
                     weight_fusion_lanes * (right_x - right_x_prev) + (1 - weight_fusion_lanes) * (left_x - left_x_prev) + right_x_prev

            """if speed_approximator is None:
                speed_approximator = ((left_x - left_x_prev) + (right_x - right_x_prev)) / 2
            else:
                mean_speed = ((left_x - left_x_prev) + (right_x - right_x_prev)) / 2
                speed_approximator = (1 - speed_update_beta)*speed_approximator + speed_update_beta*mean_speed"""
            
            #print(left_x, right_x)
            if left_x is not None:
                left_points.append([left_x, i])

            if right_x is not None:
                right_points.append([right_x, i])




        #return np.stack((left_x, left_y), axis=1), np.stack((right_x, right_y), axis=1)
        return np.array(left_points), np.array(right_points)

        """
    

        weight_fusion_lanes = 0.5
        weight_from_estimation = 0.8

        peak_left = None
        peak_right = None

        left_lane_points = []
        right_lane_points = []
        for offset_y in range(frame.shape[0], 0, -step_y):
            start_y = max(offset_y - margin_y, 0)
            end_y = offset_y
            middle_y = (end_y + start_y) / 2

            # Estimation
            peak_left_estim = self.left_lane.points_from_poly(self.left_lane.coefs.get_last(), y_values=np.array([middle_y]))[0][0]
            peak_right_estim = self.right_lane.points_from_poly(self.right_lane.coefs.get_last(), y_values=np.array([middle_y]))[0][0]

            if peak_left is None:
                prev_peak_left = peak_left_estim
                prev_peak_right = peak_right_estim
            else:
                prev_peak_left = peak_left
                prev_peak_right = peak_right


            # Left hist
            margin_left = (math.floor(peak_left_estim - margin_x), math.ceil(peak_left_estim + margin_x))
            nonzero_left = np.where(frame[start_y:end_y, margin_left[0]:margin_left[1]])[1]
            if len(nonzero_left) / ((end_y - start_y)*(2*margin_x)) > 0.001:
                tmp_peak_left = margin_left[0] + np.mean(nonzero_left)
            else:
                tmp_peak_left = peak_left_estim

            # Right hist
            margin_right = (math.floor(peak_right_estim - margin_x), math.ceil(peak_right_estim + margin_x))
            nonzero_right = np.where(frame[start_y:end_y, margin_right[0]:margin_right[1]])[1]
            if len(nonzero_right) / ((end_y - start_y)*(2*margin_x)) > 0.001:
                tmp_peak_right = margin_right[0] + np.mean(nonzero_right)
            else:
                tmp_peak_right = peak_right_estim

            # Fusion increments
            tmp_peak_left, tmp_peak_right = weight_fusion_lanes * (tmp_peak_left - prev_peak_left) + (1 - weight_fusion_lanes) * (tmp_peak_right - prev_peak_right) + prev_peak_left, \
                weight_fusion_lanes * (tmp_peak_right - prev_peak_right) + (1 - weight_fusion_lanes) * (tmp_peak_left - prev_peak_left) + prev_peak_right

            # Increment approximator
            peak_left = weight_from_estimation * tmp_peak_left + (1 - weight_from_estimation) * peak_left_estim
            peak_right = weight_from_estimation * tmp_peak_right + (1 - weight_from_estimation) * peak_right_estim
                
            # Points accumulator
            left_lane_points.append([peak_left, middle_y])
            right_lane_points.append([peak_right, middle_y])

        return np.array(left_lane_points), np.array(right_lane_points)"""

    def compute_relative_center(self, frame, left_estim, right_estim):
        xm_per_pix = 3.7/700 # meteres per pixel in x dimension
        car_center = frame.shape[1] / 2
        road_center = (left_estim[-1, 0] + right_estim[-1, 0]) / 2
        dist_from_center = (car_center - road_center) * xm_per_pix
        self.dist_from_center.set(dist_from_center)

    def run(self, frame):
        if self.left_lane is None:
            self.left_lane = LanesEstimator.LineAccum((frame.shape[1], frame.shape[0]))
        if self.right_lane is None:
            self.right_lane = LanesEstimator.LineAccum((frame.shape[1], frame.shape[0]))

        if self.frames_counter == 0:
            left_points, right_points = self.compute_from_hist(frame)
            self.left_lane.set(len(left_points) > 5, left_points)
            self.right_lane.set(len(right_points) > 5, right_points)
        else:
            left_points, right_points = self.compute_from_poly(frame)
            self.left_lane.set(len(left_points) > 10, left_points)
            self.right_lane.set(len(right_points) > 10, right_points)

        left_estim = self.left_lane.points_from_poly(self.left_lane.coefs.get_mean())
        right_estim = self.right_lane.points_from_poly(self.right_lane.coefs.get_mean())

        self.compute_relative_center(frame, left_estim, right_estim)

        self.frames_counter += 1
        return left_estim, \
            right_estim, \
            (self.left_lane.curvature.get_mean() + self.right_lane.curvature.get_mean()) / 2, \
            self.dist_from_center.get_last()



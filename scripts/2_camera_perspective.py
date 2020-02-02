import cv2
import os, sys
import glob
import numpy as np
import pickle
curr_path = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.dirname(curr_path)
config_path = os.path.join(project_path, '.config')
os.makedirs(config_path, exist_ok=True)


im = cv2.imread(os.path.join(project_path, 'test_images', 'straight_lines1.jpg'))

"""
    POINT SELECTION
"""
is_selection_done = False
points_reference = []
curr_point = None
def click_and_crop(event, x, y, flags, param):
    global curr_point, points_reference, is_selection_done
    if len(points_reference) >= 4:
        is_selection_done = True
        return

    curr_point = (x, y)
    if event == cv2.EVENT_LBUTTONDOWN:
        points_reference.append([x, y])

cv2.namedWindow("image")
cv2.setMouseCallback("image", click_and_crop)
while not is_selection_done:
    image_draw = im.copy()
    if curr_point is not None:
        cv2.circle(image_draw, curr_point,  10, (0, 0, 255), -1)
    for i, (x, y) in enumerate(points_reference):
        cv2.circle(image_draw, (x, y),  5, (0, 255, 0), -1)
        cv2.putText(image_draw, str(i), (x, y),cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    cv2.imshow("image", image_draw)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("r"):
        image_draw = im.copy()
        points_reference = []
 
    elif key == ord("q"):
        break

def norm(vec):
    return np.sqrt(vec.dot(vec))

def u_vector(vec):
    return vec / norm(vec)

def fix_point(height, start, end):
    dir_se = (end - start)
    dir_se_norm = u_vector(dir_se)

    alpha = (height - start[1]) / dir_se_norm[1]
    return start[0] + dir_se_norm[0] * alpha, height

def get_roi_points(width, height, M_inv):
    pts = np.float32([[0, 0], [width, 0], [width, height], [0, height]]).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(pts, M_inv).reshape(-1, 2)

# def refine_M(M, width, height):
#     pts = np.float32([[0, 0], [width, 0], [width, height], [0, height]]).reshape(-1, 1, 2)
#     pts_fit = cv2.perspectiveTransform(pts, M).reshape(-1, 2)

#     """
#     x1 = np.float32([[0, 0, 1], [width, 0, 1], [width, height, 1], [0, height, 1]]).T
#     x2 = np.matmul(M, x1)
#     x2 = x2 / np.tile(np.expand_dims(x2[2, :], axis=0), [3, 1])
#     print(pts, pts_fit)
#     print(np.int32(x1), np.int32(x2.T))
#     """

#     xmin, ymin = np.int32(pts_fit.min(axis=0).ravel() - 0.5)
#     xmax, ymax = np.int32(pts_fit.max(axis=0).ravel() + 0.5)
#     M_fix = np.array([[1, 0, - xmin], [0, 1, - ymin], [0, 0, 1]])
#     return M_fix.dot(M), (xmax - xmin, ymax - ymin)

# Fix points
points_reference = np.array(points_reference)
# Same y in a, d = height of image
# Same y in b, c
a, b, c, d = points_reference
points_reference[0] = fix_point(im.shape[0], a, b)
points_reference[3] = fix_point(im.shape[0], d, c)
points_reference[2] = fix_point(b[1], d, c)

image_draw = im.copy()
for i, (x, y) in enumerate(points_reference):
    cv2.circle(image_draw, (x, y),  5, (0, 255, 0), -1)
    cv2.putText(image_draw, str(i), (x, y),cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

cv2.imshow('image', image_draw)
cv2.waitKey(0)
cv2.destroyAllWindows()

"""
    PERSPECTIVE TRANSFORM
"""
margin_top = 50
margin_x = 180
margin_bottom = 0
real_points = points_reference.copy()

new_height = norm(b - a)
estimated_points = np.array([
    [a[0] + margin_x, im.shape[0] - margin_bottom], 
    [a[0] + margin_x, margin_top], 
    [d[0] - margin_x, margin_top],
    [d[0] - margin_x, im.shape[0] - margin_bottom]])

width = im.shape[1]
height = im.shape[0]

M = cv2.getPerspectiveTransform(real_points.astype('float32'), estimated_points.astype('float32'))
M_inv = np.linalg.inv(M)
# M, new_size = refine_M(M, im.shape[1], im.shape[0])
im_warped = cv2.warpPerspective(im, M, (im.shape[1], im.shape[0]), flags=cv2.INTER_LINEAR)
image_draw = im_warped.copy()
for i, (x, y) in enumerate(estimated_points):
    cv2.circle(image_draw, (int(x), int(y)),  5, (0, 255, 0), -1)
    cv2.putText(image_draw, str(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
cv2.imshow('image', image_draw)
cv2.waitKey(0)
cv2.destroyAllWindows()

print('OWWW NICE! You have an amazing perspective transform')
with open(os.path.join(config_path, 'perspective_transform.pickle'), 'wb') as f:
    pickle.dump({
        'matrix': M,
        'inv_matrix': M_inv,
        'roi_points': get_roi_points(width, height, M_inv)
    }, f, protocol=pickle.HIGHEST_PROTOCOL)


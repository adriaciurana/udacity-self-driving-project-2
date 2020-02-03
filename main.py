import cv2
import glob
from library.pipeline import LanePipeline
from library.params import PARAMS
from library.video import VideoProcess

"""for im_path in glob.glob('test_images/*.jpg'):
    im = cv2.imread(im_path)
    p = LanePipeline(**PARAMS)
    p.run(im)
    p.run(im)
    p.run(im)
"""
vid = VideoProcess('project_video.mp4')
out = cv2.VideoWriter('project_video_output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 15, vid.size)
def __callback(im, i, total):
    print(i, total)
    output = p.run(im)['output']
    out.write(output)
vid.callback = __callback
p = LanePipeline(**PARAMS)
#vid = VideoProcess('challenge_video.mp4', __callback)
#vid = VideoProcess('harder_challenge_video.mp4', __callback)
vid.run()
out.release()
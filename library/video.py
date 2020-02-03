import cv2
class VideoProcess(object):
	def __init__(self, video_path, callback=None):
		self.video = cv2.VideoCapture(video_path)
		self.callback = callback

	def run(self):
		total = int(cv2.VideoCapture.get(self.video, int(cv2.CAP_PROP_FRAME_COUNT) ))

		i = 0
		while True:
			ret, frame = self.video.read()
			if not ret:
				break

			if self.callback is not None:
				self.callback(frame, i, total)
			i += 1

	@property
	def size(self):
		return int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))
	


import cv2

class camera:
    def __init__(self, video_path):
        self.video_path = video_path
        self.video_capture = cv2.VideoCapture(self.video_path)
        if not self.video_capture.isOpened():
            raise ValueError("Unable to open video file:", self.video_path)

    def play(self):
        while self.video_capture.isOpened():
            ret, frame = self.video_capture.read()
            return frame
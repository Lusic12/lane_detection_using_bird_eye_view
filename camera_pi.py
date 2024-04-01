from picamera2 import Picamera2

class MyCamera:
    def __init__(self):
        # Initialize the Picamera2 instance
        self.piCam = Picamera2()
        # Configure preview settings
        self.piCam.preview_configuration.main.size = (1280, 720)
        self.piCam.preview_configuration.main.format = "RGB888"
        self.piCam.preview_configuration.align()
        self.piCam.configure("preview")
        # Start preview
        self.piCam.start()

    def capture_image(self):
        # Capture an image and return it
        return self.piCam.capture_array()

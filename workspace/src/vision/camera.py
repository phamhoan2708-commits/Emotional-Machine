import cv2

class Camera:
    """Small explicit camera owner. Always call release(), including on Stop."""
    def __init__(self, index=0, width=640, height=480):
        self.index, self.width, self.height = index, width, height
        self.capture = None

    def open(self):
        if self.capture is not None and self.capture.isOpened():
            return True
        self.capture = cv2.VideoCapture(self.index)
        if not self.capture.isOpened():
            self.release()
            return False
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        return True

    def read(self):
        if self.capture is None or not self.capture.isOpened():
            return False, None
        return self.capture.read()

    def release(self):
        if self.capture is not None:
            self.capture.release()
            self.capture = None

from dataclasses import dataclass
from pathlib import Path
import cv2

@dataclass
class DetectionResult:
    faces: list
    frame: object
    message: str

class FaceDetector:
    """OpenCV Haar detector; returns boxes and quality flags, never identity."""
    def __init__(self, scale_factor=1.1, min_neighbors=5):
        cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
        self.cascade = cv2.CascadeClassifier(str(cascade_path))
        if self.cascade.empty():
            raise RuntimeError("OpenCV face detector cascade could not be loaded")
        self.scale_factor, self.min_neighbors = scale_factor, min_neighbors

    def detect(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        boxes = self.cascade.detectMultiScale(gray, self.scale_factor, self.min_neighbors,
                                               minSize=(40, 40))
        faces = [(int(x), int(y), int(w), int(h)) for x, y, w, h in boxes]
        if not faces:
            message = "Không phát hiện khuôn mặt."
        elif len(faces) > 1:
            message = "Phát hiện nhiều khuôn mặt; hãy để một người trong khung hình."
        else:
            x, y, w, h = faces[0]
            crop = gray[y:y+h, x:x+w]
            blur = cv2.Laplacian(crop, cv2.CV_64F).var() if crop.size else 0
            brightness = float(crop.mean()) if crop.size else 0
            message = "Ảnh có thể bị mờ." if blur < 35 else ("Ảnh có thể thiếu sáng." if brightness < 35 else "Đã phát hiện một khuôn mặt.")
        return DetectionResult(faces, frame, message)

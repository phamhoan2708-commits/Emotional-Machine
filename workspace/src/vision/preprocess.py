import cv2
import numpy as np

def preprocess_face(face_bgr, input_size=48, color_mode="grayscale"):
    """Return one normalized HWC batch accepted by a Keras image classifier."""
    if face_bgr is None or face_bgr.size == 0:
        raise ValueError("face crop is empty")
    if color_mode == "grayscale":
        image = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY) if face_bgr.ndim == 3 else face_bgr
        image = cv2.resize(image, (input_size, input_size), interpolation=cv2.INTER_AREA)
        image = image[..., np.newaxis]
    elif color_mode == "rgb":
        image = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (input_size, input_size), interpolation=cv2.INTER_AREA)
    else:
        raise ValueError(f"unsupported color_mode: {color_mode}")
    return (image.astype(np.float32) / 255.0)[np.newaxis, ...]

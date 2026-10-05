from pathlib import Path
import numpy as np

class ModelUnavailableError(RuntimeError):
    pass

class ExpressionClassifier:
    """Lazy Keras wrapper; class index is mapped using configured label order."""
    def __init__(self, model_path, labels, input_size=48, color_mode="grayscale", threshold=0.60):
        self.model_path = Path(model_path)
        self.labels = list(labels)
        self.input_size, self.color_mode, self.threshold = input_size, color_mode, threshold
        self.model = None

    def load(self):
        if not self.model_path.is_file():
            raise ModelUnavailableError(f"Model chưa có tại {self.model_path}. Hãy đặt model đã huấn luyện vào vị trí này.")
        try:
            from tensorflow.keras.models import load_model
            self.model = load_model(self.model_path, compile=False)
            output_shape = self.model.output_shape
            if output_shape[-1] != len(self.labels):
                raise ValueError(f"model có {output_shape[-1]} output nhưng config có {len(self.labels)} nhãn")
        except ModelUnavailableError:
            raise
        except Exception as exc:
            raise ModelUnavailableError(f"Không tải được model hoặc model không tương thích: {exc}") from exc
        return self

    @property
    def ready(self):
        return self.model is not None

    def predict(self, batch):
        if not self.ready:
            self.load()
        raw = self.model.predict(batch, verbose=0)
        scores = np.asarray(raw).reshape(-1)
        if scores.size != len(self.labels):
            raise ValueError("model output không khớp mapping nhãn")
        # Normalize logits/probabilities defensively, retaining explicit class mapping.
        if np.any(scores < 0) or not np.isclose(scores.sum(), 1.0, atol=0.05):
            scores = np.exp(scores - scores.max())
        scores = scores / scores.sum()
        idx = int(np.argmax(scores))
        confidence = float(scores[idx])
        return {"label": self.labels[idx] if confidence >= self.threshold else None,
                "top_label": self.labels[idx], "confidence": confidence,
                "scores": dict(zip(self.labels, map(float, scores))),
                "confident": confidence >= self.threshold}

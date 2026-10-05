from collections import deque
import numpy as np

class PredictionSmoother:
    def __init__(self, labels, window_size=7):
        if window_size < 1:
            raise ValueError("window_size must be positive")
        self.labels = list(labels)
        self.values = deque(maxlen=window_size)

    def reset(self):
        self.values.clear()

    def update(self, scores):
        if set(scores) != set(self.labels):
            raise ValueError("score keys must match configured labels")
        self.values.append([scores[name] for name in self.labels])
        mean = np.mean(self.values, axis=0)
        idx = int(np.argmax(mean))
        return {"label": self.labels[idx], "confidence": float(mean[idx]),
                "scores": dict(zip(self.labels, map(float, mean))), "frames": len(self.values)}

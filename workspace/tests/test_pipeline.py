import numpy as np
import pytest
from src.vision.preprocess import preprocess_face
from src.analysis.smoothing import PredictionSmoother
from src.models.expression_classifier import ExpressionClassifier, ModelUnavailableError

def test_preprocess_grayscale_shape_and_range():
    image = np.full((64, 80, 3), 128, dtype=np.uint8)
    batch = preprocess_face(image, 48, "grayscale")
    assert batch.shape == (1, 48, 48, 1)
    assert batch.dtype == np.float32
    assert 0 <= batch.min() <= batch.max() <= 1

def test_preprocess_rejects_empty_face():
    with pytest.raises(ValueError):
        preprocess_face(np.empty((0, 0), dtype=np.uint8))

def test_prediction_uses_explicit_label_order_and_threshold():
    model = ExpressionClassifier("absent.keras", ["sad", "happy"], threshold=.6)
    class FakeModel:
        output_shape = (None, 2)
        def predict(self, batch, verbose=0):
            return np.array([[.2, .8]])
    model.model = FakeModel()
    result = model.predict(np.zeros((1, 48, 48, 1)))
    assert result["label"] == "happy"
    assert result["top_label"] == "happy"
    model.threshold = .9
    assert model.predict(np.zeros((1, 48, 48, 1)))["label"] is None

def test_smoothing_averages_short_window():
    smoother = PredictionSmoother(["sad", "happy"], window_size=2)
    assert smoother.update({"sad": .9, "happy": .1})["label"] == "sad"
    result = smoother.update({"sad": .1, "happy": .9})
    assert result["label"] == "sad"
    assert result["frames"] == 2
    assert smoother.update({"sad": .0, "happy": 1.})["label"] == "happy"

def test_smoother_checks_mapping():
    smoother = PredictionSmoother(["sad", "happy"], 2)
    with pytest.raises(ValueError):
        smoother.update({"sad": .5})

def test_missing_model_is_clear_error(tmp_path):
    classifier = ExpressionClassifier(tmp_path / "no-model.keras", ["sad", "happy"])
    with pytest.raises(ModelUnavailableError, match="chưa có"):
        classifier.load()

from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = {
    "labels": ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"],
    "model": {"path": "models/expression_model.keras", "input_size": 48, "color_mode": "grayscale",
              "confidence_threshold": 0.60, "face_scale_factor": 1.1, "min_neighbors": 5},
    "smoothing": {"window_size": 7},
    "llm": {"enabled": False, "provider": "openai_compatible", "endpoint_env": "LLM_API_URL",
            "api_key_env": "LLM_API_KEY", "model_env": "LLM_MODEL", "min_call_interval_seconds": 30},
}

def load_config(path=ROOT / "config.yaml"):
    path = Path(path)
    if not path.exists():
        return DEFAULT_CONFIG.copy()
    with path.open("r", encoding="utf-8") as stream:
        loaded = yaml.safe_load(stream) or {}
    # Keep defaults for omitted sections while allowing explicit label order.
    result = {**DEFAULT_CONFIG, **loaded}
    for section in ("model", "smoothing", "llm"):
        result[section] = {**DEFAULT_CONFIG[section], **loaded.get(section, {})}
    if not result["labels"] or len(set(result["labels"])) != len(result["labels"]):
        raise ValueError("labels must be a non-empty list of unique names")
    return result

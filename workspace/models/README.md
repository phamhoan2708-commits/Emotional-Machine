# Model directory

Place your own trained Keras `.keras` model here as `expression_model.keras`. Do not commit model files unless you have verified their license and redistribution terms. The model must accept a batch shaped `(N, 48, 48, 1)` with grayscale float values in `[0, 1]`, and return seven class probabilities/logits in the order configured in `config.yaml`. Update both input settings and label order to match your model.

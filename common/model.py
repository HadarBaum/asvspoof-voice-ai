"""
Loads the model artifact pipeline/train_classifier.py saves - a dict bundling
the fitted classifier with its own EER-optimal decision threshold - and applies
it consistently everywhere a clip needs to be scored (the streaming consumer,
the web app's /classify endpoint). Keeping "how do we turn a probability into
a label" in one place means nothing downstream can silently drift back to
scikit-learn's default 0.5 threshold, which is exactly the value that produced
the accuracy-paradox result documented in docs/EXPLANATION.md.
"""

import joblib


class ScoredModel:
    def __init__(self, model, model_name: str, threshold: float):
        self.model = model
        self.model_name = model_name
        self.threshold = threshold

    def classify(self, feature_vector):
        """feature_vector: a single-row DataFrame from common.features.features_to_vector.
        Returns (label, confidence) where label is 'bonafide' or 'spoof'."""
        proba_spoof = float(self.model.predict_proba(feature_vector)[0, 1])
        is_spoof = proba_spoof >= self.threshold
        confidence = proba_spoof if is_spoof else (1.0 - proba_spoof)
        return ("spoof" if is_spoof else "bonafide"), confidence


def load_model(path: str) -> ScoredModel:
    artifact = joblib.load(path)
    return ScoredModel(artifact["model"], artifact["model_name"], artifact["threshold"])

"""
Demo web app. Two things, both operating on real pipeline artifacts:

  /             overview + links
  /classify     upload a clip -> extract_features() -> the SAME saved model
                the batch/streaming pipeline uses -> Human or AI-generated + confidence
  /dashboard    insights pulled live from Elasticsearch (accuracy by attack
                type, confusion matrix, class balance, training metrics) -
                the same aggregations pipeline/insights.py computes, queried
                fresh on each page load rather than reading a cached file

Run with:
    python -m app.server
"""

import json
import os
import sys

import joblib
from flask import Flask, render_template, request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common import es_client  # noqa: E402
from common.features import extract_features, features_to_vector  # noqa: E402

MODEL_PATH = os.environ.get("MODEL_PATH", os.path.join(os.path.dirname(__file__), "..", "models", "voice_classifier.joblib"))
METRICS_PATH = os.path.join(os.path.dirname(__file__), "..", "docs", "training_metrics.json")

app = Flask(__name__)
_model = None


def get_model():
    global _model
    if _model is None:
        _model = joblib.load(MODEL_PATH)
    return _model


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/classify", methods=["GET", "POST"])
def classify():
    result = None
    error = None
    if request.method == "POST":
        uploaded = request.files.get("audio")
        if not uploaded or uploaded.filename == "":
            error = "Please choose an audio file (wav, flac, or mp3)."
        else:
            try:
                audio_bytes = uploaded.read()
                feats = extract_features(audio_bytes)
                vector = features_to_vector(feats)
                clf = get_model()
                proba = clf.predict_proba(vector)[0]  # [P(bonafide), P(spoof)]
                is_spoof = proba[1] >= 0.5
                result = {
                    "label": "AI-generated" if is_spoof else "Human",
                    "confidence": round(float(max(proba)) * 100, 1),
                }
            except Exception as exc:  # noqa: BLE001
                error = f"Could not process that file: {exc}"
    return render_template("classify.html", result=result, error=error)


@app.route("/dashboard")
def dashboard():
    es = es_client.get_client()
    metrics = None
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH) as f:
            metrics = json.load(f)

    n_predictions = 0
    accuracy_by_attack = []
    class_balance = []
    try:
        n_predictions = es.count(index=es_client.PREDICTIONS_INDEX)["count"]
        if n_predictions:
            agg = es.search(
                index=es_client.PREDICTIONS_INDEX,
                size=0,
                aggs={
                    "by_attack": {
                        "terms": {"field": "attack_id", "size": 25},
                        "aggs": {"accuracy": {"avg": {"script": "doc['correct'].value ? 1 : 0"}}},
                    },
                    "by_true_label": {"terms": {"field": "true_label"}},
                },
            )["aggregations"]
            accuracy_by_attack = sorted(
                (
                    {"attack_id": b["key"], "count": b["doc_count"], "accuracy": round(b["accuracy"]["value"] * 100, 1)}
                    for b in agg["by_attack"]["buckets"]
                ),
                key=lambda r: r["attack_id"],
            )
            class_balance = [{"label": b["key"], "count": b["doc_count"]} for b in agg["by_true_label"]["buckets"]]
    except Exception as exc:  # noqa: BLE001 - dashboard should still render if ES/pipeline hasn't run yet
        print(f"WARN: dashboard aggregation failed: {exc}")

    return render_template(
        "dashboard.html",
        metrics=metrics,
        n_predictions=n_predictions,
        accuracy_by_attack=accuracy_by_attack,
        class_balance=class_balance,
    )


if __name__ == "__main__":
    # debug=True enables Werkzeug's interactive debugger, which allows arbitrary
    # code execution from the browser - fine on localhost-only, never alongside
    # host="0.0.0.0". Default to localhost + debug off; opt into either explicitly.
    debug = os.environ.get("FLASK_DEBUG") == "1"
    host = "0.0.0.0" if os.environ.get("FLASK_EXPOSE") == "1" else "127.0.0.1"
    app.run(host=host, port=5000, debug=debug)

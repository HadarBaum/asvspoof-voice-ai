"""
Step 3 of the pipeline: train the AI capability itself - a bonafide-vs-spoof
("human"-vs-"AI-generated") classifier - on the Parquet feature table produced
by batch_feature_extraction.py.

A Random Forest on the acoustic summary-statistics features is deliberately
simple: it's fast to train, easy to reason about (feature importances are
directly inspectable), and - per the project brief - a simpler model you
fully understand and can defend beats a more complex one you can't.

Train/dev split follows the dataset's own partitions (ASVspoof2019 ships
train and dev as separate speaker-disjoint sets), not a random split -
this matters for anti-spoofing specifically, since a random split could leak
the same speaker/attack style into both sides and inflate accuracy.

Usage:
    python -m pipeline.train_classifier --features pipeline/features_train_dev.parquet --model-out models/voice_classifier.joblib
"""

import argparse
import json
import os
from datetime import datetime, timezone

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score

from common import es_client
from common.features import FEATURE_NAMES


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True, help="Parquet feature table path")
    parser.add_argument("--model-out", required=True)
    parser.add_argument("--metrics-out", default=os.path.join(os.path.dirname(__file__), "..", "docs", "training_metrics.json"))
    args = parser.parse_args()

    df = pd.read_parquet(args.features)
    df["y"] = (df["label"] == "spoof").astype(int)  # 1 = AI-generated, 0 = human

    train_df = df[df["partition"] == "train"]
    dev_df = df[df["partition"] == "dev"]
    if train_df.empty or dev_df.empty:
        raise ValueError("Feature table must contain both 'train' and 'dev' partitions.")

    X_train, y_train = train_df[FEATURE_NAMES], train_df["y"]
    X_dev, y_dev = dev_df[FEATURE_NAMES], dev_df["y"]

    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        class_weight="balanced",  # spoof attacks heavily outnumber bonafide clips in ASVspoof2019
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_dev)
    metrics = {
        "run_id": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "n_train": len(train_df),
        "n_dev": len(dev_df),
        "accuracy_dev": accuracy_score(y_dev, y_pred),
        "precision_dev": precision_score(y_dev, y_pred),
        "recall_dev": recall_score(y_dev, y_pred),
        "f1_dev": f1_score(y_dev, y_pred),
    }

    print(json.dumps(metrics, indent=2))
    print(classification_report(y_dev, y_pred, target_names=["bonafide (human)", "spoof (AI)"]))

    feature_importance = sorted(
        zip(FEATURE_NAMES, clf.feature_importances_), key=lambda kv: kv[1], reverse=True
    )[:10]
    print("Top 10 most important features:")
    for name, importance in feature_importance:
        print(f"  {name}: {importance:.4f}")

    os.makedirs(os.path.dirname(args.model_out), exist_ok=True)
    joblib.dump(clf, args.model_out)
    print(f"Saved model to {args.model_out}")

    os.makedirs(os.path.dirname(args.metrics_out), exist_ok=True)
    with open(args.metrics_out, "w") as f:
        json.dump({**metrics, "top_features": feature_importance}, f, indent=2)

    es = es_client.get_client()
    es_client.ensure_indices(es)
    es_client.bulk_index(es, es_client.TRAINING_RESULTS_INDEX, [metrics], id_field="run_id")
    print(f"Indexed training run into Elasticsearch ({es_client.TRAINING_RESULTS_INDEX}).")


if __name__ == "__main__":
    main()

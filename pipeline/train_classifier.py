"""
Step 3 of the pipeline: train the AI capability itself - a bonafide-vs-spoof
("human"-vs-"AI-generated") classifier - on the Parquet feature table produced
by batch_feature_extraction.py.

Trains two candidate models (Random Forest and Gradient Boosting) on the same
features and picks the one with the lower EER. EER was an official secondary
countermeasure evaluation metric in ASVspoof2019 (min t-DCF was the primary
challenge metric), and it is a useful choice here because it directly balances
the two classification error types on an imbalanced dev set. Both are simple enough to fully explain in
Q&A (unlike, say, a CNN over raw spectrograms) while giving a real point of
comparison for the results write-up rather than a single unchallenged number.

The saved model artifact bundles the classifier together with its own
EER-optimal decision threshold (see `compute_eer` below) - every downstream
consumer (the streaming enrichment consumer, the web app's /classify
endpoint) loads both from one file and never hardcodes a threshold itself.

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
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.utils.class_weight import compute_sample_weight

from common import es_client
from common.features import FEATURE_NAMES

CHARTS_DIR = os.path.join(os.path.dirname(__file__), "..", "app", "static", "charts")


def compute_eer(y_true_spoof, y_score_spoof):
    """Compute Equal Error Rate (EER) for scores where 1 = spoof and higher
    scores indicate greater support for the spoof class.

    With this convention:
      - FPR = bonafide clips incorrectly classified as spoof.
      - FNR = spoof clips incorrectly classified as bonafide.

    EER is the operating point where these two error rates are equal, or as
    close as possible on the finite ROC curve. EER was an official secondary
    countermeasure evaluation metric in ASVspoof2019; min t-DCF was the primary
    challenge metric.

    Returns the EER, the corresponding P(spoof) decision threshold, and the
    ROC-curve arrays used for plotting.
    """
    fpr, tpr, thresholds = roc_curve(y_true_spoof, y_score_spoof)
    fnr = 1 - tpr
    eer_idx = np.nanargmin(np.abs(fnr - fpr))
    eer = float((fpr[eer_idx] + fnr[eer_idx]) / 2.0)
    return {
        "eer": eer,
        "threshold": float(thresholds[eer_idx]),
        "fpr": fpr,
        "tpr": tpr,
        "eer_idx": int(eer_idx),
    }


def evaluate_at_threshold(y_dev, y_proba_spoof, threshold):
    """Both classes, both directions, at one operating point.

    The bonafide *precision* and F1 entries were missing here originally, which meant
    the cost of moving off the 0.5 threshold was not recorded anywhere: only the
    bonafide recall gain (80.2% -> 90.7%) was persisted, not the precision drop that
    pays for it (69.1% -> 52.8%). The deck's threshold trade-off table needs both
    sides to be honest, and balanced_accuracy is the one headline that improves,
    so it is stored rather than recomputed by every consumer.
    """
    y_pred = (y_proba_spoof >= threshold).astype(int)
    return {
        "threshold": threshold,
        "accuracy": accuracy_score(y_dev, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_dev, y_pred),
        "precision_spoof": precision_score(y_dev, y_pred, zero_division=0),
        "recall_spoof": recall_score(y_dev, y_pred, zero_division=0),
        "f1_spoof": f1_score(y_dev, y_pred, zero_division=0),
        "precision_bonafide": precision_score(y_dev, y_pred, pos_label=0, zero_division=0),
        "recall_bonafide": recall_score(y_dev, y_pred, pos_label=0, zero_division=0),
        "f1_bonafide": f1_score(y_dev, y_pred, pos_label=0, zero_division=0),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True, help="Parquet feature table path")
    parser.add_argument("--model-out", required=True)
    parser.add_argument("--metrics-out", default=os.path.join(os.path.dirname(__file__), "..", "docs", "training_metrics.json"))
    parser.add_argument(
        "--comparison-out", default=os.path.join(os.path.dirname(__file__), "..", "docs", "model_comparison.json")
    )
    args = parser.parse_args()

    df = pd.read_parquet(args.features)
    df["y"] = (df["label"] == "spoof").astype(int)  # 1 = AI-generated, 0 = human

    train_df = df[df["partition"] == "train"]
    dev_df = df[df["partition"] == "dev"]
    if train_df.empty or dev_df.empty:
        raise ValueError("Feature table must contain both 'train' and 'dev' partitions.")

    X_train, y_train = train_df[FEATURE_NAMES], train_df["y"]
    X_dev, y_dev = dev_df[FEATURE_NAMES], dev_df["y"]

    # RandomForestClassifier has a class_weight option, but
    # GradientBoostingClassifier does not. To treat both models consistently,
    # we reweight both using the same per-sample weights.
    sample_weight_train = compute_sample_weight("balanced", y_train)

    candidates = {
        "random_forest": RandomForestClassifier(
            n_estimators=200, max_depth=None, random_state=42, n_jobs=-1
        ),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=150, max_depth=3, learning_rate=0.1, random_state=42
        ),
    }

    comparison = {}
    fitted = {}
    for name, clf in candidates.items():
        print(f"Training {name} ...")
        clf.fit(X_train, y_train, sample_weight=sample_weight_train)
        fitted[name] = clf

        y_proba = clf.predict_proba(X_dev)[:, 1]
        eer_result = compute_eer(y_dev, y_proba)
        eer, eer_threshold = eer_result["eer"], eer_result["threshold"]
        comparison[name] = {
            "roc_auc": roc_auc_score(y_dev, y_proba),
            "eer": eer,
            "eer_threshold": eer_threshold,
            "roc_fpr": eer_result["fpr"],
            "roc_tpr": eer_result["tpr"],
            "roc_eer_idx": eer_result["eer_idx"],
            # at the default 0.5 threshold scikit-learn's own .predict() would use -
            # kept to make the accuracy-paradox comparison concrete per model
            "at_default_threshold_0.5": evaluate_at_threshold(y_dev, y_proba, 0.5),
            # at this model's own EER threshold - the operating point actually used
            "at_eer_threshold": evaluate_at_threshold(y_dev, y_proba, eer_threshold),
        }
        print(f"  {name}: EER={eer:.4f}  ROC-AUC={comparison[name]['roc_auc']:.4f}")

    best_name = min(comparison, key=lambda n: comparison[n]["eer"])
    best_clf = fitted[best_name]
    best = comparison[best_name]
    print(f"\nSelected '{best_name}' (lowest EER = {best['eer']:.4f}) as the deployed model.")

    y_pred_best = (best_clf.predict_proba(X_dev)[:, 1] >= best["eer_threshold"]).astype(int)
    print(f"\nClassification report for '{best_name}' at its EER threshold ({best['eer_threshold']:.3f}):")
    print(classification_report(y_dev, y_pred_best, target_names=["bonafide (human)", "spoof (AI)"], zero_division=0))

    feature_importance = []
    if hasattr(best_clf, "feature_importances_"):
        feature_importance = sorted(
            zip(FEATURE_NAMES, best_clf.feature_importances_), key=lambda kv: kv[1], reverse=True
        )[:10]
        print("Top 10 most important features:")
        for name, importance in feature_importance:
            print(f"  {name}: {importance:.4f}")

    os.makedirs(CHARTS_DIR, exist_ok=True)

    # --- ROC curve, both candidates, EER point marked on the deployed model ---
    plt.figure(figsize=(5, 5))
    for name in comparison:
        style = "-" if name == best_name else "--"
        plt.plot(
            comparison[name]["roc_fpr"],
            comparison[name]["roc_tpr"],
            style,
            label=f"{name.replace('_', ' ').title()} (AUC={comparison[name]['roc_auc']:.3f})",
        )
    idx = best["roc_eer_idx"]
    plt.plot(best["roc_fpr"][idx], best["roc_tpr"][idx], "ko", markersize=8, label=f"EER = {best['eer']:.1%}")
    plt.plot([0, 1], [0, 1], ":", color="gray", linewidth=1)
    plt.xlabel("False positive rate (bonafide called spoof)")
    plt.ylabel("True positive rate (spoof caught)")
    plt.title("ROC curve - deployed model's EER point marked")
    plt.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(CHARTS_DIR, "roc_curve.png"), dpi=120)
    plt.close()

    # --- feature importance, the deployed model only ---
    if feature_importance:
        names, importances = zip(*reversed(feature_importance))
        plt.figure(figsize=(6, 4))
        plt.barh(names, importances, color="#4C72B0")
        plt.xlabel("Importance")
        plt.title(f"Top 10 features - {best_name.replace('_', ' ').title()}")
        plt.tight_layout()
        plt.savefig(os.path.join(CHARTS_DIR, "feature_importance.png"), dpi=120)
        plt.close()

    os.makedirs(os.path.dirname(args.model_out), exist_ok=True)
    # Bundle the model with its own decision threshold so nothing downstream
    # (the streaming consumer, the web app) ever has to hardcode 0.5 again.
    joblib.dump({"model": best_clf, "model_name": best_name, "threshold": best["eer_threshold"]}, args.model_out)
    print(f"Saved model to {args.model_out}")

    metrics = {
        "run_id": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "n_train": len(train_df),
        "n_dev": len(dev_df),
        "model_name": best_name,
        "decision_threshold": best["eer_threshold"],
        "roc_auc_dev": best["roc_auc"],
        "eer_dev": best["eer"],
        "accuracy_dev": best["at_eer_threshold"]["accuracy"],
        "precision_dev": best["at_eer_threshold"]["precision_spoof"],
        "recall_dev": best["at_eer_threshold"]["recall_spoof"],
        "recall_bonafide_dev": best["at_eer_threshold"]["recall_bonafide"],
        "f1_dev": best["at_eer_threshold"]["f1_spoof"],
        # kept so the accuracy-paradox story stays backed by real numbers even
        # after the threshold fix is applied (see the dashboard's comparison table)
        "accuracy_dev_default_threshold_0.5": best["at_default_threshold_0.5"]["accuracy"],
        "recall_bonafide_dev_default_threshold_0.5": best["at_default_threshold_0.5"]["recall_bonafide"],
        "recall_dev_default_threshold_0.5": best["at_default_threshold_0.5"]["recall_spoof"],
        "f1_dev_default_threshold_0.5": best["at_default_threshold_0.5"]["f1_spoof"],
    }

    os.makedirs(os.path.dirname(args.metrics_out), exist_ok=True)
    with open(args.metrics_out, "w") as f:
        json.dump({**metrics, "top_features": feature_importance}, f, indent=2)

    # roc_fpr/roc_tpr are numpy arrays kept around only to plot the ROC curve
    # above - not JSON-serializable, and not useful in a hand-readable
    # comparison file, so they're dropped before writing it out.
    json_safe_comparison = {
        name: {k: v for k, v in c.items() if k not in ("roc_fpr", "roc_tpr", "roc_eer_idx")}
        for name, c in comparison.items()
    }
    with open(args.comparison_out, "w") as f:
        json.dump({"selected": best_name, "candidates": json_safe_comparison}, f, indent=2)
    print(f"Wrote model comparison to {args.comparison_out}")

    es = es_client.get_client()
    es_client.ensure_indices(es)
    es_client.bulk_index(es, es_client.TRAINING_RESULTS_INDEX, [metrics], id_field="run_id")
    print(f"Indexed training run into Elasticsearch ({es_client.TRAINING_RESULTS_INDEX}).")


if __name__ == "__main__":
    main()

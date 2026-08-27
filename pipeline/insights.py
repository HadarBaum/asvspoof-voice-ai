"""
Step 5 of the pipeline: turn what's in Elasticsearch into the "results and
insights" deliverable - overall + per-attack-type detection accuracy, a
confusion matrix, a confidence histogram, and class balance - as both a
markdown report (docs/RESULTS.md) and PNG charts the web dashboard reuses.

Everything here is a straight Elasticsearch aggregation query; no numbers
are invented or hand-typed, so this script can be re-run after any pipeline
run (sample or full-scale) to regenerate real results.

Usage:
    python -m pipeline.insights
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import es_client

DOCS_DIR = os.path.join(os.path.dirname(__file__), "..", "docs")
CHARTS_DIR = os.path.join(os.path.dirname(__file__), "..", "app", "static", "charts")


def _agg(es, index, body):
    return es.search(index=index, size=0, aggs=body)["aggregations"]


def main():
    os.makedirs(CHARTS_DIR, exist_ok=True)
    es = es_client.get_client()

    n_predictions = es.count(index=es_client.PREDICTIONS_INDEX)["count"]
    if n_predictions == 0:
        print("No documents in the predictions index yet - run the streaming pipeline first.")
        return

    overall = _agg(
        es,
        es_client.PREDICTIONS_INDEX,
        {"accuracy": {"avg": {"script": "doc['correct'].value ? 1 : 0"}}},
    )
    overall_accuracy = overall["accuracy"]["value"]

    by_attack = _agg(
        es,
        es_client.PREDICTIONS_INDEX,
        {
            "by_attack": {
                "terms": {"field": "attack_id", "size": 25},
                "aggs": {"accuracy": {"avg": {"script": "doc['correct'].value ? 1 : 0"}}},
            }
        },
    )["by_attack"]["buckets"]

    class_balance = _agg(
        es, es_client.PREDICTIONS_INDEX, {"by_true_label": {"terms": {"field": "true_label"}}}
    )["by_true_label"]["buckets"]

    confusion = _agg(
        es,
        es_client.PREDICTIONS_INDEX,
        {
            "by_true": {
                "terms": {"field": "true_label"},
                "aggs": {"by_pred": {"terms": {"field": "predicted_label"}}},
            }
        },
    )["by_true"]["buckets"]

    # --- charts -----------------------------------------------------------
    attack_ids = [b["key"] for b in by_attack]
    attack_acc = [b["accuracy"]["value"] for b in by_attack]
    plt.figure(figsize=(8, 4))
    plt.bar(attack_ids, attack_acc, color="#4C72B0")
    plt.ylabel("Detection accuracy")
    plt.title("Detection accuracy by attack type (- = bonafide)")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(os.path.join(CHARTS_DIR, "accuracy_by_attack.png"), dpi=120)
    plt.close()

    plt.figure(figsize=(4, 4))
    plt.bar([b["key"] for b in class_balance], [b["doc_count"] for b in class_balance], color="#55A868")
    plt.title("Streamed utterances by true label")
    plt.tight_layout()
    plt.savefig(os.path.join(CHARTS_DIR, "class_balance.png"), dpi=120)
    plt.close()

    # --- confusion matrix heatmap ------------------------------------------
    labels = ["bonafide", "spoof"]
    matrix = np.zeros((2, 2), dtype=int)  # rows = true label, cols = predicted label
    for true_bucket in confusion:
        true_idx = labels.index(true_bucket["key"])
        for pred_bucket in true_bucket["by_pred"]["buckets"]:
            if pred_bucket["key"] in labels:
                matrix[true_idx, labels.index(pred_bucket["key"])] = pred_bucket["doc_count"]

    plt.figure(figsize=(4, 4))
    plt.imshow(matrix, cmap="Blues")
    plt.xticks([0, 1], labels)
    plt.yticks([0, 1], labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion matrix (streamed predictions)")
    for i in range(2):
        for j in range(2):
            color = "white" if matrix[i, j] > matrix.max() / 2 else "black"
            plt.text(j, i, str(matrix[i, j]), ha="center", va="center", color=color, fontsize=13)
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(os.path.join(CHARTS_DIR, "confusion_matrix.png"), dpi=120)
    plt.close()

    # --- confidence histogram, correct vs incorrect predictions ------------
    conf_hits = es.search(
        index=es_client.PREDICTIONS_INDEX,
        size=min(n_predictions, 10000),
        source=["confidence", "correct"],
        query={"match_all": {}},
    )["hits"]["hits"]
    correct_conf = [h["_source"]["confidence"] for h in conf_hits if h["_source"]["correct"]]
    incorrect_conf = [h["_source"]["confidence"] for h in conf_hits if not h["_source"]["correct"]]

    plt.figure(figsize=(6, 4))
    bins = np.linspace(0.5, 1.0, 21)  # confidence is always >= 0.5 by construction (see common/model.py)
    plt.hist(correct_conf, bins=bins, alpha=0.7, label=f"Correct (n={len(correct_conf)})", color="#55A868")
    plt.hist(incorrect_conf, bins=bins, alpha=0.7, label=f"Incorrect (n={len(incorrect_conf)})", color="#C44E52")
    plt.xlabel("Model confidence")
    plt.ylabel("Utterances")
    plt.title("Confidence distribution: correct vs. incorrect predictions")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(CHARTS_DIR, "confidence_histogram.png"), dpi=120)
    plt.close()

    # --- per-class recall and balanced accuracy -----------------------------
    # Overall accuracy is actively misleading on this partition: eval is ~90% spoof,
    # so a trivial "always predict spoof" baseline scores about as well as a real
    # model does. This is the same imbalance trap the decision threshold already had
    # to be fixed for (see docs/EXPLANATION.md §4) - it just reappears in the report
    # if the report leads with raw accuracy. Balanced accuracy (the unweighted mean
    # of the per-class recalls) is what separates the model from that baseline, so
    # it is what RESULTS.md leads with; raw accuracy is still reported, second.
    per_class_recall = {}
    for idx, label in enumerate(labels):
        support = matrix[idx].sum()
        per_class_recall[label] = matrix[idx, idx] / support if support else float("nan")
    balanced_accuracy = float(np.nanmean(list(per_class_recall.values())))

    support_per_class = matrix.sum(axis=1)
    majority_label = labels[int(support_per_class.argmax())]
    majority_share = support_per_class.max() / matrix.sum() if matrix.sum() else float("nan")

    # --- markdown report ----------------------------------------------------
    lines = [
        "# Results and insights",
        "",
        f"Generated from {n_predictions} utterances scored by the streaming enrichment pipeline.",
    ]
    if n_predictions < 1000:
        lines.append(
            "\n*Small sample size - either this ran against the committed synthetic/sample "
            "dataset (see `pipeline/dev_generate_synthetic_sample.py`) rather than the full "
            "ASVspoof2019 dataset, or the streaming run against the real data was deliberately "
            "capped short of the full eval queue (see docs/EXPLANATION.md's streaming trade-off "
            "note for why). Either way, treat per-attack-type breakdowns with correspondingly "
            "small per-bucket counts as directional, not precise.*"
        )
    lines += [
        "",
        f"**Balanced streaming detection accuracy: {balanced_accuracy:.1%}** "
        f"(bonafide recall {per_class_recall['bonafide']:.1%}, "
        f"spoof recall {per_class_recall['spoof']:.1%})",
        "",
        f"Overall (unbalanced) accuracy: {overall_accuracy:.1%}. This is reported second on "
        f"purpose - {majority_share:.1%} of these utterances are `{majority_label}`, so a "
        f'trivial "always predict {majority_label}" baseline would also score '
        f"{majority_share:.1%} while learning nothing. Balanced accuracy above is the number "
        "that actually distinguishes this model from that baseline, and is the one to quote.",
        "",
        "## Accuracy by attack type",
        "",
        "| Attack ID | Utterances | Accuracy |",
        "|---|---|---|",
    ]
    for b in sorted(by_attack, key=lambda b: b["key"]):
        lines.append(f"| {b['key']} | {b['doc_count']} | {b['accuracy']['value']:.1%} |")

    lines += ["", "## Confusion matrix (true label -> predicted label counts)", "", "| True \\ Predicted | bonafide | spoof |", "|---|---|---|"]
    for true_bucket in confusion:
        row = {b["key"]: b["doc_count"] for b in true_bucket["by_pred"]["buckets"]}
        lines.append(f"| {true_bucket['key']} | {row.get('bonafide', 0)} | {row.get('spoof', 0)} |")

    lines += ["", "## Class balance (streamed eval utterances)", "", "| True label | Count |", "|---|---|"]
    for b in class_balance:
        lines.append(f"| {b['key']} | {b['doc_count']} |")

    lines += [
        "",
        "## Visualizations",
        "",
        "Six charts, all regenerated from the data above (and from the training run - see "
        "`docs/training_metrics.json` / `docs/model_comparison.json`), live in "
        "`app/static/charts/` and render on the `/dashboard` page:",
"`accuracy_by_attack.png`, `class_balance.png`, `confusion_matrix.png`, and "
        "`confidence_histogram.png` (written by this script), plus `roc_curve.png` and "
        "`feature_importance.png` (written by `pipeline/train_classifier.py` instead, since "
        "those need the model's dev-set predictions rather than the streamed predictions "
        "this script reads).",
    ]

    out_path = os.path.join(DOCS_DIR, "RESULTS.md")
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Wrote {out_path}")
    print(f"Wrote charts to {CHARTS_DIR}")


if __name__ == "__main__":
    main()

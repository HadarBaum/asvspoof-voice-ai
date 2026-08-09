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

    # --- markdown report ----------------------------------------------------
    lines = [
        "# Results and insights",
        "",
        f"Generated from {n_predictions} utterances scored by the streaming enrichment pipeline.",
    ]
    if n_predictions < 1000:
        lines.append(
            "\n*Small sample size - this run used the committed synthetic/sample "
            "dataset (see `pipeline/dev_generate_synthetic_sample.py`), not the full "
            "ASVspoof2019 dataset. Re-run the pipeline against `data/raw/LA` for real numbers.*"
        )
    lines += [
        "",
        f"**Overall streaming detection accuracy: {overall_accuracy:.1%}**",
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

    out_path = os.path.join(DOCS_DIR, "RESULTS.md")
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Wrote {out_path}")
    print(f"Wrote charts to {CHARTS_DIR}")


if __name__ == "__main__":
    main()

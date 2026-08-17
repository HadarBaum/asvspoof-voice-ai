# Results and insights

Generated from 498 utterances scored by the streaming enrichment pipeline.

*Small sample size - either this ran against the committed synthetic/sample dataset (see `pipeline/dev_generate_synthetic_sample.py`) rather than the full ASVspoof2019 dataset, or the streaming run against the real data was deliberately capped short of the full eval queue (see docs/EXPLANATION.md's streaming trade-off note for why). Either way, treat per-attack-type breakdowns with correspondingly small per-bucket counts as directional, not precise.*

**Overall streaming detection accuracy: 84.9%**

## Accuracy by attack type

| Attack ID | Utterances | Accuracy |
|---|---|---|
| - | 59 | 91.5% |
| A07 | 33 | 100.0% |
| A08 | 32 | 100.0% |
| A09 | 31 | 100.0% |
| A10 | 39 | 100.0% |
| A11 | 30 | 100.0% |
| A12 | 38 | 86.8% |
| A13 | 35 | 100.0% |
| A14 | 34 | 100.0% |
| A15 | 31 | 96.8% |
| A16 | 43 | 83.7% |
| A17 | 38 | 13.2% |
| A18 | 30 | 63.3% |
| A19 | 25 | 48.0% |

## Confusion matrix (true label -> predicted label counts)

| True \ Predicted | bonafide | spoof |
|---|---|---|
| spoof | 70 | 369 |
| bonafide | 54 | 5 |

## Class balance (streamed eval utterances)

| True label | Count |
|---|---|
| spoof | 439 |
| bonafide | 59 |

## Visualizations

Six charts, all regenerated from the data above (and from the training run - see `docs/training_metrics.json` / `docs/model_comparison.json`), live in `app/static/charts/` and render on the `/dashboard` page:
`accuracy_by_attack.png`, `class_balance.png`, `confusion_matrix.png`, and `confidence_histogram.png` (written by this script), plus `roc_curve.png` and `feature_importance.png` (written by `pipeline/train_classifier.py` instead, since those need the model's dev-set predictions rather than the streamed predictions this script reads).

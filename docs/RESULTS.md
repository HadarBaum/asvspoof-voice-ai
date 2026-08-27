# Results and insights

Generated from 2000 utterances scored by the streaming enrichment pipeline.

**Balanced streaming detection accuracy: 89.3%** (bonafide recall 95.3%, spoof recall 83.3%)

Overall (unbalanced) accuracy: 84.8%. This is reported second on purpose - 88.2% of these utterances are `spoof`, so a trivial "always predict spoof" baseline would also score 88.2% while learning nothing. Balanced accuracy above is the number that actually distinguishes this model from that baseline, and is the one to quote.

## Accuracy by attack type

| Attack ID | Utterances | Accuracy |
|---|---|---|
| - | 236 | 95.3% |
| A07 | 142 | 98.6% |
| A08 | 135 | 100.0% |
| A09 | 131 | 100.0% |
| A10 | 128 | 100.0% |
| A11 | 138 | 99.3% |
| A12 | 155 | 88.4% |
| A13 | 146 | 100.0% |
| A14 | 135 | 99.3% |
| A15 | 122 | 95.1% |
| A16 | 152 | 84.2% |
| A17 | 132 | 15.9% |
| A18 | 119 | 52.9% |
| A19 | 129 | 41.9% |

## Confusion matrix (true label -> predicted label counts)

| True \ Predicted | bonafide | spoof |
|---|---|---|
| spoof | 294 | 1470 |
| bonafide | 225 | 11 |

## Class balance (streamed eval utterances)

| True label | Count |
|---|---|
| spoof | 1764 |
| bonafide | 236 |

## Visualizations

Six charts, all regenerated from the data above (and from the training run - see `docs/training_metrics.json` / `docs/model_comparison.json`), live in `app/static/charts/` and render on the `/dashboard` page:
`accuracy_by_attack.png`, `class_balance.png`, `confusion_matrix.png`, and `confidence_histogram.png` (written by this script), plus `roc_curve.png` and `feature_importance.png` (written by `pipeline/train_classifier.py` instead, since those need the model's dev-set predictions rather than the streamed predictions this script reads).

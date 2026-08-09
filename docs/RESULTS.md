# Results and insights

Generated from 20 utterances scored by the streaming enrichment pipeline.

*Small sample size - this run used the committed synthetic/sample dataset (see `pipeline/dev_generate_synthetic_sample.py`), not the full ASVspoof2019 dataset. Re-run the pipeline against `data/raw/LA` for real numbers.*

**Overall streaming detection accuracy: 100.0%**

## Accuracy by attack type

| Attack ID | Utterances | Accuracy |
|---|---|---|
| - | 10 | 100.0% |
| A01 | 1 | 100.0% |
| A02 | 3 | 100.0% |
| A03 | 2 | 100.0% |
| A04 | 1 | 100.0% |
| A05 | 2 | 100.0% |
| A06 | 1 | 100.0% |

## Confusion matrix (true label -> predicted label counts)

| True \ Predicted | bonafide | spoof |
|---|---|---|
| bonafide | 10 | 0 |
| spoof | 0 | 10 |

## Class balance (streamed eval utterances)

| True label | Count |
|---|---|
| bonafide | 10 |
| spoof | 10 |

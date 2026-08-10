# Results and insights

Generated from 1165 utterances scored by the streaming enrichment pipeline.

**Overall streaming detection accuracy: 87.8%**

## Accuracy by attack type

| Attack ID | Utterances | Accuracy |
|---|---|---|
| - | 145 | 29.7% |
| A07 | 81 | 100.0% |
| A08 | 79 | 100.0% |
| A09 | 75 | 100.0% |
| A10 | 75 | 100.0% |
| A11 | 82 | 100.0% |
| A12 | 86 | 100.0% |
| A13 | 88 | 100.0% |
| A14 | 77 | 100.0% |
| A15 | 71 | 100.0% |
| A16 | 95 | 98.9% |
| A17 | 76 | 56.6% |
| A18 | 66 | 100.0% |
| A19 | 69 | 91.3% |

## Confusion matrix (true label -> predicted label counts)

| True \ Predicted | bonafide | spoof |
|---|---|---|
| spoof | 40 | 980 |
| bonafide | 43 | 102 |

## Class balance (streamed eval utterances)

| True label | Count |
|---|---|
| spoof | 1020 |
| bonafide | 145 |

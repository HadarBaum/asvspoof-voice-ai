# Human-or-AI Voice Detector

BIU Big Data and AI course project. A bonafide-vs-spoof (human-vs-AI-generated speech)
classifier built on an ETL/streaming pipeline over the
[ASVspoof2019](https://www.kaggle.com/datasets/awsaf49/asvpoof-2019-dataset) dataset
(Logical Access partition: real speech vs. TTS/voice-conversion synthetic speech).

See [`docs/DESIGN.md`](docs/DESIGN.md) for the architecture and
[`docs/EXPLANATION.md`](docs/EXPLANATION.md) for a full walkthrough of every
component (written so any part of it can be defended in Q&A, per the course brief).

## Prerequisites

- Python 3.12, Docker Desktop, a JDK (17+) for PySpark
- A Kaggle account + API token (`kaggle.json`) — only needed for step 1 below

**Run everything from a path with no spaces.** PySpark's worker-process
spawning breaks on Windows when the project path contains spaces (this bit us
once already — see `docs/EXPLANATION.md`).

## Setup

```bash
python -m venv .venv
.venv/Scripts/activate            # .venv/bin/activate on macOS/Linux
pip install -r requirements.txt
docker compose up -d              # MinIO (9000/9001), Kafka (9092), Elasticsearch (9200)
```

## Running the pipeline

Every command below is `python -m pipeline.<script> ...` run from the repo root.
Use `--la-root data/sample/LA` to try everything against the small committed
sample first (no download needed), or `--la-root data/raw/LA` once you've
downloaded the real dataset.

```bash
# 0. (one-time) download the real dataset - needs ~/.kaggle/kaggle.json
python -m pipeline.download_dataset

# 1. upload the raw flac corpus to MinIO
python -m pipeline.ingest_to_minio --la-root data/sample/LA

# 2. Spark batch job: features for train+dev -> Parquet + Elasticsearch
python -m pipeline.batch_feature_extraction --la-root data/sample/LA --partitions train dev --out pipeline/features_train_dev.parquet

# 3. train the classifier (compares Random Forest vs Gradient Boosting, keeps the lower-EER one)
python -m pipeline.train_classifier --features pipeline/features_train_dev.parquet --model-out models/voice_classifier.joblib

# 3b. index acoustic-feature embeddings for the web app's similarity search (any order vs. step 3)
python -m pipeline.index_embeddings --features pipeline/features_train_dev.parquet

# 4. streaming enrichment - run these two in separate terminals
# (use a new --group-id whenever you swap in a newly-trained model, so the whole
#  topic gets re-scored from the start instead of resuming from an old offset)
python -m pipeline.streaming_enrichment --model models/voice_classifier.joblib --group-id asvspoof-streaming-enrichment
python -m pipeline.kafka_producer --la-root data/sample/LA --limit 200 --delay-seconds 0.1

# 5. generate insights (charts + docs/RESULTS.md) from what's in Elasticsearch
python -m pipeline.insights

# 6. regenerate the slide deck from the latest metrics/comparison/charts
python docs/slides/generate_slides.py

# 7. run the demo web app
python -m app.server        # http://localhost:5000
```

## Demo

- `http://localhost:5000/classify` — upload a clip *or record one live from your
  microphone*, get Human / AI-generated + confidence, plus the 5 most acoustically
  similar training clips (Elasticsearch k-NN search), each one **playable** right
  in the results
- `http://localhost:5000/dashboard` — live insights from Elasticsearch: a
  Random-Forest-vs-Gradient-Boosting comparison table, the deployed model's
  metrics at its EER threshold vs. what the default 0.5 threshold would give,
  and six charts (accuracy by attack, class balance, ROC curve, feature
  importance, confusion matrix, confidence calibration)

## Repository layout

```
common/     shared code: feature extraction, MinIO client, Elasticsearch client, dataset layout
pipeline/   the ETL + AI pipeline scripts, run in the order shown above
app/        Flask demo web app
docs/       design doc, full explanation, generated results, slides
data/       data/sample (committed) and data/raw (gitignored, populated by download_dataset.py)
models/     trained classifier artifact
```

## Credits

Dataset: Wang, X. et al., "ASVspoof2019," via
[Kaggle](https://www.kaggle.com/datasets/awsaf49/asvpoof-2019-dataset).

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

# 3. train the classifier
python -m pipeline.train_classifier --features pipeline/features_train_dev.parquet --model-out models/voice_classifier.joblib

# 4. streaming enrichment - run these two in separate terminals
python -m pipeline.streaming_enrichment --model models/voice_classifier.joblib
python -m pipeline.kafka_producer --la-root data/sample/LA --limit 200 --delay-seconds 0.1

# 5. generate insights (charts + docs/RESULTS.md) from what's in Elasticsearch
python -m pipeline.insights

# 6. run the demo web app
python -m app.server        # http://localhost:5000
```

## Demo

- `http://localhost:5000/classify` — upload a clip, get Human / AI-generated + confidence
- `http://localhost:5000/dashboard` — live insights from Elasticsearch

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

# Explanation — every part of this project, in detail

This document exists so any line of code or design choice in this submission can be
explained and defended, per the course brief's emphasis on understanding over
complexity. It walks through every file, why it exists, what it does, and the
trade-offs behind it, then maps the project onto the grading rubric.

## 1. The problem and the dataset

[ASVspoof2019](https://www.kaggle.com/datasets/awsaf49/asvpoof-2019-dataset) is an
anti-spoofing benchmark for automatic speaker verification. It has two partitions:

- **LA (Logical Access)**: bonafide (real human) speech vs. spoofed speech generated
  by text-to-speech (TTS) or voice-conversion (VC) systems — i.e. genuinely
  **AI-generated** audio. Each spoof clip is tagged with an attack ID (A01-A19)
  identifying which TTS/VC system produced it.
- **PA (Physical Access)**: bonafide speech vs. *replayed* recordings (someone plays
  a recording of a real voice back through a speaker near a microphone). This is
  spoofing, but not AI-generated — a human recorded it, a speaker replayed it.

**We use only LA.** "Human-or-AI voice detection" only makes sense against synthetic
audio; PA's replay attacks would silently mislabel "a human recording played back" as
if it were "AI," which is a different (and false) claim.

Each partition ships:
- a `flac/` folder of audio clips (16kHz mono),
- a protocol `.txt` file: one line per clip — `<speaker_id> <file_id> - <attack_id> <bonafide|spoof>`
  (`attack_id` is `-` for bonafide clips).

Train, dev, and eval are **speaker-disjoint** — no speaker appears in more than one
partition. This matters (see §5).

## 2. Repository layout and why it's shaped this way

```
common/       code shared by more than one entry point - written once, used everywhere
pipeline/     one script per pipeline stage, run in order (numbered in README.md)
app/          the Flask demo, imports from common/ - never duplicates pipeline logic
docs/         this file, the design doc, generated results/charts, slide deck
data/         data/sample (committed, synthetic) and data/raw (gitignored, real dataset)
models/       the one trained model artifact every other component reads
```

The guiding rule: **feature extraction and the model live in exactly one place**
(`common/features.py`, `models/voice_classifier.joblib`), used identically by the
batch job, the streaming consumer, and the web app. This is what makes the whole
project "one AI capability applied three ways" rather than three different,
harder-to-defend AI features.

## 3. `common/` — shared code

### `common/features.py`
`extract_features(audio_bytes) -> dict` decodes any audio format librosa can read,
resamples to 16kHz mono, and computes:
- 20 MFCC coefficients (mean + std across frames) — the standard compact
  representation of the spectral envelope (roughly, timbre) of speech,
- spectral centroid / bandwidth / rolloff (mean + std) — describe the shape of the
  frequency spectrum,
- zero-crossing rate (mean + std) — how "noisy"/high-frequency the signal is,
- RMS energy (mean + std),
- pitch (F0) mean + std via `librosa.pyin` — TTS/VC systems often produce
  unnaturally stable pitch (no vibrato), which is one of the more informative
  features (see feature importances in `docs/RESULTS.md`),
- clip duration.

Mean/std of frame-level features (rather than the raw frame sequences) gives every
clip a **fixed-length** vector regardless of duration — required for a standard
tabular classifier. `features_to_vector()` projects a features dict onto the same
column order the model was trained on (`FEATURE_NAMES`), so training and inference
can never silently disagree on column order.

### `common/minio_client.py`
A few functions around `boto3.client("s3", endpoint_url=...)`. MinIO speaks the S3
API, so no MinIO-specific SDK is needed — this is genuinely the same client code
that would talk to real AWS S3, which is part of the point of using an
S3-compatible object store.

### `common/es_client.py`
Wraps `elasticsearch-py`. Defines the three indices the project uses and a small
`bulk_index()` helper (using `elasticsearch.helpers.bulk`) that upserts by a given
id field, so re-running a pipeline stage overwrites existing documents instead of
duplicating them.

### `common/dataset_layout.py`
Knows the ASVspoof2019 directory/protocol conventions: given an `LA/` root and a
partition name, `load_partition()` parses the matching protocol file and returns a
list of `Utterance` dataclasses (speaker, file id, attack id, label, local flac
path). `minio_key()` derives the object key each utterance will have in MinIO.
Protocol files are matched by **glob** (`*.cm.*{partition}*.txt`), not an exact
filename, so the same parsing code works against both the real dataset's naming
(`ASVspoof2019.LA.cm.train.trn.txt`) and `pipeline/dev_generate_synthetic_sample.py`'s
simplified names used for local testing.

## 4. `pipeline/` — the ETL + AI pipeline, stage by stage

### `pipeline/dev_generate_synthetic_sample.py` (dev-only, not graded pipeline logic)
Generates a small fake dataset (sine-wave-based "bonafide" clips with vibrato/noise,
purer/more-stable "spoof" clips) in the exact ASVspoof2019 directory/protocol shape,
so every other script could be written and tested before Kaggle access was
available. It is clearly out of scope of the graded pipeline and is called out as
such in its own docstring; the real numbers in `docs/RESULTS.md` come from a run
against the real dataset.

### `pipeline/download_dataset.py`
Uses the `kaggle` package's API client to download and unzip the dataset into
`data/raw/`. Requires `~/.kaggle/kaggle.json` (a Kaggle API token) — the one step
that needs a manual one-time credential setup, since Kaggle requires an
authenticated account.

### `pipeline/ingest_to_minio.py`
For each requested partition, calls `dataset_layout.load_partition()` then uploads
every clip's local flac file to the `asvspoof-raw` MinIO bucket under a
partition-aware key. This is Part A's "load unstructured data into an object
store" step.

### `pipeline/batch_feature_extraction.py` — the Spark batch job
This is the project's actual "big data" step. Given the labeled utterance list
(from protocol files — small, kept local) for train+dev:
1. Builds a Spark RDD of ~50k rows (`key`, `partition`, `speaker_id`, `attack_id`, `label`),
   repartitioned to roughly 200 rows per partition/task so work actually spreads
   across CPU cores.
2. `mapPartitions(extract_partition)` — runs once **per Spark task**, not once per
   row, so a MinIO client is created once per task rather than 50,000 times: each
   task fetches its rows' audio bytes from MinIO and calls
   `common.features.extract_features` on each.
3. Converts the resulting RDD back into a Spark DataFrame, writes it to Parquet, and
   separately `foreachPartition`s over it to bulk-index the same rows into
   Elasticsearch (`asvspoof-training-features`), again creating one Elasticsearch
   client per partition rather than per row.

**Why MinIO reads happen inside `mapPartitions` instead of via a Hadoop-AWS/S3A
Spark data source connector:** the S3A connector needs specific Hadoop jars whose
versions must match the Spark/Hadoop build exactly, which is a common source of
version-mismatch failures. Reading via `boto3` inside a plain RDD transformation
sidesteps that entirely — the executors just run ordinary Python code — at the cost
of Spark not being aware these are "S3 reads" for its own optimizers, which doesn't
matter here since Spark isn't doing any I/O-aware planning we'd benefit from anyway.

**Four concrete Windows/Python-version bugs turned up while building this, all
worth being able to explain rather than having silently worked around:**

1. *Wrong Python resolved for Spark workers.* The first run crashed with
   `Python worker exited unexpectedly` / `SocketException: Connection reset`.
   PySpark's worker subprocess is resolved from `PATH` (or `PYSPARK_PYTHON` if
   set), and on this machine that resolved to an unrelated Python 3.14 install
   with none of the project's dependencies. Fixed by setting
   `os.environ["PYSPARK_PYTHON"] = sys.executable` (and the driver equivalent) at
   the top of the script, so driver and workers always share this exact venv.
2. *Spaces in the project path.* The project initially lived under
   `...\OneDrive - Hailo Technologies LTD\...`. Even with the right Python pinned,
   PySpark's own executable-resolution logic on Windows breaks on paths containing
   spaces (visible as `Missing Python executable '...', defaulting to ...` in the
   driver log). Fixed by moving the whole project to a space-free path
   (`C:\Users\<user>\dev\asvspoof-voice-ai`) before continuing.
3. *pyspark 3.5.1 vs. Python 3.12.* Even after fixing (1) and (2), a *trivial*
   `rdd.map(...).collect()` still crashed the same way. This turned out to be a
   genuine pyspark/Python-3.12 incompatibility on Windows (a Python 3.12 change in
   socket-handle cleanup that pyspark 3.5.1's worker protocol doesn't handle
   cleanly) — unrelated to anything in this project's code. Fixed by upgrading to
   `pyspark==3.5.9` (confirmed with the same trivial `map` test before trusting it
   with real code).
4. *`toPandas()` needs `distutils`, which Python 3.12 removed from the standard
   library.* Surfaced as `ModuleNotFoundError: No module named 'distutils'` deep in
   pyspark's pandas-conversion helper. Fixed by installing `setuptools`, which
   ships a compatibility shim that makes `import distutils` work again on 3.12+
   (this is also why `toPandas()` is used at all — see the write-path note below).

None of these are specific to this project's logic; they're exactly the kind of
environment/version friction any Spark-on-Windows setup hits, and are recorded here
so they can be explained rather than mistaken for a code bug.

**Why the feature table is written with pandas/pyarrow, not Spark's own
`DataFrameWriter.parquet()`:** Spark's native writer needs `winutils.exe` on
Windows — it shells out to Hadoop's local filesystem implementation to set
POSIX-style permissions on the output directory, which fails with
`HADOOP_HOME and hadoop.home.dir are unset` without that binary in place. Rather
than install a third-party native Windows binary just to satisfy a permissions
call, the (comparatively small — one row per utterance, not per audio sample)
feature table is collected to the driver with `toPandas()` and written with
`pandas.DataFrame.to_parquet()` instead. Spark still does the actual distributed
work — the `mapPartitions` feature extraction across executors — this only
changes how the small final result gets written to disk.

### `pipeline/train_classifier.py` — the AI capability, training half
Loads the Parquet feature table, splits it by the dataset's own `partition` column
(train vs. dev) — **not** a random split. Trains a `RandomForestClassifier`
(`class_weight="balanced"`, since spoof clips heavily outnumber bonafide ones in
ASVspoof2019) on train, evaluates on dev (accuracy/precision/recall/F1 +
per-feature importances), saves the model with `joblib`, and writes both a local
metrics JSON and a summary document into Elasticsearch (`asvspoof-training-results`).

**Why a Random Forest, not a neural net:** feature importances are directly
inspectable (which acoustic properties actually separate human from AI speech —
see `docs/RESULTS.md`), training is fast enough to iterate on a laptop, and a model
this simple is one the whole team can read `sklearn`'s source for and fully explain
in Q&A. A CNN over raw spectrograms would likely score higher but would also be a
much harder thing to defend as "fully understood."

**Why the dataset's own train/dev split, not a random one:** a random split could
put the same speaker's utterances, or utterances from the same attack system, on
both sides of the split. The model could then partly learn to recognize *that
speaker's voice* or *that particular attack's fingerprint* rather than general
bonafide-vs-spoof cues, inflating dev accuracy in a way that wouldn't generalize.
ASVspoof2019's speaker-disjoint partitions exist specifically to prevent this.

**What the full-scale run (25,380 train + 24,844 dev utterances) actually found —
and why accuracy alone would have hidden it:** dev accuracy at the default 0.5
decision threshold is 92.3%, which sounds strong, but the per-class breakdown tells
a different story:

```
                  precision    recall  f1-score   support
bonafide (human)       0.93      0.27      0.42      2548
      spoof (AI)       0.92      1.00      0.96     22296
```

The model catches essentially all spoofed speech (99.8% recall) but correctly
flags only 27% of real human speech as bonafide - it's heavily biased toward
predicting "spoof." This is the **accuracy paradox**: dev is ~90% spoof, so a
model that just leans toward the majority class scores well on accuracy while
failing the minority class. `class_weight="balanced"` reweights the *training*
loss, but it doesn't change the *default 0.5 threshold* `predict()` applies at
evaluation time - the two are different levers, and only fixing one isn't enough.

This is exactly why the training script also reports **ROC-AUC (0.969)** and
**EER (9.09%)** — both threshold-independent, and EER specifically is the metric
the official ASVspoof challenge itself is scored on (see `equal_error_rate()` in
this file). A 9% EER means the underlying probability scores separate the two
classes reasonably well (consistent with the strong ROC-AUC); the poor recall
above is a threshold-placement problem on top of a genuinely OK-scoring model,
not evidence the features carry no signal. A natural next step - noted here
rather than implemented, to keep the submission's scope honest about what was
and wasn't done - would be to move the decision threshold to the EER operating
point instead of 0.5, which would trade some spoof recall for much better
bonafide recall.

### `pipeline/kafka_producer.py` + `pipeline/streaming_enrichment.py` — the AI capability, streaming half
`kafka_producer.py` replays the eval partition's utterance metadata (MinIO key +
attack id + true label — **not** the audio itself) onto the `asvspoof-events` Kafka
topic, one message per utterance, with a configurable delay to simulate audio
"arriving" over time.

`streaming_enrichment.py` is a Kafka consumer that, for every message: fetches the
referenced clip from MinIO, extracts the same features, runs the already-trained
model (`predict_proba`), and writes an enriched document (prediction, confidence,
whether it matched the known true label, a timestamp) into Elasticsearch
(`asvspoof-predictions`) — one document per utterance, as it's scored, not in a
batch after the fact.

**On the full-scale run, the streaming demo processed 1,165 of the 5,000 eval
utterances queued, not all 5,000.** The Kafka producer/consumer pair itself
worked correctly throughout - every message that was consumed was fetched,
scored, and written to Elasticsearch with no errors - but consumer throughput
on this machine was, unpredictably, far below what standalone benchmarking of
the same `extract_features` + `predict_proba` call chain measured (roughly
0.1-0.3s/utterance in isolation vs. an effective multi-second/utterance rate
under sustained background execution). The batch Spark job hit the same kind
of unpredictable slowdown earlier in this run and eventually pushed through it;
the streaming step was capped at 1,165 scored utterances - still enough to
cover every attack type in eval (see `docs/RESULTS.md`) - rather than spend
several more hours on what is a secondary demonstration of an AI capability
already fully validated by the batch training run above. The root cause was
never conclusively identified; the leading candidate is Windows throttling
long-running background console processes, though this was never proven.

The consumer joins a named consumer group (`asvspoof-streaming-enrichment`) with
auto-commit enabled, so Kafka tracks how far it's read. Without a group id, every
run would start from the earliest offset and re-score every message ever
published to the topic (this is exactly what happened during testing — a first
run scored 20 utterances, and re-running without a group id scored the same 20
again instead of picking up only new ones). With the group id in place, a second
run against an unchanged topic correctly scores zero new utterances.

**Why a plain Python Kafka consumer, not Spark Structured Streaming:** Spark's job
in this project is the batch feature-extraction step, where parallelizing tens of
thousands of files across cores is genuinely valuable. Per-message scoring here is
comparatively light; routing it through Spark Structured Streaming would mean
pulling in the `spark-sql-kafka-0-10` connector (resolved via Maven at runtime) for
a workload that doesn't need Spark's distributed execution. A plain consumer is
simpler to run, debug, and explain, while Kafka itself is still doing real work as
the streaming/event backbone.

### `pipeline/insights.py`
Runs Elasticsearch aggregation queries against `asvspoof-predictions` — overall
accuracy, accuracy by attack ID, a confusion matrix, class balance — and writes both
`docs/RESULTS.md` and the PNG charts the dashboard displays. Every number here comes
from a live query, not a hand-typed table, so re-running the pipeline (sample or
full-scale) and re-running this script always reflects the actual data.

## 5. `app/` — the demo web app

`app/server.py` is a small Flask app with three routes:
- `/` — links/overview.
- `/classify` — accepts an uploaded audio file, runs it through
  `common.features.extract_features` and the same saved model the pipeline trains,
  returns "Human" or "AI-generated" + confidence. This is the same code path as
  `streaming_enrichment.py`, just triggered by an HTTP upload instead of a Kafka
  message.
- `/dashboard` — runs the same style of Elasticsearch aggregations as
  `pipeline/insights.py`, live, so the page always reflects current pipeline state.

No JavaScript build step, no Node — server-rendered Jinja2 templates
(`app/templates/`) and a small hand-written stylesheet (`app/static/style.css`).

## 6. Infrastructure — `docker-compose.yml`

Three services, all with host ports exposed so pipeline scripts (running directly
on the host, not in containers) can reach them at `localhost`:
- **MinIO** (`9000` S3 API, `9001` console) — object store.
- **Kafka** (`9092`), running in **KRaft mode** (no separate ZooKeeper container) —
  the modern way to run Kafka; ZooKeeper appears in the course's general topics list
  but not in the specific "pick at least one" technology list the brief asks for
  in §5.1, so its absence here isn't a gap against the requirements.
- **Elasticsearch** (`9200`), single-node, security disabled — appropriate for a
  local course project, not a production deployment.

Spark itself runs directly on the host via `pip install pyspark` (local mode,
`SparkSession.builder.getOrCreate()`), rather than in its own container — this
avoids Spark-in-Docker networking complexity (getting a containerized Spark driver
to see host-exposed Kafka/MinIO/Elasticsearch ports reliably on Windows) while still
being a legitimate, standard way to use Spark for a project this size.

## 7. Mapping to the grading rubric

| Rubric line | Where it's satisfied |
|---|---|
| Data and pipeline (25%) | `pipeline/ingest_to_minio.py` → `batch_feature_extraction.py` → Parquet/Elasticsearch is a real ETL pipeline over unstructured audio data |
| Use of course technologies (20%) | Object store (MinIO), streaming (Kafka), NoSQL (Elasticsearch), Spark, Docker — five course technologies, each with a clear, non-decorative role |
| AI capability (25%) | A trained classifier (§6.2f) operating on the data, applied in both batch validation and streaming enrichment (§6.2e-style) — not a bolted-on separate feature |
| Results and insights (15%) | `docs/RESULTS.md`, generated by live Elasticsearch aggregations in `pipeline/insights.py` |
| Presentation and demo (10%) | `docs/slides/generate_slides.py` builds the deck; `/classify` and `/dashboard` are the live demo |
| Understanding and Q&A (5%) | This document — every component's purpose and the trade-offs behind it, in plain language |

## 8. How to reproduce

See `README.md` for exact commands. Short version: `docker compose up -d`, then run
`pipeline/ingest_to_minio.py` → `batch_feature_extraction.py` → `train_classifier.py`
→ (`kafka_producer.py` + `streaming_enrichment.py` together) → `insights.py`, then
`python -m app.server`. Every step accepts `--la-root data/sample/LA` (the committed
synthetic sample, no download needed) or `--la-root data/raw/LA` (the real dataset,
after `pipeline/download_dataset.py`).

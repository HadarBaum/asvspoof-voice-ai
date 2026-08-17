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

### `common/model.py`
Loads the artifact `pipeline/train_classifier.py` saves - a dict bundling the
fitted classifier together with its own decision threshold - and exposes one
`ScoredModel.classify(vector) -> (label, confidence)` method used identically
by `pipeline/streaming_enrichment.py` and `app/server.py`. Before this existed,
both of those files independently hardcoded `>= 0.5` as the spoof/bonafide
cutoff; centralizing it here is what makes the threshold fix in §4 actually
take effect everywhere at once instead of needing to be repeated (and
potentially forgotten in one place) at every call site.

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
(train vs. dev) — **not** a random split. Trains **two** candidate classifiers on
the same features - `RandomForestClassifier` and `GradientBoostingClassifier`,
both reweighted for the train set's spoof/bonafide imbalance (`class_weight`
for the former, `sample_weight` computed via `compute_sample_weight("balanced",
...)` for the latter, which has no `class_weight` parameter of its own) -
evaluates both on dev, and **keeps whichever has the lower EER** (see below for
why EER, not accuracy, is the selection criterion). The winner's model, its
name, and its EER-optimal decision threshold are bundled into one artifact via
`joblib.dump({"model": ..., "model_name": ..., "threshold": ...}, ...)` -
everything a downstream consumer needs to score a clip correctly, in one file
(see `common/model.py`).

**Why compare two models instead of committing to one:** feature importances
stay directly inspectable either way (both are tree ensembles), training is
fast enough to iterate on a laptop, and both are simple enough for the whole
team to read `sklearn`'s source for and fully explain in Q&A - a CNN over raw
spectrograms would likely score higher but be a much harder thing to defend as
"fully understood." Between the two, a real comparison (see
`docs/model_comparison.json` for the full numbers) is more defensible than an
unchallenged single number: on the full-scale run, Gradient Boosting won with
EER 9.26% vs. Random Forest's 12.31% (ROC-AUC 0.970 vs. 0.951).

**Why the dataset's own train/dev split, not a random one:** a random split could
put the same speaker's utterances, or utterances from the same attack system, on
both sides of the split. The model could then partly learn to recognize *that
speaker's voice* or *that particular attack's fingerprint* rather than general
bonafide-vs-spoof cues, inflating dev accuracy in a way that wouldn't generalize.
ASVspoof2019's speaker-disjoint partitions exist specifically to prevent this.

**The accuracy paradox this project actually hit, and how it was fixed (not just
diagnosed):** the first version of this pipeline evaluated only at scikit-learn's
default 0.5 decision threshold. Dev accuracy was 92.3%, which sounds strong, but
the per-class breakdown told a different story:

```
                  precision    recall  f1-score   support
bonafide (human)       0.93      0.27      0.42      2548
      spoof (AI)       0.92      1.00      0.96     22296
```

The model caught essentially all spoofed speech (99.8% recall) but correctly
flagged only 27% of real human speech as bonafide - heavily biased toward
predicting "spoof." Dev is ~90% spoof, so a model that just leans toward the
majority class scores well on accuracy while failing the minority class.
`class_weight`/`sample_weight` reweight the *training* loss, but that doesn't
change the *default 0.5 threshold* `.predict()` applies at evaluation time -
the two are different levers, and fixing only one isn't enough.

This is exactly why the training script also reports **ROC-AUC** and **EER**
(`compute_eer()` in this file) - both threshold-independent, and EER specifically
is the metric the official ASVspoof challenge itself is scored on. A ~9% EER
means the underlying probability scores separate the two classes reasonably
well; the poor recall above was a threshold-*placement* problem on top of an
already reasonably-scoring model, not evidence the features carried no signal.
`compute_eer()` returns not just the error rate but the **score threshold** at
that operating point, and the model artifact now stores it and uses it instead
of 0.5. The result, re-measured on the same dev set with the winning Gradient
Boosting model at its own EER threshold (0.689, not 0.5):

```
                  precision    recall  f1-score   support
bonafide (human)       0.53      0.91      0.67      2548
      spoof (AI)       0.99      0.91      0.95     22296
```

Bonafide recall: 27% → 91%. Precision on bonafide dropped (0.93 → 0.53) - the
model now calls more things bonafide, including some spoof clips it used to
catch - which is the real trade-off EER makes explicit rather than hiding: you
cannot maximize both classes' accuracy simultaneously on an imbalanced problem,
you can only choose *which* trade-off point on the ROC curve to operate at. EER
picks the point where both error types are equal, which is a principled default
rather than the arbitrary default of 0.5. This was re-confirmed independently on
the live Kafka streaming run against real held-out eval data, not just recomputed
on the same dev set the threshold was chosen from - see the streaming section
below and `docs/RESULTS.md`.

### `pipeline/index_embeddings.py` — the AI capability, semantic search half
Builds the "embeddings and semantic search" option from the course brief
(§6.2b), reusing the *same* acoustic feature vectors already computed for the
classifier as the embedding, rather than introducing a separate embedding
model. Raw features span wildly different scales (`duration_seconds` ~1-10 vs.
`spectral_rolloff` in the thousands), so cosine similarity over them unscaled
would just be dominated by whichever feature has the largest absolute
magnitude - the script fits a `StandardScaler` on train (zero mean, unit
variance per feature), stores it (`models/embedding_scaler.joblib`) so
`app/server.py` can standardize a freshly-uploaded clip's features the exact
same way, and writes the standardized vectors into a `dense_vector` field
(`embedding`, cosine similarity) on the already-existing
`asvspoof-training-features` index via a partial `_op_type: update` bulk
request - not a full reindex, since the raw feature columns
`batch_feature_extraction.py` already wrote for each document don't need to
change.

**Why add the field to an already-existing index instead of recreating it:**
this index already held 50,224 real, expensively-computed documents from the
batch job by the time embeddings were added. Elasticsearch allows adding a new
field to an existing mapping in place (`common.es_client.ensure_embedding_mapping()`,
a `put_mapping` call) as long as the field doesn't already exist under a
conflicting type - no reindex, no reprocessing, no re-running Spark.

`app/server.py`'s `/classify` route calls this same standardization + a
`knn` search (`k=5`, `num_candidates=100`) against this index after every
classification, surfacing the 5 most acoustically similar training clips -
this is nearest-neighbor semantic search replacing a keyword match, directly
built on the Elasticsearch topic from the course.

### `pipeline/train_classifier.py` and `pipeline/index_embeddings.py` — order of operations
`index_embeddings.py` reads the *same* Parquet feature table
`batch_feature_extraction.py` produced and `train_classifier.py` trains on, so
it can run any time after the batch job (independently of, before, or after
training) - it doesn't depend on the trained model at all, only on the raw
features.

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

**On both full-scale runs, the streaming demo processed a capped subset of the
5,000 eval utterances queued, not all 5,000** (1,165 with the original Random
Forest + 0.5-threshold model; 498 after switching to the Gradient Boosting +
EER-threshold model, which needed a fresh consumer group - see below - to
re-score from the start of the topic). The Kafka producer/consumer pair itself
worked correctly throughout both runs - every message that was consumed was
fetched, scored, and written to Elasticsearch with no errors - but consumer
throughput on this machine was, unpredictably, far below what standalone
benchmarking of the same `extract_features` + `predict_proba` call chain
measured (roughly 0.1-0.3s/utterance in isolation vs. an effective
multi-second/utterance rate under sustained background execution). The batch
Spark job hit the same kind of unpredictable slowdown and eventually pushed
through it; the streaming step was capped both times rather than spend several
more hours on what is a secondary demonstration of an AI capability already
fully validated by the batch training run above. The root cause was never
conclusively identified; the leading candidate is Windows throttling
long-running background console processes, though this was never proven.

**The second (498-utterance) run independently confirms the threshold fix
worked - not just on the dev set the threshold was chosen from, but on live,
never-trained-on eval data streamed through Kafka:** bonafide accuracy on the
"-" (bonafide) row in `docs/RESULTS.md` went from 29.7% (Random Forest, 0.5
threshold) to 91.5% (Gradient Boosting, EER threshold) - matching the dev-set
improvement almost exactly. The trade-off shows up too: a handful of the
harder/later attack IDs (A17, A18, A19) score noticeably worse than they did
under the old model (e.g. A17 dropped from 56.6% to 13.2%) - the new threshold
buys bonafide recall by giving up some margin on the spoof attacks whose scores
sit closest to the bonafide range. This is a real, worthwhile trade to make
(catching genuine human speech reliably matters more than catching every
attack variant), but it is a trade, not a strict improvement, and is reported
here rather than only showing the metric that improved.

The consumer joins a named consumer group (`--group-id`, default
`asvspoof-streaming-enrichment`) with auto-commit enabled, so Kafka tracks how
far it's read. Without a group id, every run would start from the earliest
offset and re-score every message ever published to the topic (this is exactly
what happened during early testing — a first run scored 20 utterances, and
re-running without a group id scored the same 20 again instead of picking up
only new ones). With a group id in place, a second run against an *unchanged*
model resumes correctly and scores zero new utterances - but that's also
exactly why switching models required passing a **new** `--group-id`
(`asvspoof-streaming-enrichment-gb`): reusing the old group would have resumed
from its committed offset and only scored the ~3,800 *remaining* messages with
the new model, mixing old-model and new-model predictions in the same
comparison. A fresh group name re-reads the topic from the beginning instead.

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
`docs/RESULTS.md` and the PNG charts the dashboard displays:
- `accuracy_by_attack.png` / `class_balance.png` — straight from the terms
  aggregations above.
- `confusion_matrix.png` — the same 2×2 true-label × predicted-label counts
  already computed for the markdown table, now also rendered as a heatmap
  (`plt.imshow` + text annotations) so it reads at a glance instead of
  requiring someone to parse four numbers in a table.
- `confidence_histogram.png` — the one chart here that *isn't* an aggregation:
  Elasticsearch's aggregation API gives you terms/stats/percentiles, but not a
  literal distribution, so this pulls the raw `(confidence, correct)` pairs for
  every scored utterance (`es.search(..., source=["confidence", "correct"])`,
  cheap since there are only hundreds to low-thousands of predictions) and
  bins them locally into two overlaid histograms - correct vs. incorrect
  predictions. A well-calibrated model's incorrect predictions should cluster
  at *lower* confidence than its correct ones; on this run they mostly do, but
  a real (if smaller) share of incorrect predictions land at high confidence
  too - the model is sometimes confidently wrong, which is worth being able to
  say out loud if asked about calibration.

Every number here comes from a live query, not a hand-typed table, so
re-running the pipeline (sample or full-scale) and re-running this script
always reflects the actual data.

**Where the ROC curve and feature-importance charts come from instead:** they're
generated in `pipeline/train_classifier.py`, not here, because they need
`y_dev`/`y_proba` and the fitted model's `.feature_importances_` - data that
only exists in that script's scope, not in anything Elasticsearch stores. The
ROC curve plots *both* candidate models (so the comparison table's numbers
have a visual companion) with the deployed model's exact EER operating point
marked as a dot; the feature-importance chart is the same top-10 list already
printed to the console and saved in `docs/training_metrics.json`, now a
horizontal bar chart. One implementation detail worth knowing: the ROC curve's
`fpr`/`tpr` arrays are numpy arrays, kept around in memory for plotting, but
stripped out before `docs/model_comparison.json` is written - numpy arrays
aren't JSON-serializable, and a machine-readable comparison file doesn't need
the full curve anyway, just the summary numbers.

## 5. `app/` — the demo web app

`app/server.py` is a small Flask app with four routes:
- `/` — links/overview.
- `/classify` — accepts an uploaded audio file **or a live microphone
  recording** (see below), runs it through `common.features.extract_features`
  and the same saved model the pipeline trains (via `common.model.load_model`),
  returns "Human" or "AI-generated" + confidence, and additionally runs
  `find_similar_clips()` - the same standardization + Elasticsearch `knn`
  search `pipeline/index_embeddings.py` set up - to show the 5 most
  acoustically similar training clips, **each one playable in the browser**
  (see `/audio` below). Classification is the same code path as
  `streaming_enrichment.py`, just triggered by an HTTP upload instead of a
  Kafka message; the similarity search never blocks the main result if it
  fails (wrapped in its own try/except), since it's a bonus, not the core
  answer.
- `/audio/<path:key>` — streams a clip straight out of MinIO by its object key
  (`minio_client.get_object_bytes()`, the exact same call every other
  MinIO-reading component uses) and returns it as `audio/flac`. This exists
  so the similar-clips list is actually useful in a live demo: a similarity
  *score* is an abstraction, but hearing that the top match really does sound
  alike is what makes the audience believe the embedding is doing something
  real. `<path:key>` (not the default `<string:key>` converter) matters here -
  MinIO keys contain `/` (e.g. `LA/ASVspoof2019_LA_train/flac/LA_T_....flac`),
  and the default converter stops matching at the first slash.
- `/dashboard` — runs the same style of Elasticsearch aggregations as
  `pipeline/insights.py`, live, so the page always reflects current pipeline
  state, plus:
  - an inline **model comparison table** (Random Forest vs. Gradient Boosting,
    read from `docs/model_comparison.json` server-side - the file path itself
    is never shown to the audience, only the real numbers it contains,
    rendered as an actual table),
  - a **threshold comparison table** (deployed EER threshold vs. what the
    default 0.5 threshold would have given) with the accuracy-paradox
    explanation written out in the page itself rather than pointing at this
    document,
  - the **ROC curve, feature importance, confusion matrix, and confidence
    histogram** charts (see `pipeline/train_classifier.py` and
    `pipeline/insights.py` above for how each is generated).

  Two of the threshold-comparison table's cells used to render as an em-dash
  placeholder (`—`) instead of a number: `train_classifier.py` computed
  spoof-recall and F1 at the default threshold internally but only ever
  persisted 2 of those 4 "at default threshold" metrics into
  `training_metrics.json`. Not a fundamental limitation, just an incomplete
  first pass - fixed by saving the other two fields once the gap was noticed.

No JavaScript build step, no Node, no external CDN dependency — server-rendered
Jinja2 templates (`app/templates/`) and a small hand-written stylesheet
(`app/static/style.css`). The UI deliberately never references its own source
files (no "see docs/EXPLANATION.md" text on-screen) - anything worth knowing
while looking at a page is written directly into that page instead, since an
audience watching a demo can't open the repo mid-presentation.

**Live microphone recording (`classify.html`'s inline `<script>`):** clicking
"Start recording" calls `navigator.mediaDevices.getUserMedia({audio: true})`,
then pipes the stream through a `ScriptProcessorNode` to capture raw Float32
PCM samples directly (rather than `MediaRecorder`, whose default output on
most browsers is Opus-encoded WebM/Ogg - a codec `librosa`/`soundfile` can't
reliably decode without a system `ffmpeg` install). On "Stop," the captured
samples are hand-encoded into a 16-bit PCM WAV file client-side (a ~40-line
vanilla-JS WAV header writer - no library), wrapped in a `File`, attached to
the existing file `<input>` via the `DataTransfer` API, and the *same*
multipart form is submitted normally. This means the server-side `/classify`
handler needed zero changes to support live recording - as far as Flask is
concerned, a recorded clip and an uploaded file are indistinguishable, both
arriving as a normal `multipart/form-data` upload. `ScriptProcessorNode` is
deprecated in favor of `AudioWorklet`, but the latter requires loading a
separate worklet module file; `ScriptProcessorNode` needed no extra file and
works in every current browser, which mattered more here than avoiding a
deprecation warning.

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
| AI capability (25%) | Two AI capabilities from the brief, both operating on the data itself: (f) a compared/selected classifier (Random Forest vs. Gradient Boosting, picked by EER) applied in both batch validation and streaming enrichment (§6.2e-style); and (b) embeddings + Elasticsearch k-NN semantic search over the same acoustic features — not bolted-on separate features, all sharing one feature-extraction implementation |
| Results and insights (15%) | `docs/RESULTS.md`, generated by live Elasticsearch aggregations in `pipeline/insights.py`, backed on the dashboard by six generated charts (accuracy by attack, class balance, ROC curve, feature importance, confusion matrix, confidence calibration) |
| Presentation and demo (10%) | `docs/slides/generate_slides.py` builds a 13-slide deck (model comparison table, stat cards, all six charts) styled to match the live app's color palette; `/classify` (upload or live mic recording, playable similar-clip search) and `/dashboard` are the live demo |
| Understanding and Q&A (5%) | This document — every component's purpose and the trade-offs behind it, in plain language |

## 8. How to reproduce

See `README.md` for exact commands. Short version: `docker compose up -d`, then run
`pipeline/ingest_to_minio.py` → `batch_feature_extraction.py` →
(`train_classifier.py` and `index_embeddings.py`, either order - both only need
the Parquet feature table) → (`kafka_producer.py` + `streaming_enrichment.py`
together) → `insights.py` → `docs/slides/generate_slides.py` (regenerates the
deck from whatever's currently in `training_metrics.json`, `model_comparison.json`,
and `app/static/charts/` - always run it last), then `python -m app.server`.
Every step accepts `--la-root data/sample/LA` (the committed synthetic sample,
no download needed) or `--la-root data/raw/LA` (the real dataset, after
`pipeline/download_dataset.py`).

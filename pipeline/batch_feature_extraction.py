"""
Step 2 of the pipeline: the Spark batch job.

Reads the train+dev protocol metadata (small text files, kept local - only
the labels/keys, not the audio), fans out across Spark executors to fetch
each clip's audio bytes from MinIO and compute acoustic features
(common.features.extract_features - the same function used everywhere else),
then writes:
  - a Parquet feature table (input to pipeline/train_classifier.py)
  - the same rows into Elasticsearch's asvspoof-training-features index
    (lets us run aggregations/insights without re-reading Parquet)

This is the actual "big data" step: with the full dataset train+dev is
~50k audio files, decoded and feature-extracted in parallel across cores/
executors instead of one Python process working through them serially.

Usage:
    python -m pipeline.batch_feature_extraction --la-root data/sample/LA --out pipeline/features_train_dev.parquet
"""

import argparse
import os
import sys

# PySpark workers spawn their own Python subprocess and, unless told otherwise,
# resolve it from PATH - which may not be this venv (and crashes the worker
# with a socket/protocol error if the resolved Python lacks our dependencies).
# Pinning both to sys.executable guarantees driver and workers use the same
# interpreter/venv this script is running in.
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

from pyspark.sql import Row, SparkSession

from common import dataset_layout, es_client, minio_client


def extract_partition(rows):
    """Runs once per Spark partition (i.e. once per worker task, not once per
    row) so we only pay the cost of creating a MinIO client that many times,
    not once per utterance."""
    client = minio_client.get_client()
    for row in rows:
        try:
            audio_bytes = minio_client.get_object_bytes(client, row["key"])
            from common.features import extract_features  # imported here: must exist on executors too

            feats = extract_features(audio_bytes)
        except Exception as exc:  # noqa: BLE001 - one bad file must not kill the whole job
            print(f"WARN: failed to process {row['key']}: {exc}")
            continue
        yield Row(**{**row.asDict(), **feats})


def index_partition_to_es(rows):
    client = es_client.get_client()
    docs = [row.asDict() for row in rows]
    if docs:
        es_client.bulk_index(client, es_client.TRAINING_FEATURES_INDEX, docs, id_field="key")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--la-root", required=True)
    parser.add_argument("--partitions", nargs="+", default=["train", "dev"])
    parser.add_argument("--out", required=True, help="Output Parquet path")
    args = parser.parse_args()

    utterance_rows = []
    for partition in args.partitions:
        for u in dataset_layout.load_partition(args.la_root, partition):
            utterance_rows.append(
                {
                    "key": dataset_layout.minio_key(u),
                    "partition": u.partition,
                    "speaker_id": u.speaker_id,
                    "attack_id": u.attack_id,
                    "label": u.label,
                }
            )
    print(f"Loaded {len(utterance_rows)} labeled utterances from protocol files.")

    spark = SparkSession.builder.appName("asvspoof-batch-feature-extraction").getOrCreate()

    # Repartition so extraction actually spreads across cores instead of running
    # as one giant partition; ~200 files per task is a reasonable chunk size.
    n_partitions = max(1, len(utterance_rows) // 200)
    base_rdd = spark.sparkContext.parallelize([Row(**r) for r in utterance_rows], numSlices=n_partitions)

    feature_rdd = base_rdd.mapPartitions(extract_partition)
    feature_df = spark.createDataFrame(feature_rdd)
    feature_df.cache()

    n_ok = feature_df.count()
    print(f"Extracted features for {n_ok}/{len(utterance_rows)} utterances.")

    # Spark's native Parquet writer needs winutils.exe on Windows (it shells out to
    # set POSIX-style permissions on the output directory via Hadoop's local
    # filesystem implementation). Rather than install that third-party native
    # binary, we collect the (small - one row per utterance, not per audio sample)
    # feature table to the driver and write it with pandas/pyarrow instead. Spark
    # still did the actual distributed work above (mapPartitions across executors);
    # only this final, comparatively tiny write bypasses Spark's own writer.
    feature_df.toPandas().to_parquet(args.out, index=False)
    print(f"Wrote Parquet feature table to {args.out}")

    es_client.ensure_indices(es_client.get_client())
    feature_df.rdd.foreachPartition(index_partition_to_es)
    print(f"Indexed {n_ok} feature rows into Elasticsearch ({es_client.TRAINING_FEATURES_INDEX}).")

    spark.stop()


if __name__ == "__main__":
    main()

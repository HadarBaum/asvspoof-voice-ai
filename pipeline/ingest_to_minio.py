"""
Step 1 of the pipeline: upload the raw flac corpus into MinIO, our object
store / data lake. Downstream steps (Spark batch job, streaming consumer)
read audio from here, never from the local filesystem - that's the point of
landing it in an object store: any worker, anywhere, can fetch a clip by key.

Usage:
    python -m pipeline.ingest_to_minio --la-root data/sample/LA
    python -m pipeline.ingest_to_minio --la-root data/raw/LA   (once the full dataset is downloaded)
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

from common import dataset_layout, minio_client


def _upload_one(client, utterance):
    key = dataset_layout.minio_key(utterance)
    minio_client.upload_file(client, utterance.flac_path, key)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--la-root", required=True, help="Path to the LA/ folder")
    parser.add_argument(
        "--partitions", nargs="+", default=["train", "dev", "eval"], help="Which partitions to ingest"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=32,
        help="Concurrent upload threads - this step is I/O-bound (many small files), so "
        "threading helps a lot even though Python threads don't parallelize CPU work",
    )
    parser.add_argument(
        "--eval-limit",
        type=int,
        default=None,
        help="Cap how many eval utterances to ingest. The eval partition (~72k files) is only "
        "used by the Kafka streaming demo (pipeline/kafka_producer.py's own --limit), so "
        "ingesting more of it into MinIO than the demo will ever stream is wasted work. "
        "train/dev are always ingested in full - they feed the batch feature extraction "
        "and training, where using the whole partition is the point.",
    )
    args = parser.parse_args()

    client = minio_client.get_client()
    minio_client.ensure_bucket(client)

    total = 0
    for partition in args.partitions:
        utterances = dataset_layout.load_partition(args.la_root, partition)
        if partition == "eval" and args.eval_limit:
            utterances = utterances[: args.eval_limit]
        print(f"{partition}: {len(utterances)} utterances to upload")
        done = 0
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(_upload_one, client, u) for u in utterances]
            for future in as_completed(futures):
                future.result()  # re-raise if any upload failed
                done += 1
                if done % 1000 == 0:
                    print(f"  {partition}: {done}/{len(utterances)} uploaded")
        total += done

    print(f"Done. Uploaded {total} objects to bucket '{minio_client.RAW_BUCKET}'.")


if __name__ == "__main__":
    main()

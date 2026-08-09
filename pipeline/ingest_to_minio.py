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

from common import dataset_layout, minio_client


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--la-root", required=True, help="Path to the LA/ folder")
    parser.add_argument(
        "--partitions", nargs="+", default=["train", "dev", "eval"], help="Which partitions to ingest"
    )
    args = parser.parse_args()

    client = minio_client.get_client()
    minio_client.ensure_bucket(client)

    total = 0
    for partition in args.partitions:
        utterances = dataset_layout.load_partition(args.la_root, partition)
        print(f"{partition}: {len(utterances)} utterances to upload")
        for i, u in enumerate(utterances):
            key = dataset_layout.minio_key(u)
            minio_client.upload_file(client, u.flac_path, key)
            total += 1
            if (i + 1) % 500 == 0:
                print(f"  {partition}: {i + 1}/{len(utterances)} uploaded")

    print(f"Done. Uploaded {total} objects to bucket '{minio_client.RAW_BUCKET}'.")


if __name__ == "__main__":
    main()

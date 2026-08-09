"""
Thin wrapper around boto3's S3 client pointed at MinIO. MinIO speaks the S3
API, so plain boto3 is enough - no special MinIO SDK needed.
"""

import os

import boto3
from botocore.client import Config

MINIO_ENDPOINT = os.environ.get("MINIO_ENDPOINT", "http://localhost:9000")
MINIO_ACCESS_KEY = os.environ.get("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.environ.get("MINIO_SECRET_KEY", "minioadmin")
RAW_BUCKET = "asvspoof-raw"


def get_client():
    return boto3.client(
        "s3",
        endpoint_url=MINIO_ENDPOINT,
        aws_access_key_id=MINIO_ACCESS_KEY,
        aws_secret_access_key=MINIO_SECRET_KEY,
        config=Config(signature_version="s3v4"),
        region_name="us-east-1",
    )


def ensure_bucket(client, bucket: str = RAW_BUCKET):
    existing = [b["Name"] for b in client.list_buckets().get("Buckets", [])]
    if bucket not in existing:
        client.create_bucket(Bucket=bucket)


def upload_file(client, local_path: str, key: str, bucket: str = RAW_BUCKET):
    client.upload_file(local_path, bucket, key)


def get_object_bytes(client, key: str, bucket: str = RAW_BUCKET) -> bytes:
    return client.get_object(Bucket=bucket, Key=key)["Body"].read()


def list_keys(client, prefix: str, bucket: str = RAW_BUCKET):
    """Yield every object key under `prefix`, transparently following pagination."""
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            yield obj["Key"]

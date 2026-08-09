"""
Step 4a of the pipeline: simulate new audio "arriving" by publishing one
small JSON event per eval-partition utterance onto a Kafka topic. Only a
reference (the MinIO object key + attack id) goes on the wire, not the
audio itself - Kafka is an event bus for metadata, MinIO stays the place
where the actual bytes live. This mirrors a real deployment: a service
that receives an audio upload would drop the blob into object storage and
publish an event, rather than pushing megabytes through Kafka.

Run this alongside pipeline/streaming_enrichment.py to see the two work
together: producer publishes -> consumer scores each utterance with the
already-trained model and writes an enriched prediction to Elasticsearch.

Usage:
    python -m pipeline.kafka_producer --la-root data/sample/LA --limit 200 --delay-seconds 0.2
"""

import argparse
import json
import time

from kafka import KafkaProducer

from common import dataset_layout

TOPIC = "asvspoof-events"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--la-root", required=True)
    parser.add_argument("--limit", type=int, default=None, help="Cap how many eval utterances to stream (demo-friendly)")
    parser.add_argument("--delay-seconds", type=float, default=0.1, help="Delay between messages, simulating arrival rate")
    parser.add_argument("--bootstrap-servers", default="localhost:9092")
    args = parser.parse_args()

    utterances = dataset_layout.load_partition(args.la_root, "eval")
    if args.limit:
        utterances = utterances[: args.limit]

    producer = KafkaProducer(
        bootstrap_servers=args.bootstrap_servers,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
    )

    print(f"Streaming {len(utterances)} events to topic '{TOPIC}' ...")
    for i, u in enumerate(utterances):
        event = {
            "key": dataset_layout.minio_key(u),
            "attack_id": u.attack_id,
            "true_label": u.label,
        }
        producer.send(TOPIC, value=event)
        if (i + 1) % 50 == 0:
            print(f"  sent {i + 1}/{len(utterances)}")
        time.sleep(args.delay_seconds)

    producer.flush()
    print("Done streaming.")


if __name__ == "__main__":
    main()

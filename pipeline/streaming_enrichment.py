"""
Step 4b of the pipeline: the near-real-time AI enrichment consumer.

Subscribes to the Kafka topic pipeline/kafka_producer.py publishes to.
For every event: fetch the referenced clip from MinIO, extract the same
features used in training, score it with the already-trained model, and
write an enriched document (prediction + confidence + correctness against
the known eval label) into Elasticsearch as it happens - not in a nightly
batch. This is the AI capability (the trained classifier) applied inline
in a streaming pipeline, the same code path as the batch job and the web
app, just triggered per-message instead of per-partition.

Design note: this is a plain Kafka consumer, not a Spark Structured
Streaming job. Spark's real "big data" job in this project is the batch
feature-extraction step (parallelizing across tens of thousands of files);
routing this comparatively light per-message inference through a second
Spark job would mean pulling in the spark-sql-kafka connector (a Maven
dependency resolved at run time) for no benefit here. See docs/EXPLANATION.md
for the full trade-off.

Usage:
    python -m pipeline.streaming_enrichment --model models/voice_classifier.joblib
    (run pipeline/kafka_producer.py in another terminal to feed it events)
"""

import argparse
import json
from datetime import datetime, timezone

import joblib
from kafka import KafkaConsumer

from common import es_client, minio_client
from common.features import extract_features, features_to_vector

TOPIC = "asvspoof-events"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--bootstrap-servers", default="localhost:9092")
    parser.add_argument(
        "--consumer-timeout-ms",
        type=int,
        default=15000,
        help="Stop after this many ms with no new message (0 = run forever)",
    )
    args = parser.parse_args()

    clf = joblib.load(args.model)
    minio = minio_client.get_client()
    es = es_client.get_client()
    es_client.ensure_indices(es)

    consumer = KafkaConsumer(
        TOPIC,
        bootstrap_servers=args.bootstrap_servers,
        value_deserializer=lambda v: json.loads(v.decode("utf-8")),
        # A named consumer group lets Kafka track our offset: on first run we start
        # from the beginning of the topic, but a restart resumes from where we left
        # off instead of re-scoring every utterance that was ever published.
        group_id="asvspoof-streaming-enrichment",
        auto_offset_reset="earliest",
        enable_auto_commit=True,
        consumer_timeout_ms=args.consumer_timeout_ms,
    )

    print(f"Listening on topic '{TOPIC}' (idle timeout {args.consumer_timeout_ms}ms) ...")
    n_scored = 0
    for message in consumer:
        event = message.value
        try:
            audio_bytes = minio_client.get_object_bytes(minio, event["key"])
            feats = extract_features(audio_bytes)
            vector = features_to_vector(feats)

            proba = clf.predict_proba(vector)[0]  # [P(bonafide), P(spoof)]
            predicted_label = "spoof" if proba[1] >= 0.5 else "bonafide"
            confidence = float(max(proba))

            doc = {
                "key": event["key"],
                "attack_id": event.get("attack_id", "-"),
                "true_label": event.get("true_label"),
                "predicted_label": predicted_label,
                "confidence": confidence,
                "correct": predicted_label == event.get("true_label"),
                "scored_at": datetime.now(timezone.utc).isoformat(),
            }
            es.index(index=es_client.PREDICTIONS_INDEX, id=event["key"], document=doc)
            n_scored += 1
            if n_scored % 25 == 0:
                print(f"  scored {n_scored} utterances so far")
        except Exception as exc:  # noqa: BLE001 - one bad message must not kill the consumer loop
            print(f"WARN: failed to score {event.get('key')}: {exc}")

    print(f"Consumer idle-timed-out. Scored {n_scored} utterances total.")


if __name__ == "__main__":
    main()

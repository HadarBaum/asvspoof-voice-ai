"""
Thin wrapper around the Elasticsearch python client: connection + the three
index mappings the project uses.

  asvspoof-training-features  one doc per train/dev utterance: label + features
                               (written by pipeline/batch_feature_extraction.py)
  asvspoof-training-results   one doc per training run: metrics summary
                               (written by pipeline/train_classifier.py)
  asvspoof-predictions         one doc per utterance scored by the streaming
                               consumer: prediction + confidence + true label
                               (written by pipeline/streaming_enrichment.py)
"""

import os

from elasticsearch import Elasticsearch, helpers

ES_URL = os.environ.get("ES_URL", "http://localhost:9200")

TRAINING_FEATURES_INDEX = "asvspoof-training-features"
TRAINING_RESULTS_INDEX = "asvspoof-training-results"
PREDICTIONS_INDEX = "asvspoof-predictions"

_FEATURE_FIELDS = {
    # dynamic mapping handles the ~53 numeric feature columns fine; we only
    # need to pin down the fields we actually query/aggregate on.
    "key": {"type": "keyword"},
    "partition": {"type": "keyword"},
    "speaker_id": {"type": "keyword"},
    "attack_id": {"type": "keyword"},
    "label": {"type": "keyword"},
}

_PREDICTIONS_FIELDS = {
    "key": {"type": "keyword"},
    "attack_id": {"type": "keyword"},
    "true_label": {"type": "keyword"},
    "predicted_label": {"type": "keyword"},
    "confidence": {"type": "float"},
    "correct": {"type": "boolean"},
    "scored_at": {"type": "date"},
}

_RESULTS_FIELDS = {
    "run_id": {"type": "keyword"},
    "trained_at": {"type": "date"},
    "n_train": {"type": "integer"},
    "n_dev": {"type": "integer"},
    "accuracy_dev": {"type": "float"},
    "precision_dev": {"type": "float"},
    "recall_dev": {"type": "float"},
    "f1_dev": {"type": "float"},
}


def get_client() -> Elasticsearch:
    return Elasticsearch(ES_URL)


def ensure_indices(client: Elasticsearch):
    specs = {
        TRAINING_FEATURES_INDEX: _FEATURE_FIELDS,
        TRAINING_RESULTS_INDEX: _RESULTS_FIELDS,
        PREDICTIONS_INDEX: _PREDICTIONS_FIELDS,
    }
    for index, fields in specs.items():
        if not client.indices.exists(index=index):
            client.indices.create(
                index=index,
                mappings={"properties": fields, "dynamic": True},
            )


def bulk_index(client: Elasticsearch, index: str, docs, id_field: str | None = None):
    """docs: iterable of dicts. If id_field is given, its value becomes the ES
    document id, so re-running a step upserts instead of duplicating."""

    def _actions():
        for doc in docs:
            action = {"_index": index, "_source": doc}
            if id_field and id_field in doc:
                action["_id"] = doc[id_field]
            yield action

    helpers.bulk(client, _actions())

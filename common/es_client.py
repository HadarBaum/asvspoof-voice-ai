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

from common.features import FEATURE_NAMES

ES_URL = os.environ.get("ES_URL", "http://localhost:9200")

TRAINING_FEATURES_INDEX = "asvspoof-training-features"
TRAINING_RESULTS_INDEX = "asvspoof-training-results"
PREDICTIONS_INDEX = "asvspoof-predictions"

EMBEDDING_FIELD = "embedding"
EMBEDDING_DIMS = len(FEATURE_NAMES)

_FEATURE_FIELDS = {
    # dynamic mapping handles the ~53 numeric feature columns fine; we only
    # need to pin down the fields we actually query/aggregate on.
    "key": {"type": "keyword"},
    "partition": {"type": "keyword"},
    "speaker_id": {"type": "keyword"},
    "attack_id": {"type": "keyword"},
    "label": {"type": "keyword"},
    # The standardized feature vector, indexed for k-NN similarity search (see
    # pipeline/index_embeddings.py and app/server.py's /similar route). This is
    # the "embeddings and semantic search" AI capability option from the course
    # brief - built on the same acoustic features already computed for the
    # classifier rather than a separate embedding model, since those features
    # are exactly the thing meant to capture "how does this clip sound."
    EMBEDDING_FIELD: {"type": "dense_vector", "dims": EMBEDDING_DIMS, "index": True, "similarity": "cosine"},
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
    "roc_auc_dev": {"type": "float"},
    "eer_dev": {"type": "float"},
    "model_name": {"type": "keyword"},
    "decision_threshold": {"type": "float"},
    "recall_bonafide_dev": {"type": "float"},
    "accuracy_dev_default_threshold_0.5": {"type": "float"},
    "recall_bonafide_dev_default_threshold_0.5": {"type": "float"},
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


def ensure_embedding_mapping(client: Elasticsearch):
    """Adds the embedding field to an already-existing training-features index.
    ensure_indices() only sets up the mapping for an index it creates fresh;
    this covers the (very real, this-project-hit-it) case where the index
    already existed from an earlier pipeline run before embeddings existed.
    Elasticsearch allows adding new fields to an existing mapping in place -
    no reindex needed - as long as the field didn't exist under a conflicting
    type already."""
    client.indices.put_mapping(
        index=TRAINING_FEATURES_INDEX,
        properties={EMBEDDING_FIELD: _FEATURE_FIELDS[EMBEDDING_FIELD]},
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

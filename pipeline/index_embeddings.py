"""
Step 3b of the pipeline (runs after batch_feature_extraction.py, alongside
train_classifier.py): builds the "embeddings and semantic search" AI capability
from the course brief.

Reuses the same acoustic feature vectors already computed for the classifier
as the embedding - no separate embedding model - but standardizes them first
(zero mean, unit variance per feature, fit on train only) before storing them
as Elasticsearch dense_vector fields, since raw features span wildly different
scales (duration_seconds ~1-10 vs spectral_rolloff ~thousands) and cosine
similarity over unscaled features would just be dominated by whichever feature
has the largest absolute magnitude, not the most acoustically meaningful one.

The fitted scaler is saved so app/server.py's /similar route can standardize
a freshly-uploaded clip's features the exact same way before searching.

Usage:
    python -m pipeline.index_embeddings --features pipeline/features_train_dev_full.parquet --scaler-out models/embedding_scaler.joblib
"""

import argparse
import os

import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

from common import es_client
from common.features import FEATURE_NAMES


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True, help="Parquet feature table path")
    parser.add_argument(
        "--scaler-out", default=os.path.join(os.path.dirname(__file__), "..", "models", "embedding_scaler.joblib")
    )
    args = parser.parse_args()

    df = pd.read_parquet(args.features)
    train_df = df[df["partition"] == "train"]
    if train_df.empty:
        raise ValueError("Feature table must contain a 'train' partition to fit the embedding scaler on.")

    scaler = StandardScaler()
    scaler.fit(train_df[FEATURE_NAMES])

    os.makedirs(os.path.dirname(args.scaler_out), exist_ok=True)
    joblib.dump(scaler, args.scaler_out)
    print(f"Fit embedding scaler on {len(train_df)} train rows, saved to {args.scaler_out}")

    embeddings = scaler.transform(df[FEATURE_NAMES])

    es = es_client.get_client()
    es_client.ensure_indices(es)
    es_client.ensure_embedding_mapping(es)

    docs = []
    for i, (_, row) in enumerate(df.iterrows()):
        docs.append(
            {
                "key": row["key"],
                "partition": row["partition"],
                "speaker_id": row["speaker_id"],
                "attack_id": row["attack_id"],
                "label": row["label"],
                es_client.EMBEDDING_FIELD: embeddings[i].tolist(),
            }
        )

    # Partial-update (not a full re-index) so we only touch the embedding field
    # and leave the raw feature columns batch_feature_extraction.py already
    # wrote for this same document untouched.
    def _actions():
        for doc in docs:
            yield {
                "_op_type": "update",
                "_index": es_client.TRAINING_FEATURES_INDEX,
                "_id": doc["key"],
                "doc": doc,
                "doc_as_upsert": True,
            }

    from elasticsearch import helpers

    success, errors = helpers.bulk(es, _actions(), raise_on_error=False, stats_only=False)
    print(f"Indexed embeddings for {success} utterances into {es_client.TRAINING_FEATURES_INDEX}.")
    if errors:
        print(f"WARN: {len(errors)} documents failed (showing first 3): {errors[:3]}")


if __name__ == "__main__":
    main()

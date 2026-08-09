"""
Knows the (slightly inconsistent) ASVspoof2019 LA directory/protocol layout,
so every other script can just ask "give me the labeled utterances for
partition X" without re-parsing protocol files itself.

Real dataset protocol file names: ASVspoof2019.LA.cm.train.trn.txt,
ASVspoof2019.LA.cm.dev.trl.txt, ASVspoof2019.LA.cm.eval.trl.txt.
Each non-empty line: "<speaker_id> <file_id> - <attack_id_or_dash> <bonafide|spoof>"

We match protocol files by glob (`*.cm.*{partition}*.txt`) rather than an exact
suffix so this also works against pipeline/dev_generate_synthetic_sample.py's
simplified file names during local testing.
"""

import glob
import os
from dataclasses import dataclass

PARTITION_FOLDERS = {
    "train": "ASVspoof2019_LA_train",
    "dev": "ASVspoof2019_LA_dev",
    "eval": "ASVspoof2019_LA_eval",
}


@dataclass(frozen=True)
class Utterance:
    partition: str
    speaker_id: str
    file_id: str
    attack_id: str  # "-" for bonafide
    label: str  # "bonafide" or "spoof"
    flac_path: str  # local filesystem path, used only for ingestion


def _find_protocol_file(la_root: str, partition: str) -> str:
    protocol_dir = os.path.join(la_root, "ASVspoof2019_LA_cm_protocols")
    matches = sorted(glob.glob(os.path.join(protocol_dir, f"*.cm.*{partition}*.txt")))
    if not matches:
        raise FileNotFoundError(f"No protocol file found for partition '{partition}' under {protocol_dir}")
    return matches[0]


def load_partition(la_root: str, partition: str) -> list[Utterance]:
    """la_root: path to the LA/ folder (contains ASVspoof2019_LA_train/, _dev/, _eval/, _cm_protocols/)."""
    protocol_path = _find_protocol_file(la_root, partition)
    flac_dir = os.path.join(la_root, PARTITION_FOLDERS[partition], "flac")

    utterances = []
    with open(protocol_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            speaker_id, file_id, _env, attack_id, label = parts[0], parts[1], parts[2], parts[3], parts[4]
            utterances.append(
                Utterance(
                    partition=partition,
                    speaker_id=speaker_id,
                    file_id=file_id,
                    attack_id=attack_id,
                    label=label,
                    flac_path=os.path.join(flac_dir, f"{file_id}.flac"),
                )
            )
    return utterances


def minio_key(utterance: Utterance) -> str:
    """Object key under the asvspoof-raw bucket in MinIO."""
    return f"LA/{PARTITION_FOLDERS[utterance.partition]}/flac/{utterance.file_id}.flac"

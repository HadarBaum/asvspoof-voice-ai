"""
DEV-ONLY UTILITY - NOT part of the graded pipeline.

Generates a small synthetic stand-in for the ASVspoof2019 LA dataset
(fake flac files + protocol text files, in the exact directory/format the
real dataset uses) so the rest of the pipeline can be built and tested
before Kaggle access is available.

"bonafide" clips: a fundamental tone with natural vibrato + breath noise.
"spoof" clips: a purer, perfectly stable harmonic tone with far less noise,
   mimicking the "too clean" artifact real TTS/voice-conversion systems
   tend to leave behind.

These signals are NOT real speech and the classifier trained on them will
NOT reflect real-world accuracy - this script exists purely so every stage
of the pipeline (ingest -> features -> train -> stream -> serve) can be
exercised end-to-end locally. Once the real dataset is downloaded, this
sample is superseded and the real numbers come from the real data.
"""

import os
import random

import numpy as np
import soundfile as sf

random.seed(0)
np.random.seed(0)

SR = 16000
OUT_ROOT = os.path.join(os.path.dirname(__file__), "..", "data", "sample", "LA")

PARTITIONS = {
    "train": ("ASVspoof2019_LA_train", 30),
    "dev": ("ASVspoof2019_LA_dev", 20),
    "eval": ("ASVspoof2019_LA_eval", 20),
}
ATTACK_IDS = [f"A{i:02d}" for i in range(1, 7)]  # toy subset of the real A01-A19 range


def synth_bonafide(duration_s: float) -> np.ndarray:
    t = np.linspace(0, duration_s, int(SR * duration_s), endpoint=False)
    f0 = 140 + 8 * np.sin(2 * np.pi * 4.5 * t)  # vibrato: pitch wobble ~4-5Hz
    phase = 2 * np.pi * np.cumsum(f0) / SR
    signal = 0.5 * np.sin(phase) + 0.15 * np.sin(2 * phase) + 0.05 * np.sin(3 * phase)
    signal += np.random.normal(0, 0.03, size=signal.shape)  # breath/room noise
    envelope = 0.6 + 0.4 * np.abs(np.sin(2 * np.pi * 2.0 * t))
    return (signal * envelope).astype(np.float32)


def synth_spoof(duration_s: float) -> np.ndarray:
    t = np.linspace(0, duration_s, int(SR * duration_s), endpoint=False)
    f0 = 150.0  # perfectly stable pitch, no vibrato - the "too clean" tell
    signal = 0.5 * np.sin(2 * np.pi * f0 * t) + 0.1 * np.sin(2 * np.pi * 2 * f0 * t)
    signal += np.random.normal(0, 0.005, size=signal.shape)  # near-silent noise floor
    return signal.astype(np.float32)


def main():
    protocol_dir = os.path.join(OUT_ROOT, "ASVspoof2019_LA_cm_protocols")
    os.makedirs(protocol_dir, exist_ok=True)

    for partition, (folder, n_files) in PARTITIONS.items():
        flac_dir = os.path.join(OUT_ROOT, folder, "flac")
        os.makedirs(flac_dir, exist_ok=True)
        protocol_lines = []

        for i in range(n_files):
            is_bonafide = i % 2 == 0
            speaker_id = f"LA_{1000 + (i % 5)}"
            file_id = f"LA_{partition[0].upper()}_{100000 + i}"
            duration = round(random.uniform(1.5, 3.5), 2)

            audio = synth_bonafide(duration) if is_bonafide else synth_spoof(duration)
            sf.write(os.path.join(flac_dir, f"{file_id}.flac"), audio, SR, format="FLAC")

            attack_id = "-" if is_bonafide else random.choice(ATTACK_IDS)
            key = "bonafide" if is_bonafide else "spoof"
            protocol_lines.append(f"{speaker_id} {file_id} - {attack_id} {key}")

        protocol_path = os.path.join(protocol_dir, f"ASVspoof2019.LA.cm.{partition}.txt")
        with open(protocol_path, "w") as f:
            f.write("\n".join(protocol_lines) + "\n")

        print(f"{partition}: wrote {n_files} files to {flac_dir}")
        print(f"{partition}: wrote protocol to {protocol_path}")


if __name__ == "__main__":
    main()

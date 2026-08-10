"""
Single source of truth for turning a raw audio clip into a fixed-length
numeric feature vector. Used by:
  - pipeline/batch_feature_extraction.py  (Spark batch job over train+dev)
  - pipeline/streaming_enrichment.py      (Kafka consumer, near-real-time)
  - app/server.py                          (the /classify demo endpoint)

Keeping this in one place guarantees the web app scores audio with the exact
same features the model was trained on.
"""

import io

import librosa
import numpy as np
import pandas as pd

SAMPLE_RATE = 16000  # ASVspoof2019 audio is already 16kHz mono; resample anything else to match
N_MFCC = 20

# Order matters: this defines both the dict keys AND the column order fed to the model.
FEATURE_NAMES = (
    [f"mfcc_{i}_mean" for i in range(N_MFCC)]
    + [f"mfcc_{i}_std" for i in range(N_MFCC)]
    + [
        "spectral_centroid_mean",
        "spectral_centroid_std",
        "spectral_bandwidth_mean",
        "spectral_bandwidth_std",
        "spectral_rolloff_mean",
        "spectral_rolloff_std",
        "zero_crossing_rate_mean",
        "zero_crossing_rate_std",
        "rmse_mean",
        "rmse_std",
        "pitch_mean",
        "pitch_std",
        "duration_seconds",
    ]
)


def extract_features(audio_bytes: bytes) -> dict:
    """Decode an audio file (flac/wav/mp3, any format librosa/soundfile can read)
    and compute a fixed-length dict of acoustic features.

    We use summary statistics (mean/std) of frame-level features rather than the
    raw frame sequences so every clip - regardless of length - maps to the same
    fixed-size vector, which is what a standard tabular classifier expects.
    """
    y, sr = librosa.load(io.BytesIO(audio_bytes), sr=SAMPLE_RATE, mono=True)

    if y.size == 0:
        # Guard against corrupt/empty clips; return a zero vector rather than crashing
        # a whole Spark partition or a live classify request on one bad file.
        return {name: 0.0 for name in FEATURE_NAMES}

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    zcr = librosa.feature.zero_crossing_rate(y=y)[0]
    rmse = librosa.feature.rms(y=y)[0]

    # Pitch (F0) tracking. librosa.pyin (probabilistic YIN with HMM smoothing over
    # a full 4+ octave vocal range) measured ~2-9s per clip in practice - at 50k+
    # clips that's many hours. Plain librosa.yin over a human-voice-only range
    # (50-500Hz covers the full range from a deep bass to a high soprano/child)
    # with a larger hop_length (fewer, coarser frames - pitch doesn't need
    # frame-by-frame resolution for a summary mean/std) gets the same kind of
    # signal in ~0.02s: no HMM smoothing, a ~4x narrower search range, and ~4x
    # fewer frames than the defaults used elsewhere in this function.
    f0 = librosa.yin(y, fmin=50, fmax=500, sr=sr, hop_length=2048)
    voiced_f0 = f0[np.isfinite(f0)]

    features = {}
    for i in range(N_MFCC):
        features[f"mfcc_{i}_mean"] = float(np.mean(mfcc[i]))
        features[f"mfcc_{i}_std"] = float(np.std(mfcc[i]))

    features["spectral_centroid_mean"] = float(np.mean(spectral_centroid))
    features["spectral_centroid_std"] = float(np.std(spectral_centroid))
    features["spectral_bandwidth_mean"] = float(np.mean(spectral_bandwidth))
    features["spectral_bandwidth_std"] = float(np.std(spectral_bandwidth))
    features["spectral_rolloff_mean"] = float(np.mean(spectral_rolloff))
    features["spectral_rolloff_std"] = float(np.std(spectral_rolloff))
    features["zero_crossing_rate_mean"] = float(np.mean(zcr))
    features["zero_crossing_rate_std"] = float(np.std(zcr))
    features["rmse_mean"] = float(np.mean(rmse))
    features["rmse_std"] = float(np.std(rmse))
    features["pitch_mean"] = float(np.mean(voiced_f0)) if voiced_f0.size else 0.0
    features["pitch_std"] = float(np.std(voiced_f0)) if voiced_f0.size else 0.0
    features["duration_seconds"] = float(len(y) / sr)

    return features


def features_to_vector(features: dict) -> pd.DataFrame:
    """Project a feature dict onto the fixed column order the model expects, as a
    single-row DataFrame - the model was trained on a DataFrame with these column
    names, and predicting from a plain array instead just makes scikit-learn warn
    that the names don't match (it still works, but the warning is noise)."""
    return pd.DataFrame([[features[name] for name in FEATURE_NAMES]], columns=FEATURE_NAMES)

"""
Audio quality scoring — pitch stability, clarity, energy, onset, noise floor.
"""

import logging
import numpy as np
import librosa

from backend.config import CompRules

log = logging.getLogger("comper")


def score_audio_chunk(audio: np.ndarray, sr: int) -> dict:
    """
    Score a chunk of audio on multiple dimensions.
    Returns dict with scores 0-1 for each dimension.
    """
    scores = {}
    rms = np.sqrt(np.mean(audio ** 2))

    # Silence detection
    if rms < 1e-6:
        return {
            "pitch_stability": 0, "clarity": 0, "energy": 0,
            "onset_strength": 0, "noise_floor": 0,
        }

    # Pitch stability (pYIN)
    try:
        f0, _, voiced_probs = librosa.pyin(
            audio, sr=sr,
            fmin=librosa.note_to_hz("C2"),
            fmax=librosa.note_to_hz("C6"),
            hop_length=512,
        )
        f0_valid = f0[~np.isnan(f0)]
        if len(f0_valid) > 2:
            median_f0 = np.median(f0_valid)
            cents_dev = 1200 * np.log2(f0_valid / median_f0 + 1e-10)
            stability = max(0, 1 - np.std(cents_dev) / 50)
        else:
            stability = 0.5
        scores["pitch_stability"] = stability
    except Exception:
        scores["pitch_stability"] = 0.5

    # Clarity (spectral flatness — lower = more tonal = better)
    flatness = librosa.feature.spectral_flatness(y=audio, hop_length=512)
    scores["clarity"] = float(1 - np.mean(flatness))

    # Energy consistency (lower variance = more consistent)
    rms_frames = librosa.feature.rms(y=audio, hop_length=512)[0]
    if rms_frames.std() > 0:
        scores["energy"] = float(1 - min(1, rms_frames.std() / rms_frames.mean()))
    else:
        scores["energy"] = 0.5

    # Onset strength (articulation quality)
    onset_env = librosa.onset.onset_strength(y=audio, sr=sr, hop_length=512)
    scores["onset_strength"] = float(min(1, np.mean(onset_env) / 3))

    # Noise floor (SNR estimate)
    spectral = np.abs(librosa.stft(audio, hop_length=512))
    noise_est = np.percentile(spectral, 10)
    signal_est = np.percentile(spectral, 90)
    if signal_est > 0:
        snr = signal_est / (noise_est + 1e-10)
        scores["noise_floor"] = float(min(1, snr / 20))
    else:
        scores["noise_floor"] = 0.5

    return scores


def compute_weighted_score(scores: dict, rules: CompRules) -> float:
    """Compute weighted total from individual scores."""
    return (
        scores.get("pitch_stability", 0) * rules.weight_pitch_stability +
        scores.get("clarity", 0) * rules.weight_clarity +
        scores.get("energy", 0) * rules.weight_energy +
        scores.get("onset_strength", 0) * rules.weight_onset_strength +
        scores.get("noise_floor", 0) * rules.weight_noise_floor
    )

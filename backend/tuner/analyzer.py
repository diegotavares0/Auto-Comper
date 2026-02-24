"""
Pitch analysis — pYIN detection + Krumhansl-Schmuckler key estimation.
"""

import logging
from typing import Callable, Optional

import numpy as np
import librosa

from backend.config import TunerConfig
from backend.utils.musical_constants import (
    hz_to_midi, midi_to_name, NOTE_NAMES, SCALES,
)

log = logging.getLogger("comper")

# Krumhansl-Kessler key profiles (empirical pitch-class distributions)
_MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
                           2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
_MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
                           2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


def _get_pitch_range(instrument_mode: str):
    """Get fmin/fmax in Hz based on instrument mode."""
    if instrument_mode == "guitar":
        return librosa.note_to_hz("E2"), librosa.note_to_hz("E5")
    # voice or auto
    return librosa.note_to_hz("C2"), librosa.note_to_hz("C6")


def _estimate_key(midi_values: np.ndarray, voiced_probs: np.ndarray) -> dict:
    """
    Estimate the musical key using Krumhansl-Schmuckler algorithm.
    Returns: {root: str, scale: str, confidence: float}
    """
    if len(midi_values) < 10:
        return {"root": "C", "scale": "chromatic", "confidence": 0.0}

    # Build weighted pitch-class histogram
    pitch_classes = np.round(midi_values) % 12
    histogram = np.zeros(12)
    for pc, prob in zip(pitch_classes.astype(int), voiced_probs):
        histogram[pc % 12] += prob

    if histogram.sum() < 1:
        return {"root": "C", "scale": "chromatic", "confidence": 0.0}

    # Normalize
    histogram = histogram / histogram.sum()

    # Correlate against all 12 rotations of major and minor profiles
    best_corr = -1
    best_root = 0
    best_scale = "major"
    second_best = -1

    for root in range(12):
        rotated_hist = np.roll(histogram, -root)

        for profile, scale_name in [(_MAJOR_PROFILE, "major"),
                                     (_MINOR_PROFILE, "minor")]:
            corr = np.corrcoef(rotated_hist, profile)[0, 1]

            if corr > best_corr:
                second_best = best_corr
                best_corr = corr
                best_root = root
                best_scale = scale_name
            elif corr > second_best:
                second_best = corr

    # Confidence: gap between best and second-best correlation
    confidence = max(0, min(1, (best_corr - second_best) * 3))

    return {
        "root": NOTE_NAMES[best_root],
        "scale": best_scale,
        "confidence": round(confidence, 3),
    }


def _downsample_pitch_curve(f0: np.ndarray, hop_samples: int,
                            sr: int, target_fps: float = 10.0) -> list:
    """
    Downsample pitch data to ~target_fps points per second for visualization.
    Returns list of {t: seconds, hz: float} dicts (NaN frames skipped).
    """
    frame_rate = sr / hop_samples
    step = max(1, int(frame_rate / target_fps))
    curve = []
    for i in range(0, len(f0), step):
        if not np.isnan(f0[i]):
            t = i * hop_samples / sr
            curve.append({"t": round(t, 3), "hz": round(float(f0[i]), 1)})
    return curve


def analyze_pitch(audio: np.ndarray, sr: int, config: TunerConfig,
                  progress_cb: Optional[Callable] = None) -> dict:
    """
    Full pitch analysis: pYIN detection + key estimation.
    progress_cb(pct, msg) reports 5-35%.
    Returns analysis dict with f0, midi, voiced info, key estimate, stats.
    """
    if progress_cb:
        progress_cb(8, "Detectando pitch (pYIN)...")

    hop_length = int(sr * config.hop_ms / 1000)
    fmin, fmax = _get_pitch_range(config.instrument_mode)

    # Run pYIN
    f0, voiced_flag, voiced_probs = librosa.pyin(
        audio, sr=sr, fmin=fmin, fmax=fmax, hop_length=hop_length,
    )

    if progress_cb:
        progress_cb(20, "Analise de pitch concluida, processando...")

    # Convert to MIDI
    f0_clean = f0.copy()
    midi = np.zeros_like(f0)

    # Filter by confidence threshold
    confident = voiced_probs >= config.pitch_confidence_threshold
    f0_clean[~confident] = np.nan

    # Convert valid Hz to MIDI
    valid_mask = ~np.isnan(f0_clean)
    if valid_mask.any():
        midi[valid_mask] = hz_to_midi(f0_clean[valid_mask])

    # Pitch stats
    f0_valid = f0_clean[valid_mask]
    voiced_pct = round(valid_mask.sum() / len(f0) * 100, 1) if len(f0) > 0 else 0

    if len(f0_valid) > 2:
        median_hz = float(np.median(f0_valid))
        mean_hz = float(np.mean(f0_valid))
        cents_from_median = 1200 * np.log2(f0_valid / median_hz + 1e-10)
        std_cents = float(np.std(cents_from_median))
        midi_valid = midi[valid_mask]
        range_semitones = float(midi_valid.max() - midi_valid.min())
    else:
        median_hz = 0
        mean_hz = 0
        std_cents = 0
        range_semitones = 0

    if progress_cb:
        progress_cb(25, f"Analise: {voiced_pct:.0f}% com voz detectada")

    # Key estimation
    if progress_cb:
        progress_cb(30, "Estimando tonalidade...")

    midi_voiced = midi[valid_mask]
    probs_voiced = voiced_probs[valid_mask] if valid_mask.any() else np.array([])
    key_info = _estimate_key(midi_voiced, probs_voiced)

    # If user set root_note="auto", use detected key
    effective_root = key_info["root"] if config.root_note == "auto" else config.root_note
    effective_scale = key_info["scale"] if config.root_note == "auto" else config.scale_type

    if progress_cb:
        progress_cb(33, f"Tom detectado: {key_info['root']} {key_info['scale']} "
                        f"(confianca: {key_info['confidence']*100:.0f}%)")

    # Downsample pitch curve for visualization
    pitch_curve = _downsample_pitch_curve(f0_clean, hop_length, sr)

    log.info(f"  Pitch analysis: {voiced_pct:.1f}% voiced, "
             f"median={median_hz:.1f}Hz, std={std_cents:.1f} cents, "
             f"key={key_info['root']} {key_info['scale']} "
             f"(conf={key_info['confidence']:.2f})")

    return {
        "f0": f0_clean,
        "voiced_flag": valid_mask,
        "voiced_probs": voiced_probs,
        "midi": midi,
        "hop_length": hop_length,
        "sr": sr,
        "estimated_root": key_info["root"],
        "estimated_scale": key_info["scale"],
        "key_confidence": key_info["confidence"],
        "effective_root": effective_root,
        "effective_scale": effective_scale,
        "pitch_stats": {
            "mean_hz": round(mean_hz, 1),
            "median_hz": round(median_hz, 1),
            "std_cents": round(std_cents, 1),
            "voiced_pct": voiced_pct,
            "range_semitones": round(range_semitones, 1),
        },
        "pitch_curve_original": pitch_curve,
    }

"""
Take normalizer — tempo alignment and pitch centering across takes.

Designed for the common recording scenario: ~16 takes recorded without a
click track.  Each take drifts slightly in tempo and pitch.  Normalizing
*before* scoring/assembly makes splices sound like one continuous performance.

Both tempo and pitch normalization are intensity-controlled (0-100%).
At intensity=0 the normalizer is a pure no-op — zero CPU cost.
"""

import logging
from typing import List, Optional, Callable

import numpy as np
import librosa

log = logging.getLogger("comper")


# ─────────────────────────────────────────────────
# Estimation helpers
# ─────────────────────────────────────────────────

def estimate_tempo(audio: np.ndarray, sr: int) -> float:
    """
    Estimate tempo (BPM) of an audio clip.

    Uses librosa's beat tracker which combines onset envelope analysis
    with dynamic programming to find the most likely tempo.

    Returns 0.0 if estimation fails (too short, no percussive content, etc.).
    """
    try:
        # librosa.beat.beat_track returns (tempo, beat_frames)
        # tempo is a scalar BPM estimate
        tempo, _ = librosa.beat.beat_track(y=audio, sr=sr)
        # librosa >= 0.10 returns an array; squeeze to scalar
        bpm = float(np.squeeze(tempo))
        if np.isnan(bpm) or bpm <= 0:
            return 0.0
        return bpm
    except Exception as e:
        log.warning(f"  Tempo estimation failed: {e}")
        return 0.0


def estimate_pitch_center(audio: np.ndarray, sr: int) -> float:
    """
    Estimate the median fundamental frequency (Hz) of an audio clip.

    Uses pYIN pitch detection (same algo used in scoring.py and tuner).
    Returns the median of all voiced frames, ignoring silence/unvoiced.
    Returns 0.0 if estimation fails.
    """
    try:
        f0, voiced_flag, voiced_probs = librosa.pyin(
            audio, sr=sr,
            fmin=librosa.note_to_hz("C2"),
            fmax=librosa.note_to_hz("C6"),
            hop_length=512,
        )
        # Keep only voiced frames with reasonable confidence
        valid_mask = ~np.isnan(f0) & (voiced_probs > 0.5)
        f0_valid = f0[valid_mask]

        if len(f0_valid) < 5:
            return 0.0

        return float(np.median(f0_valid))

    except Exception as e:
        log.warning(f"  Pitch center estimation failed: {e}")
        return 0.0


# ─────────────────────────────────────────────────
# Single-take normalization
# ─────────────────────────────────────────────────

def normalize_tempo(
    audio: np.ndarray,
    sr: int,
    source_bpm: float,
    target_bpm: float,
    intensity: float,
) -> np.ndarray:
    """
    Time-stretch audio so its tempo matches the target BPM.

    Parameters
    ----------
    audio : np.ndarray
        Input audio signal.
    sr : int
        Sample rate (not used by time_stretch but kept for API consistency).
    source_bpm : float
        Estimated tempo of this take.
    target_bpm : float
        Target tempo (usually the median across all takes).
    intensity : float
        0-100. How much correction to apply.
        0 = no change, 100 = fully lock to target_bpm.

    Returns
    -------
    np.ndarray : Time-stretched audio.
    """
    if intensity <= 0 or source_bpm <= 0 or target_bpm <= 0:
        return audio

    # Compute the full stretch ratio
    full_rate = target_bpm / source_bpm

    # Blend with identity (1.0) based on intensity
    # intensity=100 → use full_rate
    # intensity=0  → use 1.0 (no change)
    effective_rate = 1.0 + (full_rate - 1.0) * (intensity / 100.0)

    # Skip if the correction is negligible (< 0.1% tempo difference)
    if abs(effective_rate - 1.0) < 0.001:
        return audio

    # Cap at reasonable limits to avoid artifacts (±20% max)
    effective_rate = np.clip(effective_rate, 0.8, 1.2)

    try:
        stretched = librosa.effects.time_stretch(audio, rate=effective_rate)
        return stretched
    except Exception as e:
        log.warning(f"  Time stretch failed (rate={effective_rate:.3f}): {e}")
        return audio


def normalize_pitch(
    audio: np.ndarray,
    sr: int,
    source_center_hz: float,
    target_center_hz: float,
    intensity: float,
) -> np.ndarray:
    """
    Pitch-shift audio so its center frequency matches the target.

    Parameters
    ----------
    audio : np.ndarray
        Input audio signal.
    sr : int
        Sample rate.
    source_center_hz : float
        Median fundamental of this take.
    target_center_hz : float
        Target median fundamental (usually median across all takes).
    intensity : float
        0-100. How much correction to apply.
        0 = no change, 100 = fully snap to target pitch center.

    Returns
    -------
    np.ndarray : Pitch-shifted audio.
    """
    if intensity <= 0 or source_center_hz <= 0 or target_center_hz <= 0:
        return audio

    # Compute shift in semitones: 12 * log2(target / source)
    full_shift_semitones = 12 * np.log2(target_center_hz / source_center_hz)

    # Blend with zero (no shift) based on intensity
    effective_shift = full_shift_semitones * (intensity / 100.0)

    # Skip if the correction is negligible (< 2 cents)
    if abs(effective_shift) < 0.02:
        return audio

    # Cap at ±2 semitones to avoid audible artifacts
    effective_shift = np.clip(effective_shift, -2.0, 2.0)

    try:
        shifted = librosa.effects.pitch_shift(
            audio, sr=sr, n_steps=effective_shift
        )
        return shifted
    except Exception as e:
        log.warning(f"  Pitch shift failed (shift={effective_shift:.2f}st): {e}")
        return audio


# ─────────────────────────────────────────────────
# Orchestrator: normalize all takes
# ─────────────────────────────────────────────────

def normalize_takes(
    takes: List[np.ndarray],
    sr: int,
    tempo_intensity: float = 0.0,
    pitch_intensity: float = 0.0,
    progress_cb: Optional[Callable] = None,
) -> List[np.ndarray]:
    """
    Normalize tempo and/or pitch across a list of takes.

    Strategy:
    - Estimate tempo/pitch center for each take
    - Use the MEDIAN as the target (robust to outliers)
    - Apply per-take normalization with intensity blending

    The median-as-target strategy is important: if you have 16 takes
    and 14 are at ~80 BPM but 2 are at ~75 BPM, the median picks ~80 BPM.
    Those 2 outliers get stretched, the 14 majority stay nearly untouched.

    Parameters
    ----------
    takes : List[np.ndarray]
        Audio arrays (all same sr, may differ in length).
    sr : int
        Sample rate.
    tempo_intensity : float
        0-100. Time-stretch correction intensity.
    pitch_intensity : float
        0-100. Pitch centering intensity.
    progress_cb : callable, optional
        Progress callback(pct, msg).

    Returns
    -------
    List[np.ndarray] : Normalized takes (same order, may differ in length
                        from originals if tempo-corrected).
    """
    # Skip entirely if both intensities are zero
    if tempo_intensity <= 0 and pitch_intensity <= 0:
        return takes, None

    n = len(takes)

    def progress(pct, msg):
        if progress_cb:
            progress_cb(pct, msg)

    # ── Phase 1: Estimate tempo + pitch for all takes ──
    tempos = []
    pitches = []

    for i, audio in enumerate(takes):
        progress(16 + int(3 * i / n), f"Estimando tempo/pitch: take {i + 1}/{n}...")

        bpm = estimate_tempo(audio, sr) if tempo_intensity > 0 else 0.0
        hz = estimate_pitch_center(audio, sr) if pitch_intensity > 0 else 0.0

        tempos.append(bpm)
        pitches.append(hz)

        log.info(f"  Take {i + 1}: {bpm:.1f} BPM, pitch center {hz:.1f} Hz")

    # ── Phase 2: Compute targets (median of valid values) ──
    valid_tempos = [t for t in tempos if t > 0]
    valid_pitches = [p for p in pitches if p > 0]

    target_bpm = float(np.median(valid_tempos)) if valid_tempos else 0.0
    target_hz = float(np.median(valid_pitches)) if valid_pitches else 0.0

    log.info(f"  Normalization targets: {target_bpm:.1f} BPM, {target_hz:.1f} Hz")
    progress(19, f"Alvo: {target_bpm:.0f} BPM, {target_hz:.0f} Hz")

    # ── Phase 3: Apply normalization per take ──
    normalized = []

    for i, audio in enumerate(takes):
        pct = 19 + int(4 * (i + 1) / n)
        progress(pct, f"Normalizando take {i + 1}/{n}...")

        result = audio

        # Tempo first (changes duration → must happen before pitch)
        if tempo_intensity > 0 and tempos[i] > 0 and target_bpm > 0:
            result = normalize_tempo(
                result, sr, tempos[i], target_bpm, tempo_intensity,
            )

        # Pitch second
        if pitch_intensity > 0 and pitches[i] > 0 and target_hz > 0:
            result = normalize_pitch(
                result, sr, pitches[i], target_hz, pitch_intensity,
            )

        normalized.append(result)

    # ── Summary & Stats ──
    norm_stats = {
        "enabled": True,
        "tempo_intensity": tempo_intensity,
        "pitch_intensity": pitch_intensity,
        "target_bpm": round(target_bpm, 1) if target_bpm > 0 else None,
        "target_hz": round(target_hz, 1) if target_hz > 0 else None,
        "per_take": [],
    }

    for i in range(n):
        take_info = {"take": i + 1}
        if tempo_intensity > 0 and tempos[i] > 0:
            take_info["original_bpm"] = round(tempos[i], 1)
            take_info["bpm_correction"] = round(target_bpm - tempos[i], 1) if target_bpm > 0 else 0
        if pitch_intensity > 0 and pitches[i] > 0:
            take_info["original_hz"] = round(pitches[i], 1)
            if target_hz > 0:
                take_info["pitch_shift_cents"] = round(
                    1200 * np.log2(target_hz / pitches[i]), 1
                )
            else:
                take_info["pitch_shift_cents"] = 0
        norm_stats["per_take"].append(take_info)

    if tempo_intensity > 0 and valid_tempos:
        spread_before = max(valid_tempos) - min(valid_tempos)
        norm_stats["tempo_spread_bpm"] = round(spread_before, 1)
        log.info(f"  Tempo spread before: {spread_before:.1f} BPM "
                 f"(correction intensity: {tempo_intensity}%)")

    if pitch_intensity > 0 and valid_pitches:
        spread_cents = 1200 * np.log2(max(valid_pitches) / min(valid_pitches))
        norm_stats["pitch_spread_cents"] = round(spread_cents, 1)
        log.info(f"  Pitch spread before: {spread_cents:.1f} cents "
                 f"(correction intensity: {pitch_intensity}%)")

    progress(23, "Normalizacao concluida")

    return normalized, norm_stats

"""
Audio assembly — crossfade join, per-block gain matching, LUFS normalization,
peak limiter.
"""

import logging
from typing import List

import numpy as np
from scipy.signal import sosfilt, butter

from backend.config import CompRules

log = logging.getLogger("comper")


def crossfade_join(seg_a: np.ndarray, seg_b: np.ndarray,
                   crossfade_samples: int) -> np.ndarray:
    """Join two segments with equal-power crossfade."""
    if crossfade_samples <= 0 or len(seg_a) < crossfade_samples:
        return np.concatenate([seg_a, seg_b])

    # Equal-power curves (sin/cos) sound smoother than linear for music
    t = np.linspace(0, np.pi / 2, crossfade_samples)
    fade_out = np.cos(t)
    fade_in = np.sin(t)

    overlap_a = seg_a[-crossfade_samples:] * fade_out
    overlap_b = seg_b[:crossfade_samples] * fade_in

    return np.concatenate([
        seg_a[:-crossfade_samples],
        overlap_a + overlap_b,
        seg_b[crossfade_samples:]
    ])


def _rms(audio: np.ndarray) -> float:
    """Compute RMS of audio, avoiding log of zero."""
    val = np.sqrt(np.mean(audio ** 2))
    return max(val, 1e-10)


def _rms_db(audio: np.ndarray) -> float:
    """RMS in dB."""
    return 20 * np.log10(_rms(audio))


def assemble_comp(takes: List[np.ndarray], decisions: List[dict],
                  sr: int, rules: CompRules) -> np.ndarray:
    """
    Assemble the final comp from block decisions with per-block gain matching.

    Each block is gently gain-adjusted toward the median loudness
    of all selected blocks, preventing jarring volume jumps between takes.
    """
    crossfade_samples = int(sr * rules.crossfade_ms / 1000)

    # ── Step 1: Extract all segments first ──
    segments = []
    for dec in decisions:
        take_idx = dec["take_idx"]
        start = int(dec["start_s"] * sr)
        end = int(dec["end_s"] * sr)
        segment = takes[take_idx][start:end]
        segments.append(segment)

    # ── Step 2: Per-block gain matching ──
    # Compute RMS for each segment, then match to median
    rms_values = [_rms_db(seg) for seg in segments if len(seg) > 0]

    if len(rms_values) >= 2:
        target_rms_db = float(np.median(rms_values))
        log.info(f"  Block gain matching: target RMS = {target_rms_db:.1f} dB")

        for i, seg in enumerate(segments):
            if len(seg) == 0:
                continue
            seg_rms_db = _rms_db(seg)
            diff_db = target_rms_db - seg_rms_db

            # Soft correction: only apply 70% of the difference
            # to avoid pumping and preserve natural dynamics
            correction_db = diff_db * 0.7

            # Clamp: don't boost/cut more than 6dB per block
            correction_db = max(-6.0, min(6.0, correction_db))

            if abs(correction_db) > 0.3:  # Only apply if meaningful
                gain = 10 ** (correction_db / 20)
                segments[i] = seg * gain
                log.info(f"    Block {i}: {seg_rms_db:.1f} dB → "
                         f"adjusted {correction_db:+.1f} dB")

    # ── Step 3: Crossfade join ──
    comp = np.array([], dtype=np.float64)
    for seg in segments:
        if len(comp) > 0:
            comp = crossfade_join(comp, seg, crossfade_samples)
        else:
            comp = seg

    return comp


# ─────────────────────────────────────────────
# K-weighted LUFS approximation
# ─────────────────────────────────────────────

def _k_weight_filter(sr: int):
    """
    Approximate K-weighting filter (EBU R128 / ITU-R BS.1770).
    Two stages: high-shelf boost around 2kHz + high-pass at 60Hz.
    Returns second-order sections (sos) for stable filtering.
    """
    filters = []

    # Stage 1: High-shelf — boosts ~2kHz+ by ~4dB
    # Approximated as a 1st-order high-shelf via biquad
    fc = 1500.0 / (sr / 2)
    fc = min(fc, 0.99)
    try:
        sos_shelf = butter(1, fc, btype='high', output='sos')
        filters.append(sos_shelf)
    except Exception:
        pass

    # Stage 2: High-pass at 60Hz (removes sub-bass rumble)
    fc_hp = 60.0 / (sr / 2)
    fc_hp = max(fc_hp, 0.001)
    try:
        sos_hp = butter(2, fc_hp, btype='high', output='sos')
        filters.append(sos_hp)
    except Exception:
        pass

    if filters:
        return np.vstack(filters)
    return None


def normalize_lufs(audio: np.ndarray, sr: int,
                   target_lufs: float = -16.0) -> np.ndarray:
    """
    LUFS-like normalization with K-weighting approximation.

    Uses a K-weight filter before measuring loudness, which is
    significantly more accurate than raw RMS for perceived loudness.
    """
    if len(audio) == 0:
        return audio

    rms = _rms(audio)
    if rms < 1e-10:
        return audio

    # Apply K-weighting for more accurate loudness measurement
    sos = _k_weight_filter(sr)
    if sos is not None:
        try:
            weighted = sosfilt(sos, audio)
            weighted_rms = _rms(weighted)
            # K-weighted loudness (closer to true LUFS)
            current_lufs = 20 * np.log10(weighted_rms) - 0.691
        except Exception:
            # Fallback to simple RMS
            current_lufs = 20 * np.log10(rms) - 3.0
    else:
        current_lufs = 20 * np.log10(rms) - 3.0

    gain_db = target_lufs - current_lufs
    # Safety: clamp gain to avoid extreme amplification
    gain_db = max(-20.0, min(20.0, gain_db))
    gain = 10 ** (gain_db / 20)

    result = audio * gain

    log.info(f"  LUFS normalization: measured ≈ {current_lufs:.1f} LUFS, "
             f"target = {target_lufs:.1f}, gain = {gain_db:+.1f} dB")

    # Soft clip if peaks exceed 0.98
    peak = np.abs(result).max()
    if peak > 0.98:
        result *= 0.98 / peak
        log.info(f"  Peak reduction: {peak:.3f} → 0.98")

    return result


def peak_limit(audio: np.ndarray, ceiling: float = 0.95) -> np.ndarray:
    """
    Safety peak limiter with soft-knee behavior.

    Instead of hard ratio limiting (which squashes dynamics),
    uses tanh-based soft clipping above the ceiling.
    """
    peak = np.abs(audio).max()
    if peak <= ceiling:
        return audio

    # Soft-knee: apply tanh compression above ceiling
    # This preserves transients better than hard ratio
    scale = ceiling / peak
    if scale > 0.8:
        # Close to ceiling — simple scale is fine
        audio = audio * (ceiling / peak)
    else:
        # Far above ceiling — use tanh soft clip
        normalized = audio / peak  # normalize to [-1, 1]
        audio = np.tanh(normalized * 1.5) * ceiling

    log.info(f"  Peak limiter: {peak:.3f} → {ceiling}")
    return audio

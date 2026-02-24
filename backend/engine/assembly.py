"""
Audio assembly — crossfade join, LUFS normalization, peak limiter.
"""

import logging
from typing import List

import numpy as np

from backend.config import CompRules

log = logging.getLogger("comper")


def crossfade_join(seg_a: np.ndarray, seg_b: np.ndarray,
                   crossfade_samples: int) -> np.ndarray:
    """Join two segments with linear crossfade."""
    if crossfade_samples <= 0 or len(seg_a) < crossfade_samples:
        return np.concatenate([seg_a, seg_b])

    fade_out = np.linspace(1, 0, crossfade_samples)
    fade_in = np.linspace(0, 1, crossfade_samples)

    overlap_a = seg_a[-crossfade_samples:] * fade_out
    overlap_b = seg_b[:crossfade_samples] * fade_in

    return np.concatenate([
        seg_a[:-crossfade_samples],
        overlap_a + overlap_b,
        seg_b[crossfade_samples:]
    ])


def assemble_comp(takes: List[np.ndarray], decisions: List[dict],
                  sr: int, rules: CompRules) -> np.ndarray:
    """Assemble the final comp from block decisions."""
    crossfade_samples = int(sr * rules.crossfade_ms / 1000)
    comp = np.array([], dtype=np.float64)

    for dec in decisions:
        take_idx = dec["take_idx"]
        start = int(dec["start_s"] * sr)
        end = int(dec["end_s"] * sr)
        segment = takes[take_idx][start:end]

        if len(comp) > 0:
            comp = crossfade_join(comp, segment, crossfade_samples)
        else:
            comp = segment

    return comp


def normalize_lufs(audio: np.ndarray, sr: int,
                   target_lufs: float = -16.0) -> np.ndarray:
    """Simple LUFS-like normalization (RMS-based approximation)."""
    rms = np.sqrt(np.mean(audio ** 2))
    if rms < 1e-10:
        return audio
    current_db = 20 * np.log10(rms)
    current_lufs_approx = current_db - 3
    gain_db = target_lufs - current_lufs_approx
    gain = 10 ** (gain_db / 20)
    result = audio * gain
    peak = np.abs(result).max()
    if peak > 0.98:
        result *= 0.98 / peak
    return result


def peak_limit(audio: np.ndarray, ceiling: float = 0.95) -> np.ndarray:
    """Safety peak limiter."""
    peak = np.abs(audio).max()
    if peak > ceiling:
        audio = audio * (ceiling / peak)
        log.info(f"  Peak limiter: {peak:.3f} -> {ceiling}")
    return audio

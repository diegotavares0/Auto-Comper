"""
Audio segmentation — musical block detection, fixed blocks, custom blocks.
"""

import logging
from typing import List, Tuple

import numpy as np
import librosa

from backend.config import CompRules

log = logging.getLogger("comper")


def detect_musical_blocks(audio: np.ndarray, sr: int,
                          rules: CompRules) -> List[Tuple[int, int]]:
    """
    Detect musical phrase boundaries using energy envelope analysis.
    Finds pauses and significant energy dips as phrase boundaries.
    """
    min_samples = int(sr * rules.min_segment_ms / 1000)
    target_samples = int(sr * rules.target_segment_ms / 1000)
    max_samples = int(sr * rules.max_segment_ms / 1000)

    # Compute energy envelope
    hop = 2048
    rms = librosa.feature.rms(y=audio, hop_length=hop)[0]

    # Smooth envelope (~1 second window)
    smooth_window = max(1, int(sr / hop))
    rms_smooth = np.convolve(rms, np.ones(smooth_window) / smooth_window, mode='same')

    # Find energy dips (potential phrase boundaries)
    dip_threshold = 0.4
    candidates = []

    window = max(1, int(2 * sr / hop))  # ~2 second context
    for i in range(window, len(rms_smooth) - window):
        local_avg = np.mean(rms_smooth[max(0, i - window):i + window])
        if local_avg > 0 and rms_smooth[i] < local_avg * dip_threshold:
            sample_pos = i * hop
            candidates.append(sample_pos)

    # Also detect actual silence
    silence_threshold = np.percentile(rms, 10) * 2
    for i in range(len(rms)):
        if rms[i] < silence_threshold:
            sample_pos = i * hop
            if not any(abs(sample_pos - c) < min_samples for c in candidates):
                candidates.append(sample_pos)

    candidates.sort()

    # Build blocks respecting min/max duration
    blocks = []
    prev = 0

    for candidate in candidates:
        block_len = candidate - prev

        if block_len < min_samples:
            continue
        elif block_len > max_samples:
            while candidate - prev > max_samples:
                split_point = prev + target_samples
                blocks.append((prev, split_point))
                prev = split_point
            if candidate - prev >= min_samples:
                blocks.append((prev, candidate))
                prev = candidate
        else:
            blocks.append((prev, candidate))
            prev = candidate

    # Final block
    if prev < len(audio):
        remaining = len(audio) - prev
        if remaining < min_samples and blocks:
            blocks[-1] = (blocks[-1][0], len(audio))
        else:
            blocks.append((prev, len(audio)))

    # Fallback: split at target intervals
    if not blocks:
        for start in range(0, len(audio), target_samples):
            end = min(start + target_samples, len(audio))
            if end - start >= min_samples:
                blocks.append((start, end))
            elif blocks:
                blocks[-1] = (blocks[-1][0], end)

    log.info(f"  {len(blocks)} blocos musicais detectados")
    for i, (s, e) in enumerate(blocks):
        log.info(f"    Bloco {i+1}: {s/sr:.1f}s - {e/sr:.1f}s ({(e-s)/sr:.1f}s)")

    return blocks


def detect_fixed_blocks(audio: np.ndarray, sr: int,
                        rules: CompRules) -> List[Tuple[int, int]]:
    """Simple fixed-interval blocks."""
    seg_samples = int(sr * rules.fixed_segment_ms / 1000)
    blocks = []
    for start in range(0, len(audio), seg_samples):
        end = min(start + seg_samples, len(audio))
        if end - start >= int(sr * rules.min_segment_ms / 1000):
            blocks.append((start, end))
        elif blocks:
            blocks[-1] = (blocks[-1][0], end)
    return blocks


def detect_custom_blocks(audio: np.ndarray, sr: int,
                         rules: CompRules) -> List[Tuple[int, int]]:
    """Use user-defined sections from Structure tab as blocks."""
    blocks = []
    audio_len = len(audio)
    for sec in rules.custom_sections:
        start = int(float(sec["start_s"]) * sr)
        end = int(float(sec["end_s"]) * sr)
        start = max(0, min(start, audio_len))
        end = max(start, min(end, audio_len))
        if end - start >= int(sr * 0.5):
            blocks.append((start, end))

    if not blocks:
        log.warning("  Custom sections empty/invalid, falling back to musical")
        return detect_musical_blocks(audio, sr, rules)

    log.info(f"  {len(blocks)} blocos da Estrutura")
    for i, (s, e) in enumerate(blocks):
        name = rules.custom_sections[i].get("name", f"Secao {i+1}") if i < len(rules.custom_sections) else f"Secao {i+1}"
        log.info(f"    {name}: {s/sr:.1f}s - {e/sr:.1f}s ({(e-s)/sr:.1f}s)")
    return blocks


def detect_blocks(audio: np.ndarray, sr: int,
                  rules: CompRules) -> List[Tuple[int, int]]:
    """Route to the right segmentation method."""
    if rules.custom_sections:
        return detect_custom_blocks(audio, sr, rules)
    elif rules.segment_method == "fixed":
        return detect_fixed_blocks(audio, sr, rules)
    else:
        return detect_musical_blocks(audio, sr, rules)

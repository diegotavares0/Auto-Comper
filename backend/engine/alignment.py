"""
Take alignment — cross-correlation based global time shift.
No time-warping to preserve audio integrity.
"""

import logging
from typing import List

import numpy as np
from scipy.signal import correlate

log = logging.getLogger("comper")


def align_takes_xcorr(takes: List[np.ndarray], sr: int,
                      max_shift_ms: float,
                      reference_idx: int = 0) -> List[np.ndarray]:
    """
    Align takes using cross-correlation (global time shift only).

    Parameters
    ----------
    takes : list of np.ndarray
        Audio takes to align.
    sr : int
        Sample rate.
    max_shift_ms : float
        Maximum allowed time shift in milliseconds.
    reference_idx : int
        Index of the take to use as alignment reference (default 0).
        Should be the best-ranked take for optimal results.
    """
    if len(takes) < 2:
        return takes

    # Clamp reference_idx to valid range
    reference_idx = max(0, min(reference_idx, len(takes) - 1))

    log.info(f"Alinhando takes por cross-correlacao (ref=Take {reference_idx + 1})...")
    reference = takes[reference_idx]
    max_shift = int(sr * max_shift_ms / 1000)

    # Use first 10 seconds for correlation (sufficient for timing detection)
    ref_chunk_len = min(len(reference), sr * 10)
    ref_chunk = reference[:ref_chunk_len]

    aligned = [None] * len(takes)
    aligned[reference_idx] = reference  # reference stays unchanged

    for i, take in enumerate(takes):
        if i == reference_idx:
            continue

        chunk_len = min(ref_chunk_len, len(take))
        take_chunk = take[:chunk_len]

        correlation = correlate(ref_chunk[:chunk_len], take_chunk, mode="full")
        mid = len(correlation) // 2
        search_start = max(0, mid - max_shift)
        search_end = min(len(correlation), mid + max_shift)
        search = correlation[search_start:search_end]
        shift = np.argmax(search) - (mid - search_start)

        if abs(shift) > 0:
            if shift > 0:
                aligned_take = np.pad(take, (shift, 0))[:len(take)]
            else:
                aligned_take = take[-shift:]
                aligned_take = np.pad(aligned_take, (0, -shift))
            log.info(f"  Take {i+1}: shift = {shift} samples ({shift/sr*1000:.1f}ms)")
        else:
            aligned_take = take
            log.info(f"  Take {i+1}: ja alinhada")

        aligned[i] = aligned_take

    # Trim all to same length
    min_len = min(len(t) for t in aligned)
    aligned = [t[:min_len] for t in aligned]

    return aligned

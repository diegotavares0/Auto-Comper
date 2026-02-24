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
                      max_shift_ms: float) -> List[np.ndarray]:
    """
    Align takes using cross-correlation (global time shift only).
    Uses first take as reference.
    """
    if len(takes) < 2:
        return takes

    log.info("Alinhando takes por cross-correlacao...")
    reference = takes[0]
    max_shift = int(sr * max_shift_ms / 1000)
    aligned = [reference]

    for i, take in enumerate(takes[1:], 1):
        # Use first 10 seconds for correlation
        chunk_len = min(len(reference), len(take), sr * 10)
        ref_chunk = reference[:chunk_len]
        take_chunk = take[:chunk_len]

        correlation = correlate(ref_chunk, take_chunk, mode="full")
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

        aligned.append(aligned_take)

    # Trim all to same length
    min_len = min(len(t) for t in aligned)
    aligned = [t[:min_len] for t in aligned]

    return aligned

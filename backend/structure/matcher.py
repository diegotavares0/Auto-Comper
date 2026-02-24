"""
Structure matcher — propagate detected structure from the reference take
to all other takes, even if they differ in length or timing.

Uses chroma-based cross-similarity to find where each section boundary
in the reference falls in each other take.
"""

import logging
from typing import List, Dict, Optional, Callable, Tuple

import numpy as np
import librosa

log = logging.getLogger("comper")


def match_structure_to_takes(
    reference_sections: List[Dict],
    reference_audio: np.ndarray,
    takes: List[np.ndarray],
    sr: int,
    progress_cb: Optional[Callable] = None,
) -> List[List[Dict]]:
    """
    For each take, find the time positions that correspond to
    each section boundary in the reference.

    Args:
        reference_sections: sections from analyze_structure()
        reference_audio: the reference take audio
        takes: list of all take audio arrays (including reference)
        sr: sample rate
        progress_cb: (pct, msg) callback

    Returns:
        List of section lists, one per take.
        Each section has: name, label, group, start_s, end_s, covered (bool).
        "covered=False" means the take doesn't contain this section.
    """
    def progress(pct, msg):
        if progress_cb:
            progress_cb(pct, msg)

    if not reference_sections:
        return [[] for _ in takes]

    hop_length = 4096
    n_takes = len(takes)

    # Extract chroma for reference
    progress(0, "Extraindo chroma da referencia...")
    ref_chroma = librosa.feature.chroma_cqt(
        y=reference_audio, sr=sr, hop_length=hop_length, n_chroma=12,
    )

    # Extract boundary frames from reference sections
    ref_boundary_times = []
    for sec in reference_sections:
        ref_boundary_times.append(sec["start_s"])
    ref_boundary_times.append(reference_sections[-1]["end_s"])

    ref_boundary_frames = [
        librosa.time_to_frames(t, sr=sr, hop_length=hop_length)
        for t in ref_boundary_times
    ]

    # For each take, match boundaries
    all_take_sections = []

    for take_idx, take in enumerate(takes):
        pct_base = int(10 + 80 * take_idx / n_takes)
        progress(pct_base, f"Mapeando estrutura no take {take_idx + 1}/{n_takes}...")

        take_duration = len(take) / sr

        # Check if this take is the reference (by length match)
        if len(take) == len(reference_audio) and np.array_equal(take[:1000], reference_audio[:1000]):
            # Same take — copy sections directly
            take_sections = []
            for sec in reference_sections:
                take_sections.append({
                    **sec,
                    "covered": True,
                })
            all_take_sections.append(take_sections)
            continue

        # Extract chroma for this take
        take_chroma = librosa.feature.chroma_cqt(
            y=take, sr=sr, hop_length=hop_length, n_chroma=12,
        )

        # Find matching boundaries via cross-similarity
        matched_boundaries = _match_boundaries(
            ref_chroma, take_chroma,
            ref_boundary_frames,
            sr, hop_length, take_duration,
        )

        # Build sections for this take
        take_sections = _build_take_sections(
            reference_sections, matched_boundaries, take_duration,
        )
        all_take_sections.append(take_sections)

        log.info(f"  Take {take_idx + 1}: "
                 f"{sum(1 for s in take_sections if s['covered'])}"
                 f"/{len(take_sections)} secoes cobertas")

    progress(95, "Mapeamento de estrutura concluido!")
    return all_take_sections


def _match_boundaries(
    ref_chroma: np.ndarray,
    take_chroma: np.ndarray,
    ref_boundary_frames: List[int],
    sr: int,
    hop_length: int,
    take_duration: float,
) -> List[Optional[float]]:
    """
    For each boundary frame in the reference, find the best matching
    time position in the target take using local chroma cross-correlation.

    Returns list of boundary times (seconds) in the take, or None if
    no good match found.
    """
    ref_n = ref_chroma.shape[1]
    take_n = take_chroma.shape[1]

    if take_n < 2 or ref_n < 2:
        return [None] * len(ref_boundary_frames)

    # Normalize chromas
    ref_norm = ref_chroma / (np.linalg.norm(ref_chroma, axis=0, keepdims=True) + 1e-10)
    take_norm = take_chroma / (np.linalg.norm(take_chroma, axis=0, keepdims=True) + 1e-10)

    # Cross-similarity matrix (ref_n x take_n)
    cross_sim = ref_norm.T @ take_norm

    matched_times = []

    for ref_frame in ref_boundary_frames:
        ref_frame = min(ref_frame, ref_n - 1)

        # Context window around the boundary in the reference (~3 seconds each side)
        context_frames = max(1, int(3.0 * sr / hop_length))
        ctx_start = max(0, ref_frame - context_frames)
        ctx_end = min(ref_n, ref_frame + context_frames)

        # Average similarity of this context region against every take frame
        if ctx_end > ctx_start:
            context_sim = np.mean(cross_sim[ctx_start:ctx_end, :], axis=0)
        else:
            matched_times.append(None)
            continue

        # Expected position (proportional time mapping as prior)
        ref_time = librosa.frames_to_time(ref_frame, sr=sr, hop_length=hop_length)
        ref_duration = librosa.frames_to_time(ref_n, sr=sr, hop_length=hop_length)
        expected_ratio = ref_time / max(ref_duration, 0.01)
        expected_frame = int(expected_ratio * take_n)

        # Search window: ±30% of take length around expected position
        search_width = max(10, int(take_n * 0.3))
        search_start = max(0, expected_frame - search_width)
        search_end = min(take_n, expected_frame + search_width)

        if search_end <= search_start:
            matched_times.append(None)
            continue

        # Find best match within search window
        search_region = context_sim[search_start:search_end]
        best_local = np.argmax(search_region)
        best_frame = search_start + best_local
        best_sim = search_region[best_local]

        # Quality check: reject if similarity is too low
        if best_sim < 0.4:
            matched_times.append(None)
            continue

        # Convert frame to time
        match_time = librosa.frames_to_time(best_frame, sr=sr, hop_length=hop_length)
        match_time = min(match_time, take_duration)
        matched_times.append(round(float(match_time), 2))

    return matched_times


def _build_take_sections(
    reference_sections: List[Dict],
    matched_boundaries: List[Optional[float]],
    take_duration: float,
) -> List[Dict]:
    """
    Build section list for a take based on matched boundary positions.
    Sections where boundaries couldn't be found are marked covered=False.
    """
    n_sections = len(reference_sections)
    n_boundaries = len(matched_boundaries)  # should be n_sections + 1

    sections = []

    for i, ref_sec in enumerate(reference_sections):
        start_time = matched_boundaries[i] if i < n_boundaries else None
        end_time = matched_boundaries[i + 1] if (i + 1) < n_boundaries else None

        # Check if this section is covered
        covered = (start_time is not None and end_time is not None
                   and end_time > start_time)

        # Also check if section falls within the take's duration
        if covered and start_time > take_duration * 0.95:
            covered = False

        if covered:
            # Clamp to take boundaries
            start_time = max(0, start_time)
            end_time = min(take_duration, end_time)

            # Minimum duration check (~1 second)
            if end_time - start_time < 0.5:
                covered = False

        sections.append({
            "name": ref_sec["name"],
            "label": ref_sec["label"],
            "group": ref_sec["group"],
            "start_s": round(start_time, 2) if start_time is not None else 0.0,
            "end_s": round(end_time, 2) if end_time is not None else 0.0,
            "covered": covered,
            "confidence": ref_sec.get("confidence", 0.5) if covered else 0.0,
        })

    return sections

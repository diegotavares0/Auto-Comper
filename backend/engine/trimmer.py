"""
Auto-trimmer — removes silence from the beginning and end of takes.

Recording sessions produce takes with variable amounts of leading and
trailing silence (pre-count, room noise, trailing reverb).  Trimming
this silence *before* alignment makes cross-correlation more reliable
and the overall comp tighter.

Strategy:
  1. Compute short-time RMS energy in dB
  2. Find first/last frame above threshold
  3. Add pre-roll and post-roll to preserve attack/release
  4. Return trimmed audio + metadata
"""

import logging
from typing import List, Tuple, Optional, Callable, Dict

import numpy as np

log = logging.getLogger("comper")


def compute_rms_db(audio: np.ndarray, frame_size: int = 2048,
                    hop_size: int = 512) -> np.ndarray:
    """
    Compute short-time RMS energy in dB.

    Returns array of dB values (one per frame).
    Silent frames return -inf which we clamp to -100 dB.
    """
    n_frames = 1 + (len(audio) - frame_size) // hop_size
    rms = np.zeros(n_frames)

    for i in range(n_frames):
        start = i * hop_size
        frame = audio[start:start + frame_size]
        energy = np.sqrt(np.mean(frame ** 2))
        rms[i] = energy

    # Convert to dB (avoid log(0))
    rms_db = 20 * np.log10(np.maximum(rms, 1e-10))
    return rms_db


def find_trim_points(rms_db: np.ndarray, threshold_db: float,
                      min_consecutive: int = 3) -> Tuple[int, int]:
    """
    Find the first and last frames where audio exceeds the threshold.

    Uses min_consecutive: the music onset must persist for at least N
    consecutive frames to avoid triggering on random clicks/pops.

    Returns (start_frame, end_frame) — the bounds of the "music" region.
    """
    above = rms_db > threshold_db
    n = len(above)

    # Find first sustained region above threshold
    start_frame = 0
    for i in range(n - min_consecutive + 1):
        if all(above[i:i + min_consecutive]):
            start_frame = i
            break

    # Find last sustained region above threshold (scan from end)
    end_frame = n - 1
    for i in range(n - 1, min_consecutive - 2, -1):
        check_start = max(0, i - min_consecutive + 1)
        if all(above[check_start:i + 1]):
            end_frame = i
            break

    return start_frame, end_frame


def trim_audio(
    audio: np.ndarray,
    sr: int,
    threshold_db: float = -45.0,
    pre_roll_ms: float = 50.0,
    post_roll_ms: float = 100.0,
    min_music_duration_s: float = 1.0,
) -> Tuple[np.ndarray, Dict]:
    """
    Trim silence from beginning and end of audio.

    Parameters
    ----------
    audio : np.ndarray
        Input audio.
    sr : int
        Sample rate.
    threshold_db : float
        RMS threshold in dB below which audio is considered silence.
    pre_roll_ms : float
        Milliseconds of audio to keep before the first detected sound.
        Preserves the attack transient.
    post_roll_ms : float
        Milliseconds of audio to keep after the last detected sound.
        Preserves reverb tail / natural decay.
    min_music_duration_s : float
        Minimum music duration in seconds. If the detected music region
        is shorter than this, skip trimming (probably a false detection).

    Returns
    -------
    (trimmed_audio, info_dict)
        info_dict contains: trimmed_start_s, trimmed_end_s, original_duration_s,
        trimmed_duration_s, removed_start_s, removed_end_s
    """
    original_duration = len(audio) / sr
    hop_size = 512

    rms_db = compute_rms_db(audio, frame_size=2048, hop_size=hop_size)
    start_frame, end_frame = find_trim_points(rms_db, threshold_db)

    # Convert frames to samples
    start_sample = start_frame * hop_size
    end_sample = min((end_frame + 1) * hop_size + 2048, len(audio))

    # Apply pre/post-roll
    pre_roll_samples = int(pre_roll_ms / 1000 * sr)
    post_roll_samples = int(post_roll_ms / 1000 * sr)

    trim_start = max(0, start_sample - pre_roll_samples)
    trim_end = min(len(audio), end_sample + post_roll_samples)

    music_duration = (trim_end - trim_start) / sr

    # Safety: don't trim if the music region is too short
    if music_duration < min_music_duration_s:
        log.info(f"  Music region too short ({music_duration:.2f}s), "
                 f"skipping trim")
        return audio, {
            "trimmed": False,
            "original_duration_s": round(original_duration, 2),
            "trimmed_duration_s": round(original_duration, 2),
            "removed_start_s": 0.0,
            "removed_end_s": 0.0,
        }

    trimmed = audio[trim_start:trim_end]

    removed_start = trim_start / sr
    removed_end = (len(audio) - trim_end) / sr

    info = {
        "trimmed": True,
        "original_duration_s": round(original_duration, 2),
        "trimmed_duration_s": round(len(trimmed) / sr, 2),
        "removed_start_s": round(removed_start, 2),
        "removed_end_s": round(removed_end, 2),
    }

    return trimmed, info


def trim_takes(
    takes: List[np.ndarray],
    sr: int,
    threshold_db: float = -45.0,
    pre_roll_ms: float = 50.0,
    post_roll_ms: float = 100.0,
    min_music_duration_s: float = 1.0,
    progress_cb: Optional[Callable] = None,
) -> Tuple[List[np.ndarray], List[Dict]]:
    """
    Auto-trim all takes, removing leading/trailing silence.

    Parameters
    ----------
    takes : List[np.ndarray]
        Audio takes.
    sr : int
        Sample rate.
    threshold_db : float
        Silence detection threshold in dB.
    pre_roll_ms : float
        Pre-roll to preserve before music onset.
    post_roll_ms : float
        Post-roll to preserve after music end.
    min_music_duration_s : float
        Minimum viable music duration.
    progress_cb : callable, optional
        Progress callback(pct, msg).

    Returns
    -------
    (trimmed_takes, trim_infos)
        trim_infos is a list of per-take dicts with trim metadata.
    """
    n = len(takes)
    trimmed_takes = []
    trim_infos = []

    for i, audio in enumerate(takes):
        if progress_cb:
            pct = 3 + int(5 * i / n)
            progress_cb(pct, f"Trimando take {i + 1}/{n}...")

        trimmed, info = trim_audio(
            audio, sr,
            threshold_db=threshold_db,
            pre_roll_ms=pre_roll_ms,
            post_roll_ms=post_roll_ms,
            min_music_duration_s=min_music_duration_s,
        )

        trimmed_takes.append(trimmed)
        trim_infos.append(info)

        if info["trimmed"]:
            log.info(f"  Take {i + 1}: "
                     f"{info['original_duration_s']:.1f}s → "
                     f"{info['trimmed_duration_s']:.1f}s "
                     f"(removido: {info['removed_start_s']:.2f}s inicio, "
                     f"{info['removed_end_s']:.2f}s fim)")
        else:
            log.info(f"  Take {i + 1}: sem trim necessario")

    total_removed = sum(
        info["removed_start_s"] + info["removed_end_s"]
        for info in trim_infos
    )
    trimmed_count = sum(1 for info in trim_infos if info["trimmed"])

    if progress_cb:
        progress_cb(8, f"Trim: {trimmed_count}/{n} takes, "
                       f"{total_removed:.1f}s removidos")

    log.info(f"  Auto-trim: {trimmed_count}/{n} takes trimados, "
             f"{total_removed:.1f}s total removido")

    return trimmed_takes, trim_infos

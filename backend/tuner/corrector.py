"""
Pitch correction engine — segment-based pitch shifting with vibrato preservation.
"""

import logging
from typing import Callable, Optional

import numpy as np
from scipy import signal as scipy_signal
import pyrubberband as pyrb

from backend.config import TunerConfig
from backend.utils.musical_constants import (
    get_scale_midi_notes, nearest_scale_note, midi_to_hz,
)

log = logging.getLogger("comper")


def _compute_target_midi(midi: np.ndarray, voiced_mask: np.ndarray,
                         voiced_probs: np.ndarray,
                         config: TunerConfig,
                         effective_root: str,
                         effective_scale: str) -> np.ndarray:
    """
    Compute target MIDI pitch for each frame.
    Blends toward nearest scale note based on correction_amount.
    """
    scale_notes = get_scale_midi_notes(effective_root, effective_scale)
    amount = config.correction_amount / 100.0

    target = midi.copy()
    for i in range(len(midi)):
        if not voiced_mask[i] or midi[i] <= 0:
            continue

        # Find nearest scale note
        snap_to = nearest_scale_note(midi[i], scale_notes)

        # Blend: 0% = no change, 100% = full snap
        correction = (snap_to - midi[i]) * amount

        # Confidence-based attenuation for uncertain frames
        # (helps with mixed voice+guitar)
        if voiced_probs[i] < 0.85:
            correction *= voiced_probs[i] / 0.85

        target[i] = midi[i] + correction

    return target


def _extract_vibrato(midi: np.ndarray, voiced_mask: np.ndarray,
                     hop_ms: float, config: TunerConfig) -> np.ndarray:
    """
    Extract vibrato component from the pitch contour.
    Returns vibrato in MIDI units (add back to corrected pitch).
    """
    vibrato = np.zeros_like(midi)

    if not config.preserve_vibrato:
        return vibrato

    frame_rate = 1000.0 / hop_ms  # frames per second

    # Need at least ~0.5s of voiced audio for vibrato detection
    min_region_frames = int(frame_rate * 0.5)

    # Find contiguous voiced regions
    regions = []
    start = None
    for i in range(len(voiced_mask)):
        if voiced_mask[i] and midi[i] > 0:
            if start is None:
                start = i
        else:
            if start is not None:
                if i - start >= min_region_frames:
                    regions.append((start, i))
                start = None
    if start is not None and len(midi) - start >= min_region_frames:
        regions.append((start, len(midi)))

    if not regions:
        return vibrato

    # Bandpass filter for vibrato
    # Guitar vibrato is slower (2-6Hz) vs voice (4-8Hz), so use wide range
    nyquist = frame_rate / 2
    low = max(1.5, config.vibrato_threshold_hz - 2.5) / nyquist
    high = min(12.0, config.vibrato_threshold_hz + 8.0) / nyquist

    # Clamp to valid range
    low = max(0.01, min(low, 0.99))
    high = max(low + 0.01, min(high, 0.99))

    try:
        b, a = scipy_signal.butter(2, [low, high], btype='band')
    except Exception:
        log.warning("  Vibrato filter design failed, skipping preservation")
        return vibrato

    max_depth_midi = config.vibrato_max_depth_cents / 100.0  # cents → MIDI

    for start, end in regions:
        region_midi = midi[start:end].copy()

        # Local mean (100ms moving average)
        window = max(1, int(frame_rate * 0.1))
        kernel = np.ones(window) / window
        local_mean = np.convolve(region_midi, kernel, mode='same')

        # Deviation from local mean (in MIDI units)
        deviation = region_midi - local_mean

        # Apply bandpass to extract vibrato-rate oscillation
        if len(deviation) < 15:
            continue

        try:
            filtered = scipy_signal.filtfilt(b, a, deviation)
        except Exception:
            continue

        # Check vibrato depth (peak-to-peak)
        depth = filtered.max() - filtered.min()  # in MIDI units
        depth_cents = depth * 100

        if 2 < depth_cents < config.vibrato_max_depth_cents:
            # Valid vibrato — preserve it
            vibrato[start:end] = filtered

    return vibrato


def _apply_retune_speed(shift_midi: np.ndarray, voiced_mask: np.ndarray,
                        hop_ms: float, retune_speed: float) -> np.ndarray:
    """
    Smooth the pitch correction curve based on retune_speed.
    Speed=100: instant (no smoothing). Speed=0: ~500ms transition.
    """
    if retune_speed >= 99:
        return shift_midi

    frame_rate = 1000.0 / hop_ms
    max_smooth_ms = 500  # slowest retune time
    smooth_ms = max_smooth_ms * (1 - retune_speed / 100)
    smooth_frames = max(1, int(smooth_ms / hop_ms))

    if smooth_frames <= 1:
        return shift_midi

    smoothed = shift_midi.copy()
    kernel = np.ones(smooth_frames) / smooth_frames

    # Only smooth voiced regions to avoid bleeding into silence
    regions = []
    start = None
    for i in range(len(voiced_mask)):
        if voiced_mask[i]:
            if start is None:
                start = i
        else:
            if start is not None:
                regions.append((start, i))
                start = None
    if start is not None:
        regions.append((start, len(voiced_mask)))

    for s, e in regions:
        if e - s > smooth_frames:
            smoothed[s:e] = np.convolve(shift_midi[s:e], kernel, mode='same')

    return smoothed


def _group_segments(shift_semitones: np.ndarray, voiced_mask: np.ndarray,
                    hop_length: int, sr: int,
                    threshold_cents: float = 40.0) -> list:
    """
    Group consecutive frames with similar pitch shift into segments.
    Returns list of {start_sample, end_sample, shift_semitones}.
    """
    threshold_st = threshold_cents / 100.0  # cents to semitones
    segments = []
    current_start = None
    current_shift = 0
    shifts_in_group = []

    for i in range(len(shift_semitones)):
        if not voiced_mask[i] or abs(shift_semitones[i]) < 0.001:
            # Unvoiced or no correction needed
            if current_start is not None:
                avg_shift = np.mean(shifts_in_group)
                segments.append({
                    "start_frame": current_start,
                    "end_frame": i,
                    "start_sample": current_start * hop_length,
                    "end_sample": i * hop_length,
                    "shift_semitones": float(avg_shift),
                })
                current_start = None
                shifts_in_group = []
            continue

        if current_start is None:
            current_start = i
            current_shift = shift_semitones[i]
            shifts_in_group = [shift_semitones[i]]
        elif abs(shift_semitones[i] - current_shift) > threshold_st:
            # Shift changed significantly → close current segment
            avg_shift = np.mean(shifts_in_group)
            segments.append({
                "start_frame": current_start,
                "end_frame": i,
                "start_sample": current_start * hop_length,
                "end_sample": i * hop_length,
                "shift_semitones": float(avg_shift),
            })
            current_start = i
            current_shift = shift_semitones[i]
            shifts_in_group = [shift_semitones[i]]
        else:
            shifts_in_group.append(shift_semitones[i])

    # Close last segment
    if current_start is not None:
        avg_shift = np.mean(shifts_in_group)
        segments.append({
            "start_frame": current_start,
            "end_frame": len(shift_semitones),
            "start_sample": current_start * hop_length,
            "end_sample": min(len(shift_semitones) * hop_length,
                              len(shift_semitones) * hop_length),
            "shift_semitones": float(avg_shift),
        })

    return segments


def correct_pitch(audio: np.ndarray, sr: int, analysis: dict,
                  config: TunerConfig,
                  progress_cb: Optional[Callable] = None) -> tuple:
    """
    Apply pitch correction to audio based on analysis.
    Returns (corrected_audio, correction_stats).
    progress_cb(pct, msg) reports 35-80%.
    """
    midi = analysis["midi"]
    voiced_mask = analysis["voiced_flag"]
    voiced_probs = analysis["voiced_probs"]
    hop_length = analysis["hop_length"]

    if progress_cb:
        progress_cb(35, "Calculando correcoes de pitch...")

    # Step 1: Compute target pitch per frame
    target_midi = _compute_target_midi(
        midi, voiced_mask, voiced_probs, config,
        analysis["effective_root"], analysis["effective_scale"],
    )

    # Step 2: Extract vibrato component
    vibrato = _extract_vibrato(midi, voiced_mask, config.hop_ms, config)

    # Step 3: Compute shift = target - original + vibrato
    shift_midi = np.zeros_like(midi)
    for i in range(len(midi)):
        if voiced_mask[i] and midi[i] > 0:
            shift_midi[i] = (target_midi[i] - midi[i]) + vibrato[i]

    # Step 4: Apply retune speed smoothing
    shift_midi = _apply_retune_speed(
        shift_midi, voiced_mask, config.hop_ms, config.retune_speed
    )

    # Step 5: Group into segments
    segments = _group_segments(shift_midi, voiced_mask, hop_length, sr)

    if not segments:
        log.info("  No pitch corrections needed")
        return audio.copy(), _build_stats(midi, midi, voiced_mask, 0)

    if progress_cb:
        progress_cb(40, f"Corrigindo pitch... ({len(segments)} segmentos)")

    # Step 6: Apply pitch shift per segment
    corrected = audio.copy()
    n_segments = len(segments)
    crossfade_samples = int(sr * 0.100)  # 100ms crossfade — prevents "air blocking" artifacts

    corrections_cents = []

    for idx, seg in enumerate(segments):
        start = seg["start_sample"]
        end = min(seg["end_sample"], len(audio))
        shift_st = seg["shift_semitones"]

        if abs(shift_st) < 0.005 or end <= start:
            continue

        # Add margin for cleaner pitch shift — wider margin gives pyrubberband
        # more context for its overlap-add windowing
        margin = int(sr * 0.300)  # 300ms margin (wider = fewer edge artifacts)
        seg_start = max(0, start - margin)
        seg_end = min(len(audio), end + margin)

        segment_audio = audio[seg_start:seg_end]

        try:
            # pyrubberband uses Rubber Band Library — time-domain pitch shift
            # -F flag = formant preservation (keeps voice/instrument natural,
            # prevents "nasal" or "airy" artifacts when pitch changes)
            shifted = pyrb.pitch_shift(
                segment_audio, sr=sr, n_steps=shift_st,
                rbargs=["--formant"],
            )

            # Extract the core region (without margins)
            core_start = start - seg_start
            core_end = core_start + (end - start)
            shifted_core = shifted[core_start:core_end]

            # Apply with crossfade
            actual_len = min(len(shifted_core), end - start)
            if actual_len <= 0:
                continue

            # Equal-power crossfade in (sin/cos curves — constant energy,
            # prevents the 3dB dip that linear crossfades cause)
            fade_in_len = min(crossfade_samples, actual_len // 2)
            if fade_in_len > 0:
                t = np.linspace(0, np.pi / 2, fade_in_len)
                fade_in_new = np.sin(t)      # 0 → 1 (shifted audio fades in)
                fade_in_old = np.cos(t)      # 1 → 0 (original audio fades out)
                shifted_core[:fade_in_len] = (
                    corrected[start:start + fade_in_len] * fade_in_old +
                    shifted_core[:fade_in_len] * fade_in_new
                )

            # Equal-power crossfade out
            fade_out_len = min(crossfade_samples, actual_len // 2)
            if fade_out_len > 0:
                t = np.linspace(0, np.pi / 2, fade_out_len)
                fade_out_new = np.cos(t)     # 1 → 0 (shifted audio fades out)
                fade_out_old = np.sin(t)     # 0 → 1 (original audio fades back in)
                end_pos = start + actual_len
                shifted_core[-fade_out_len:] = (
                    corrected[end_pos - fade_out_len:end_pos] * fade_out_old +
                    shifted_core[-fade_out_len:] * fade_out_new
                )

            corrected[start:start + actual_len] = shifted_core[:actual_len]
            corrections_cents.append(abs(shift_st * 100))

        except Exception as e:
            log.warning(f"  Segment {idx} pitch shift failed: {e}")
            continue

        # Report progress (40% → 78%)
        if progress_cb and n_segments > 1:
            pct = 40 + int(38 * (idx + 1) / n_segments)
            progress_cb(pct, f"Corrigindo pitch... "
                             f"(segmento {idx + 1}/{n_segments})")

    if progress_cb:
        progress_cb(78, "Aplicando crossfades...")

    # Build corrected pitch curve for visualization
    corrected_midi = midi.copy()
    for i in range(len(midi)):
        if voiced_mask[i] and midi[i] > 0:
            corrected_midi[i] = midi[i] + shift_midi[i]

    stats = _build_stats(midi, corrected_midi, voiced_mask, len(corrections_cents))
    stats["avg_correction_cents"] = round(
        np.mean(corrections_cents) if corrections_cents else 0, 1
    )
    stats["max_correction_cents"] = round(
        max(corrections_cents) if corrections_cents else 0, 1
    )

    # Detect mixed signal warning
    if voiced_mask.any():
        mean_prob = np.mean(voiced_probs[voiced_mask])
        stats["mixed_signal_warning"] = bool(mean_prob < 0.7)
    else:
        stats["mixed_signal_warning"] = False

    if progress_cb:
        progress_cb(80, f"Correcao concluida ({len(corrections_cents)} segmentos)")

    log.info(f"  Correction: {len(corrections_cents)} segments, "
             f"avg={stats['avg_correction_cents']:.1f} cents, "
             f"max={stats['max_correction_cents']:.1f} cents")

    return corrected, stats


def _build_stats(original_midi, corrected_midi, voiced_mask, n_corrections):
    """Build basic correction stats."""
    return {
        "corrections_applied": n_corrections,
        "avg_correction_cents": 0,
        "max_correction_cents": 0,
        "mixed_signal_warning": False,
    }

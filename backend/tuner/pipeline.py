"""
Tuner pipeline orchestrator — analyze, correct, normalize.
"""

import logging
from typing import Tuple, Optional, Callable

import numpy as np

from backend.config import TunerConfig
from backend.tuner.analyzer import analyze_pitch, _downsample_pitch_curve
from backend.tuner.corrector import correct_pitch
from backend.engine.assembly import normalize_lufs, peak_limit

log = logging.getLogger("comper")


def run_tuner(audio: np.ndarray, sr: int, config: TunerConfig,
              progress_callback: Optional[Callable] = None) -> Tuple[np.ndarray, dict]:
    """
    Full tuning pipeline: analyze → correct → normalize → output.
    Returns: (tuned_audio, report_dict)
    """
    def progress(pct, msg):
        if progress_callback:
            progress_callback(pct, msg)
        log.info(f"[{pct}%] {msg}")

    duration = len(audio) / sr

    # Validate input
    if duration < 1.0:
        raise ValueError("Audio muito curto para tuning (minimo 1s)")

    rms = np.sqrt(np.mean(audio ** 2))
    if rms < 1e-6:
        raise ValueError("Nenhum sinal de audio detectado (silencio)")

    progress(2, "Carregando audio...")
    progress(5, f"Audio carregado: {duration:.1f}s")

    # Phase 1: Pitch Analysis (5-33%)
    analysis = analyze_pitch(audio, sr, config, progress_cb=progress)

    # Phase 2: Pitch Correction (35-80%)
    if config.correction_amount < 0.5:
        # No correction needed, pass through
        progress(80, "Correcao desativada (0%)")
        tuned_audio = audio.copy()
        correction_stats = {
            "corrections_applied": 0,
            "avg_correction_cents": 0,
            "max_correction_cents": 0,
            "mixed_signal_warning": False,
        }
    else:
        tuned_audio, correction_stats = correct_pitch(
            audio, sr, analysis, config, progress_cb=progress,
        )

    # Phase 3: Normalize (80-90%)
    if config.normalize_output:
        progress(85, "Normalizando volume...")
        tuned_audio = normalize_lufs(tuned_audio, sr, config.target_lufs)

    # Phase 4: Peak limiter (90-95%)
    progress(90, "Aplicando limiter de pico...")
    tuned_audio = peak_limit(tuned_audio)

    # Phase 5: Build report (95-100%)
    progress(95, "Gerando relatorio...")

    # Build corrected pitch curve for visualization
    # Re-analyze the corrected audio for the "after" pitch curve
    corrected_pitch_curve = _build_corrected_curve(
        analysis, config,
    )

    report = {
        "version": "1.0",
        "duration_s": round(duration, 2),
        "instrument_mode": config.instrument_mode,
        "detected_key": f"{analysis['estimated_root']} {analysis['estimated_scale']}",
        "effective_key": f"{analysis['effective_root']} {analysis['effective_scale']}",
        "key_confidence": analysis["key_confidence"],
        "correction_amount": config.correction_amount,
        "retune_speed": config.retune_speed,
        "preserve_vibrato": config.preserve_vibrato,
        "pitch_stats": analysis["pitch_stats"],
        "corrections_applied": correction_stats["corrections_applied"],
        "avg_correction_cents": correction_stats["avg_correction_cents"],
        "max_correction_cents": correction_stats["max_correction_cents"],
        "mixed_signal_warning": correction_stats["mixed_signal_warning"],
        "pitch_curve_original": analysis["pitch_curve_original"],
        "pitch_curve_corrected": corrected_pitch_curve,
    }

    progress(100, f"Tuning finalizado! Tom: {report['effective_key']}, "
                  f"{correction_stats['corrections_applied']} correcoes")

    return tuned_audio, report


def _build_corrected_curve(analysis: dict, config: TunerConfig) -> list:
    """
    Build the corrected pitch curve for visualization.
    Instead of re-analyzing the audio (slow), we compute it from
    the original MIDI + target MIDI.
    """
    from backend.utils.musical_constants import (
        get_scale_midi_notes, nearest_scale_note, midi_to_hz as m2h,
    )

    midi = analysis["midi"]
    voiced_mask = analysis["voiced_flag"]
    voiced_probs = analysis["voiced_probs"]
    hop_length = analysis["hop_length"]
    sr = analysis["sr"]

    scale_notes = get_scale_midi_notes(
        analysis["effective_root"], analysis["effective_scale"]
    )
    amount = config.correction_amount / 100.0

    frame_rate = sr / hop_length
    step = max(1, int(frame_rate / 10.0))  # ~10 points/sec

    curve = []
    for i in range(0, len(midi), step):
        if not voiced_mask[i] or midi[i] <= 0:
            continue

        original_midi = midi[i]
        target = nearest_scale_note(original_midi, scale_notes)
        correction = (target - original_midi) * amount

        # Confidence attenuation
        if voiced_probs[i] < 0.85:
            correction *= voiced_probs[i] / 0.85

        corrected_midi = original_midi + correction
        corrected_hz = float(m2h(np.array([corrected_midi]))[0])

        t = i * hop_length / sr
        curve.append({"t": round(t, 3), "hz": round(corrected_hz, 1)})

    return curve

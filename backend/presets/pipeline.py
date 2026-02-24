"""
Tone preset pipeline — orchestrates analysis, matching, and neural refinement.

Two entry points:
1. create_preset() — analyze a reference clip and save as preset
2. apply_preset() — apply a saved preset to a guitar take
"""

import logging
from typing import Callable, Optional

import numpy as np

from backend.config import PresetConfig
from backend.presets.analyzer import analyze_reference
from backend.presets.processor import apply_tone_dsp
from backend.presets import manager, neural
from backend.engine.assembly import normalize_lufs, peak_limit

log = logging.getLogger("comper")


def create_preset(
    audio: np.ndarray,
    sr: int,
    name: str,
    reference_audio_path: Optional[str] = None,
    progress_callback: Optional[Callable] = None,
) -> dict:
    """
    Create a new tone preset from a reference audio clip.

    Parameters
    ----------
    audio : np.ndarray
        Reference audio (mono, resampled).
    sr : int
        Sample rate.
    name : str
        Display name for the preset.
    reference_audio_path : str, optional
        Path to the WAV file (saved alongside preset for later use).
    progress_callback : callable, optional
        Progress updates (pct, msg).

    Returns
    -------
    dict : Preset metadata.
    """
    if progress_callback:
        progress_callback(0, "Iniciando analise de referencia...")

    # Validate
    duration_s = len(audio) / sr
    if duration_s < 2.0:
        raise ValueError("Audio de referencia muito curto (minimo 2 segundos)")

    rms = np.sqrt(np.mean(audio ** 2))
    if rms < 1e-6:
        raise ValueError("Audio de referencia parece ser silencio")

    # Analyze reference
    profile = analyze_reference(
        audio, sr,
        progress_cb=progress_callback,
    )

    if progress_callback:
        progress_callback(60, "Salvando preset...")

    # Save
    meta = manager.save_preset(
        name=name,
        profile=profile,
        reference_audio_path=reference_audio_path,
    )

    if progress_callback:
        progress_callback(100, f"Preset '{name}' criado!")

    log.info(f"  Preset created: '{name}' (id={meta['id']})")

    return meta


def apply_preset(
    audio: np.ndarray,
    sr: int,
    config: PresetConfig,
    progress_callback: Optional[Callable] = None,
) -> tuple:
    """
    Apply a saved tone preset to a guitar take.

    Parameters
    ----------
    audio : np.ndarray
        Guitar take audio (mono, resampled).
    sr : int
        Sample rate.
    config : PresetConfig
        Configuration including preset_id, intensity, use_neural, etc.
    progress_callback : callable, optional
        Progress updates.

    Returns
    -------
    tuple : (processed_audio, report_dict)
    """
    if progress_callback:
        progress_callback(0, "Iniciando tone matching...")

    # Validate input
    duration_s = len(audio) / sr
    if duration_s < 1.0:
        raise ValueError("Audio muito curto para tone matching (minimo 1s)")

    rms = np.sqrt(np.mean(audio ** 2))
    if rms < 1e-6:
        raise ValueError("Audio parece ser silencio")

    # Load preset profile
    if progress_callback:
        progress_callback(5, "Carregando preset...")

    preset_meta = manager.get_preset(config.preset_id)
    if not preset_meta:
        raise ValueError(f"Preset nao encontrado: {config.preset_id}")

    profile = manager.load_profile(config.preset_id)
    if not profile:
        raise ValueError(f"Perfil do preset nao encontrado: {config.preset_id}")

    if progress_callback:
        progress_callback(10, "Aplicando DSP tone matching...")

    # ── Stage 1: DSP matching ──
    processed, dsp_stats = apply_tone_dsp(
        audio, sr, profile,
        intensity=config.intensity,
        max_boost_db=config.max_boost_db,
        max_cut_db=config.max_cut_db,
        dynamics_match=config.dynamics_match,
        transient_preserve=config.transient_preserve,
        n_fft=config.n_fft,
        hop_length=config.hop_length,
        lifter_order=config.spectral_smoothing,
        progress_cb=progress_callback,
    )

    # ── Stage 2: Neural refinement (optional) ──
    neural_used = False
    neural_available = neural.is_available()

    if config.use_neural and neural_available:
        ref_audio_path = manager.get_reference_audio_path(config.preset_id)
        if ref_audio_path:
            if progress_callback:
                progress_callback(76, "Carregando referencia para neural...")

            from backend.utils.audio_io import load_audio_file
            ref_audio = load_audio_file(ref_audio_path, sr)

            processed = neural.refine_neural(
                processed, sr,
                reference_audio=ref_audio,
                intensity=config.intensity,
                progress_cb=progress_callback,
            )
            neural_used = True
        else:
            log.warning(
                "  Neural requested but no reference audio saved with preset"
            )

    elif config.use_neural and not neural_available:
        log.warning("  Neural requested but PyTorch not installed")

    # ── Stage 3: Normalize + Peak Limit ──
    if config.normalize_output:
        if progress_callback:
            progress_callback(92, "Normalizando volume...")
        processed = normalize_lufs(processed, sr, config.target_lufs)

    if progress_callback:
        progress_callback(95, "Aplicando limiter de pico...")
    processed = peak_limit(processed)

    # ── Build report ──
    report = {
        "version": "1.0",
        "preset_id": config.preset_id,
        "preset_name": preset_meta.get("name", ""),
        "duration_s": round(duration_s, 2),
        "intensity": config.intensity,

        # DSP stats
        "dsp": dsp_stats,

        # Neural info
        "neural_used": neural_used,
        "neural_available": neural_available,

        # Preset timbre info (for display)
        "ref_timbre": preset_meta.get("timbre", {}),
        "ref_dynamics": preset_meta.get("dynamics", {}),

        # Spectral comparison (for visualization)
        "spectral_comparison": _build_spectral_comparison(
            audio, processed, sr, config.n_fft
        ),
    }

    if progress_callback:
        progress_callback(
            100,
            f"Tone matching finalizado! Preset: {preset_meta.get('name', '')}"
        )

    return processed, report


def _build_spectral_comparison(
    original: np.ndarray,
    processed: np.ndarray,
    sr: int,
    n_fft: int,
    n_bands: int = 32,
) -> dict:
    """
    Build a simplified spectral comparison for visualization.
    Returns frequency bands with before/after dB levels.
    """
    import librosa

    S_orig = np.abs(librosa.stft(original, n_fft=n_fft))
    S_proc = np.abs(librosa.stft(processed, n_fft=n_fft))

    avg_orig = np.mean(S_orig, axis=1)
    avg_proc = np.mean(S_proc, axis=1)

    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

    # Downsample to n_bands for visualization (log-spaced)
    freq_edges = np.logspace(
        np.log10(max(freqs[1], 20)),
        np.log10(min(freqs[-1], sr / 2)),
        n_bands + 1,
    )

    bands = []
    for i in range(n_bands):
        lo, hi = freq_edges[i], freq_edges[i + 1]
        mask = (freqs >= lo) & (freqs < hi)
        if not np.any(mask):
            continue

        orig_db = float(20 * np.log10(np.mean(avg_orig[mask]) + 1e-10))
        proc_db = float(20 * np.log10(np.mean(avg_proc[mask]) + 1e-10))

        bands.append({
            "freq_hz": round(float(np.sqrt(lo * hi)), 0),  # geometric mean
            "original_db": round(orig_db, 1),
            "processed_db": round(proc_db, 1),
            "diff_db": round(proc_db - orig_db, 1),
        })

    return {"bands": bands}

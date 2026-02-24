"""
DSP tone processor — apply a spectral profile to transform guitar tone.

Three stages:
1. Spectral envelope matching (FFT-based EQ transfer)
2. Dynamics matching (level + compression character)
3. Transient preservation (protect attack transients from over-processing)
"""

import logging
from typing import Callable, Optional

import numpy as np
import librosa
from scipy.signal import butter, sosfilt

from backend.presets.analyzer import analyze_input

log = logging.getLogger("comper")


def apply_tone_dsp(
    audio: np.ndarray,
    sr: int,
    ref_profile: dict,
    intensity: float = 80.0,
    max_boost_db: float = 12.0,
    max_cut_db: float = 18.0,
    dynamics_match: bool = True,
    transient_preserve: float = 0.7,
    n_fft: int = 4096,
    hop_length: int = 1024,
    lifter_order: int = 20,
    progress_cb: Optional[Callable] = None,
) -> tuple:
    """
    Apply spectral tone matching from a reference profile to input audio.

    Returns (processed_audio, stats_dict).
    """
    if progress_cb:
        progress_cb(35, "Analisando take de entrada...")

    # ── Step 1: Analyze input ──
    input_analysis = analyze_input(
        audio, sr,
        n_fft=n_fft,
        hop_length=hop_length,
        lifter_order=lifter_order,
    )

    if progress_cb:
        progress_cb(42, "Calculando curva de EQ...")

    # ── Step 2: Compute gain curve ──
    ref_env_db = np.array(ref_profile["envelope_db"])
    inp_env_db = input_analysis["envelope_db"]

    # Ensure same length
    min_len = min(len(ref_env_db), len(inp_env_db))
    ref_env_db = ref_env_db[:min_len]
    inp_env_db = inp_env_db[:min_len]

    # Gain in dB = target - input
    gain_db = ref_env_db - inp_env_db

    # Apply intensity scaling (0-100%)
    blend = intensity / 100.0
    gain_db = gain_db * blend

    # Clamp to safe limits
    gain_db = np.clip(gain_db, -max_cut_db, max_boost_db)

    # Convert dB gain to linear
    gain_linear = 10 ** (gain_db / 20.0)

    # Pad gain to full FFT size if needed
    full_bins = n_fft // 2 + 1
    if len(gain_linear) < full_bins:
        gain_linear = np.pad(
            gain_linear,
            (0, full_bins - len(gain_linear)),
            mode='edge',
        )
    gain_linear = gain_linear[:full_bins]

    if progress_cb:
        progress_cb(50, "Aplicando matching espectral...")

    # ── Step 3: Apply spectral matching via STFT ──
    S = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)

    # Transient detection for preservation
    if transient_preserve > 0:
        onset_env = librosa.onset.onset_strength(
            S=np.abs(S), sr=sr, hop_length=hop_length
        )
        # Normalize onset envelope to 0-1
        onset_max = onset_env.max()
        if onset_max > 0:
            onset_norm = onset_env / onset_max
        else:
            onset_norm = np.zeros_like(onset_env)

        # Create per-frame gain attenuation: reduce EQ near transients
        # transient_preserve=1.0 means full protection (no EQ at transients)
        # transient_preserve=0.0 means no protection
        transient_mask = 1.0 - transient_preserve * onset_norm
        transient_mask = np.clip(transient_mask, 0.3, 1.0)

        # Expand gain to 2D: (freq_bins, 1) * (1, frames)
        # But transient mask modulates how much gain is applied per frame
        gain_2d = (
            gain_linear[:, np.newaxis] ** transient_mask[np.newaxis, :]
        )
    else:
        gain_2d = gain_linear[:, np.newaxis]

    # Apply gain (magnitude only, preserve phase)
    S_matched = S * gain_2d

    # Reconstruct
    matched_audio = librosa.istft(S_matched, hop_length=hop_length, length=len(audio))

    if progress_cb:
        progress_cb(65, "Ajustando dinamica...")

    # ── Step 4: Dynamics matching ──
    if dynamics_match and "dynamics" in ref_profile:
        matched_audio = _match_dynamics(
            matched_audio, sr,
            ref_profile["dynamics"],
            input_analysis,
            blend,
        )

    if progress_cb:
        progress_cb(75, "Tone matching concluido")

    # ── Stats ──
    stats = {
        "gain_curve_db": {
            "min": round(float(np.min(gain_db)), 1),
            "max": round(float(np.max(gain_db)), 1),
            "mean": round(float(np.mean(gain_db)), 1),
        },
        "intensity_pct": intensity,
        "dynamics_matched": dynamics_match,
        "transient_preserve": transient_preserve,
    }

    log.info(
        f"  Tone DSP: gain range [{stats['gain_curve_db']['min']:+.1f}, "
        f"{stats['gain_curve_db']['max']:+.1f}] dB, "
        f"intensity={intensity}%"
    )

    return matched_audio, stats


def _match_dynamics(
    audio: np.ndarray,
    sr: int,
    ref_dynamics: dict,
    input_analysis: dict,
    blend: float,
) -> np.ndarray:
    """
    Match dynamic characteristics: level + compression.

    Two stages:
    1. Soft compression to match crest factor (transient behavior)
    2. Level matching to align RMS
    """
    ref_crest = ref_dynamics["crest_factor_db"]
    inp_crest = input_analysis["crest_factor_db"]

    # Only compress if input has MORE dynamics than reference
    # (i.e., reference sounds more compressed)
    if inp_crest > ref_crest + 1.0:
        # Need to reduce dynamics (compress)
        crest_diff = (inp_crest - ref_crest) * blend

        # Simple soft-knee compressor
        # Threshold: set so we catch the transients
        rms = np.sqrt(np.mean(audio ** 2))
        threshold_linear = rms * 1.5  # compress above 1.5x RMS

        # Ratio derived from crest difference
        # Each 3dB of crest reduction ≈ 2:1 ratio
        ratio = 1.0 + crest_diff / 3.0
        ratio = min(ratio, 6.0)  # cap at 6:1

        audio = _soft_compress(audio, sr, threshold_linear, ratio)
        log.info(
            f"  Dynamics: compressed {crest_diff:.1f}dB, ratio={ratio:.1f}:1"
        )

    # Level matching: align RMS to reference
    ref_rms_db = ref_dynamics["rms_mean_db"]
    inp_rms = np.sqrt(np.mean(audio ** 2))
    inp_rms_db = 20 * np.log10(inp_rms + 1e-10)

    level_diff_db = (ref_rms_db - inp_rms_db) * blend
    gain = 10 ** (level_diff_db / 20)
    audio = audio * gain

    return audio


def _soft_compress(
    audio: np.ndarray,
    sr: int,
    threshold: float,
    ratio: float,
    attack_ms: float = 10.0,
    release_ms: float = 80.0,
) -> np.ndarray:
    """
    Simple feed-forward soft-knee compressor.

    Uses envelope follower for smooth gain reduction.
    """
    # Envelope follower (rectified + smoothed)
    envelope = np.abs(audio)

    # Smooth with attack/release filter
    attack_coef = np.exp(-1.0 / (sr * attack_ms / 1000))
    release_coef = np.exp(-1.0 / (sr * release_ms / 1000))

    smooth_env = np.zeros_like(envelope)
    smooth_env[0] = envelope[0]
    for i in range(1, len(envelope)):
        if envelope[i] > smooth_env[i - 1]:
            coef = attack_coef
        else:
            coef = release_coef
        smooth_env[i] = coef * smooth_env[i - 1] + (1 - coef) * envelope[i]

    # Compute gain reduction
    gain = np.ones_like(smooth_env)
    above = smooth_env > threshold
    if np.any(above):
        # dB above threshold
        over_db = 20 * np.log10(smooth_env[above] / threshold)
        # Reduce by (1 - 1/ratio) of the overshoot
        reduction_db = over_db * (1 - 1.0 / ratio)
        gain[above] = 10 ** (-reduction_db / 20)

    return audio * gain

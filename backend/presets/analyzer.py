"""
Spectral profile analyzer — extract tonal signature from a reference audio clip.

Uses cepstral-smoothed spectral envelope extraction to capture the "tone shape"
of a guitar recording without encoding individual notes or melodies.
"""

import logging
from typing import Callable, Optional

import numpy as np
import librosa
from scipy.ndimage import gaussian_filter1d

log = logging.getLogger("comper")


def analyze_reference(
    audio: np.ndarray,
    sr: int,
    n_fft: int = 4096,
    hop_length: int = 1024,
    lifter_order: int = 20,
    progress_cb: Optional[Callable] = None,
) -> dict:
    """
    Extract a tonal profile from a reference audio clip.

    The profile captures:
    - Spectral envelope (smoothed via cepstral liftering)
    - Dynamic characteristics (RMS level, crest factor, dynamic range)
    - Timbral descriptors (spectral centroid, bandwidth, rolloff)

    Parameters
    ----------
    audio : np.ndarray
        Mono audio signal.
    sr : int
        Sample rate.
    n_fft : int
        FFT size — larger = finer frequency resolution.
    hop_length : int
        STFT hop size.
    lifter_order : int
        Number of cepstral coefficients to keep. Higher = smoother envelope.
        20-30 works well for guitar.

    Returns
    -------
    dict : Profile with all data needed to reconstruct the tone.
    """
    if progress_cb:
        progress_cb(5, "Analisando espectro de referencia...")

    duration_s = len(audio) / sr
    if duration_s < 2.0:
        raise ValueError("Referencia muito curta (minimo 2 segundos)")

    # ── Step 1: STFT ──
    S = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
    mag = np.abs(S)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

    if progress_cb:
        progress_cb(15, "Extraindo envelope espectral...")

    # ── Step 2: Average magnitude spectrum (time-averaged) ──
    # Use trimmed mean to reduce impact of transients and silence
    sorted_mag = np.sort(mag, axis=1)
    n_frames = mag.shape[1]
    trim_lo = int(n_frames * 0.1)  # ignore quietest 10%
    trim_hi = int(n_frames * 0.9)  # ignore loudest 10%
    if trim_hi > trim_lo:
        avg_mag = np.mean(sorted_mag[:, trim_lo:trim_hi], axis=1)
    else:
        avg_mag = np.mean(mag, axis=1)

    # ── Step 3: Cepstral smoothing (spectral envelope) ──
    # Convert to dB, apply inverse FFT, zero high quefrency, FFT back
    avg_db = 20 * np.log10(avg_mag + 1e-10)

    # Real cepstrum
    cepstrum = np.fft.irfft(avg_db)

    # Liftering: keep only first `lifter_order` coefficients
    # This removes fine harmonic structure, leaving just the envelope
    liftered = np.zeros_like(cepstrum)
    liftered[:lifter_order] = cepstrum[:lifter_order]
    if len(liftered) > lifter_order:
        liftered[-lifter_order + 1:] = cepstrum[-lifter_order + 1:]

    envelope_db = np.fft.rfft(liftered).real

    # Also store the raw (un-smoothed) for reference
    # Apply gentle gaussian smoothing as backup
    envelope_db_gauss = gaussian_filter1d(avg_db, sigma=8)

    if progress_cb:
        progress_cb(30, "Analisando dinamica...")

    # ── Step 4: Dynamics analysis ──
    # Frame-level RMS
    rms_frames = librosa.feature.rms(
        S=mag, frame_length=n_fft, hop_length=hop_length
    )[0]

    rms_db = 20 * np.log10(rms_frames + 1e-10)
    rms_mean_db = float(np.mean(rms_db))

    # Crest factor (peak-to-RMS ratio) — indicates compression character
    peak_level = float(np.max(np.abs(audio)))
    rms_level = float(np.sqrt(np.mean(audio ** 2)))
    crest_factor_db = 20 * np.log10(peak_level / (rms_level + 1e-10))

    # Dynamic range (90th percentile RMS - 10th percentile)
    dynamic_range_db = float(
        np.percentile(rms_db, 90) - np.percentile(rms_db, 10)
    )

    if progress_cb:
        progress_cb(45, "Calculando descritores de timbre...")

    # ── Step 5: Timbral descriptors ──
    spectral_centroid = float(np.mean(
        librosa.feature.spectral_centroid(S=mag, sr=sr)[0]
    ))
    spectral_bandwidth = float(np.mean(
        librosa.feature.spectral_bandwidth(S=mag, sr=sr)[0]
    ))
    spectral_rolloff = float(np.mean(
        librosa.feature.spectral_rolloff(S=mag, sr=sr, roll_percent=0.85)[0]
    ))

    # Brightness: ratio of energy above 3kHz to total
    freq_3k_bin = int(3000 * n_fft / sr)
    total_energy = float(np.sum(avg_mag ** 2))
    high_energy = float(np.sum(avg_mag[freq_3k_bin:] ** 2))
    brightness = high_energy / (total_energy + 1e-10)

    # Warmth: ratio of energy in 200-800Hz range
    freq_200_bin = int(200 * n_fft / sr)
    freq_800_bin = int(800 * n_fft / sr)
    warmth_energy = float(np.sum(avg_mag[freq_200_bin:freq_800_bin] ** 2))
    warmth = warmth_energy / (total_energy + 1e-10)

    if progress_cb:
        progress_cb(55, "Perfil de referencia completo")

    # ── Build profile ──
    profile = {
        "sr": sr,
        "n_fft": n_fft,
        "hop_length": hop_length,
        "lifter_order": lifter_order,
        "duration_s": round(duration_s, 2),

        # Spectral envelope (the core data for tone matching)
        "envelope_db": envelope_db.tolist(),
        "envelope_db_gauss": envelope_db_gauss.tolist(),
        "freqs": freqs.tolist(),

        # Dynamics
        "dynamics": {
            "rms_mean_db": round(rms_mean_db, 1),
            "crest_factor_db": round(crest_factor_db, 1),
            "dynamic_range_db": round(dynamic_range_db, 1),
            "peak_level": round(peak_level, 4),
            "rms_level": round(rms_level, 4),
        },

        # Timbral descriptors (for UI display + neural guidance)
        "timbre": {
            "spectral_centroid_hz": round(spectral_centroid, 1),
            "spectral_bandwidth_hz": round(spectral_bandwidth, 1),
            "spectral_rolloff_hz": round(spectral_rolloff, 1),
            "brightness": round(brightness, 3),
            "warmth": round(warmth, 3),
        },
    }

    log.info(
        f"  Reference profile: {duration_s:.1f}s, "
        f"centroid={spectral_centroid:.0f}Hz, "
        f"brightness={brightness:.2f}, warmth={warmth:.2f}, "
        f"RMS={rms_mean_db:.1f}dB, crest={crest_factor_db:.1f}dB"
    )

    return profile


def analyze_input(
    audio: np.ndarray,
    sr: int,
    n_fft: int = 4096,
    hop_length: int = 1024,
    lifter_order: int = 20,
) -> dict:
    """
    Quick spectral analysis of the input guitar take.
    Returns only the data needed for matching (envelope + dynamics).
    """
    S = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
    mag = np.abs(S)

    # Average spectrum with trimmed mean
    sorted_mag = np.sort(mag, axis=1)
    n_frames = mag.shape[1]
    trim_lo = int(n_frames * 0.1)
    trim_hi = int(n_frames * 0.9)
    if trim_hi > trim_lo:
        avg_mag = np.mean(sorted_mag[:, trim_lo:trim_hi], axis=1)
    else:
        avg_mag = np.mean(mag, axis=1)

    # Cepstral smoothing
    avg_db = 20 * np.log10(avg_mag + 1e-10)
    cepstrum = np.fft.irfft(avg_db)
    liftered = np.zeros_like(cepstrum)
    liftered[:lifter_order] = cepstrum[:lifter_order]
    if len(liftered) > lifter_order:
        liftered[-lifter_order + 1:] = cepstrum[-lifter_order + 1:]
    envelope_db = np.fft.rfft(liftered).real

    # Dynamics
    rms_level = float(np.sqrt(np.mean(audio ** 2)))
    peak_level = float(np.max(np.abs(audio)))
    crest_factor_db = 20 * np.log10(peak_level / (rms_level + 1e-10))
    rms_mean_db = float(20 * np.log10(rms_level + 1e-10))

    return {
        "envelope_db": envelope_db,
        "rms_mean_db": rms_mean_db,
        "rms_level": rms_level,
        "peak_level": peak_level,
        "crest_factor_db": crest_factor_db,
    }

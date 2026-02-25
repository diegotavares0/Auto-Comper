"""
Produtora — Take pre-filtering (outlier detection).

Analyzes all takes by BPM, pitch center, and RMS energy,
then excludes outliers that deviate too far from the group median.
Runs after auto-trim, before alignment.
"""

import logging
import math
from typing import List, Tuple, Optional, Callable, Dict

import numpy as np

from backend.engine.normalizer import estimate_tempo, estimate_pitch_center

log = logging.getLogger("comper")


# ── Helpers ──────────────────────────────────────────────


def compute_rms_energy(audio: np.ndarray) -> float:
    """Compute overall RMS energy of audio signal."""
    if len(audio) == 0:
        return 0.0
    rms = float(np.sqrt(np.mean(audio ** 2)))
    return rms


def fold_tempo(bpm: float, median_bpm: float) -> float:
    """
    Fold BPM to same octave as median to avoid double/half-time
    false positives from librosa.beat.beat_track().

    Example: if median is 80 BPM and a take is detected as 160 BPM,
    fold it back to 80 BPM.
    """
    if bpm <= 0 or median_bpm <= 0:
        return bpm

    ratio = bpm / median_bpm

    # Within 1.4x-2.8x → likely double time
    if 1.4 < ratio < 2.8:
        return bpm / 2.0

    # Within 0.35x-0.7x → likely half time
    if 0.35 < ratio < 0.7:
        return bpm * 2.0

    return bpm


def hz_to_cents(hz_a: float, hz_b: float) -> float:
    """
    Compute pitch deviation in cents between two frequencies.
    Returns absolute deviation.
    """
    if hz_a <= 0 or hz_b <= 0:
        return 0.0
    return abs(1200.0 * math.log2(hz_a / hz_b))


def analyze_take(audio: np.ndarray, sr: int) -> Dict:
    """
    Analyze a single take: estimate BPM, pitch center, and RMS energy.

    Returns dict with keys: bpm, pitch_hz, rms_energy.
    Values are 0.0 if estimation fails.
    """
    bpm = estimate_tempo(audio, sr)
    pitch_hz = estimate_pitch_center(audio, sr)
    rms = compute_rms_energy(audio)

    return {
        "bpm": bpm,
        "pitch_hz": pitch_hz,
        "rms_energy": rms,
    }


# ── Main Orchestrator ────────────────────────────────────


def prefilter_takes(
    takes: List[np.ndarray],
    sr: int,
    max_bpm_deviation: float = 15.0,
    max_pitch_deviation: float = 80.0,
    max_energy_deviation: float = 40.0,
    progress_cb: Optional[Callable] = None,
) -> Tuple[List[np.ndarray], Dict]:
    """
    Pre-filter takes by excluding outliers based on BPM, pitch, and energy.

    Args:
        takes: List of audio arrays (mono, same sr).
        sr: Sample rate.
        max_bpm_deviation: Max deviation from median BPM (% of median).
        max_pitch_deviation: Max deviation from median pitch (cents).
        max_energy_deviation: Max deviation from median energy (% of median).
        progress_cb: Optional (pct, msg) callback.

    Returns:
        (filtered_takes, report_dict) tuple.
        If fewer than 3 takes, returns all takes unchanged.
    """

    def progress(pct, msg):
        if progress_cb:
            progress_cb(pct, msg)

    n = len(takes)

    # Need at least 3 takes for meaningful filtering
    if n < 3:
        return takes, {
            "enabled": True,
            "skipped": True,
            "reason": "Menos de 3 takes — pre-filtro ignorado",
            "total_takes": n,
            "excluded_count": 0,
            "kept_count": n,
        }

    # ── Step 1: Analyze all takes ──
    analyses = []
    for i, audio in enumerate(takes):
        progress(7 + int(2 * i / n), f"Pre-filtro: analisando take {i + 1}/{n}...")
        info = analyze_take(audio, sr)
        info["take"] = i + 1  # 1-indexed for display
        analyses.append(info)
        log.info(
            f"  Take {i + 1}: BPM={info['bpm']:.1f}, "
            f"Pitch={info['pitch_hz']:.1f} Hz, "
            f"RMS={info['rms_energy']:.4f}"
        )

    # ── Step 2: Compute medians ──
    valid_bpms = [a["bpm"] for a in analyses if a["bpm"] > 0]
    valid_pitches = [a["pitch_hz"] for a in analyses if a["pitch_hz"] > 0]
    valid_energies = [a["rms_energy"] for a in analyses if a["rms_energy"] > 0]

    median_bpm = float(np.median(valid_bpms)) if valid_bpms else 0.0
    median_pitch = float(np.median(valid_pitches)) if valid_pitches else 0.0
    median_energy = float(np.median(valid_energies)) if valid_energies else 0.0

    log.info(
        f"  Medians: BPM={median_bpm:.1f}, "
        f"Pitch={median_pitch:.1f} Hz, "
        f"RMS={median_energy:.4f}"
    )

    # ── Step 3: Fold tempos + compute deviations ──
    for a in analyses:
        reasons = []

        # BPM deviation
        if a["bpm"] > 0 and median_bpm > 0:
            folded = fold_tempo(a["bpm"], median_bpm)
            a["bpm_folded"] = round(folded, 1)
            dev = abs(folded - median_bpm) / median_bpm * 100
            a["bpm_deviation_pct"] = round(dev, 1)
            if dev > max_bpm_deviation:
                reasons.append("bpm")
        else:
            a["bpm_folded"] = a["bpm"]
            a["bpm_deviation_pct"] = 0.0

        # Pitch deviation (cents)
        if a["pitch_hz"] > 0 and median_pitch > 0:
            dev_cents = hz_to_cents(a["pitch_hz"], median_pitch)
            a["pitch_deviation_cents"] = round(dev_cents, 1)
            if dev_cents > max_pitch_deviation:
                reasons.append("pitch")
        else:
            a["pitch_deviation_cents"] = 0.0

        # Energy deviation
        if a["rms_energy"] > 0 and median_energy > 0:
            dev = abs(a["rms_energy"] - median_energy) / median_energy * 100
            a["energy_deviation_pct"] = round(dev, 1)
            if dev > max_energy_deviation:
                reasons.append("energy")
        else:
            a["energy_deviation_pct"] = 0.0

        a["exclusion_reasons"] = reasons
        a["excluded"] = len(reasons) > 0

    # ── Step 4: Safety check — keep at least 2 takes ──
    excluded_indices = [i for i, a in enumerate(analyses) if a["excluded"]]
    kept_indices = [i for i, a in enumerate(analyses) if not a["excluded"]]

    if len(kept_indices) < 2:
        # Too many exclusions — keep all, warn
        log.warning(
            f"  Pre-filter would exclude {len(excluded_indices)}/{n} takes "
            f"(leaving <2). Keeping all takes."
        )
        for a in analyses:
            a["excluded"] = False
            a["exclusion_reasons"] = []
        excluded_indices = []
        kept_indices = list(range(n))

    # ── Step 5: Build filtered list ──
    filtered = [takes[i] for i in kept_indices]
    excluded_count = n - len(filtered)

    if excluded_count > 0:
        excluded_takes = [a["take"] for a in analyses if a["excluded"]]
        log.info(
            f"  Pre-filter: excluiu {excluded_count} take(s): {excluded_takes}"
        )
    else:
        log.info("  Pre-filter: nenhum outlier detectado")

    progress(9, f"Pre-filtro: {excluded_count} outlier(s) excluido(s)")

    # ── Step 6: Build report ──
    report = {
        "enabled": True,
        "skipped": False,
        "median_bpm": round(median_bpm, 1),
        "median_pitch_hz": round(median_pitch, 1),
        "median_rms_energy": round(median_energy, 5),
        "thresholds": {
            "max_bpm_deviation_pct": max_bpm_deviation,
            "max_pitch_deviation_cents": max_pitch_deviation,
            "max_energy_deviation_pct": max_energy_deviation,
        },
        "total_takes": n,
        "excluded_count": excluded_count,
        "kept_count": len(filtered),
        "per_take": [
            {
                "take": a["take"],
                "bpm": round(a["bpm"], 1),
                "bpm_folded": a.get("bpm_folded", a["bpm"]),
                "pitch_hz": round(a["pitch_hz"], 1),
                "rms_energy": round(a["rms_energy"], 5),
                "bpm_deviation_pct": a["bpm_deviation_pct"],
                "pitch_deviation_cents": a["pitch_deviation_cents"],
                "energy_deviation_pct": a["energy_deviation_pct"],
                "excluded": a["excluded"],
                "exclusion_reasons": a["exclusion_reasons"],
            }
            for a in analyses
        ],
    }

    return filtered, report

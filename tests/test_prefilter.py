"""
Test: Take Pre-Filtering (Outlier Detection)

Generates synthetic takes with controlled BPM, pitch, and energy,
then verifies that prefilter_takes() correctly excludes outliers.
"""

import sys
import os
import math
import numpy as np

# Add project root to path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from backend.engine.prefilter import (
    compute_rms_energy,
    fold_tempo,
    hz_to_cents,
    analyze_take,
    prefilter_takes,
)


# ── Helpers ──────────────────────────────────

def make_tone(freq: float, duration_s: float, sr: int, amplitude: float = 0.5) -> np.ndarray:
    """Generate a sine wave at a given frequency, duration, and amplitude."""
    t = np.arange(int(duration_s * sr)) / sr
    audio = np.sin(2 * np.pi * freq * t) * amplitude
    # Fade in/out to avoid spectral artifacts
    fade = min(int(0.02 * sr), len(audio) // 4)
    if fade > 0:
        audio[:fade] *= np.linspace(0, 1, fade)
        audio[-fade:] *= np.linspace(1, 0, fade)
    return audio


def make_similar_takes(n: int, freq: float = 220.0, sr: int = 22050,
                       duration_s: float = 4.0, amplitude: float = 0.5) -> list:
    """Create n similar takes with slight random variation."""
    takes = []
    for i in range(n):
        # Small random variation in frequency (±5 Hz) and amplitude (±10%)
        f = freq + np.random.uniform(-5, 5)
        a = amplitude * np.random.uniform(0.9, 1.1)
        takes.append(make_tone(f, duration_s, sr, a))
    return takes


# ── Unit Tests ───────────────────────────────

def test_compute_rms_energy():
    """RMS of sine wave should be amplitude / sqrt(2); silence should be 0."""
    sr = 22050

    # Silence
    silence = np.zeros(sr)
    assert compute_rms_energy(silence) == 0.0

    # Empty array
    assert compute_rms_energy(np.array([])) == 0.0

    # Sine wave at amplitude 1.0 → RMS ≈ 0.707
    t = np.arange(sr) / sr
    sine = np.sin(2 * np.pi * 440 * t) * 1.0
    rms = compute_rms_energy(sine)
    assert abs(rms - 0.707) < 0.01, f"Expected ~0.707, got {rms}"

    # Sine wave at amplitude 0.5 → RMS ≈ 0.354
    sine_half = sine * 0.5
    rms_half = compute_rms_energy(sine_half)
    assert abs(rms_half - 0.354) < 0.01, f"Expected ~0.354, got {rms_half}"


def test_fold_tempo():
    """Fold BPM should correct double/half time detection errors."""
    # Normal case — no folding needed
    assert fold_tempo(80.0, 80.0) == 80.0
    assert fold_tempo(85.0, 80.0) == 85.0

    # Double time: 160 BPM when median is 80 → fold to 80
    folded = fold_tempo(160.0, 80.0)
    assert abs(folded - 80.0) < 0.01, f"Expected 80.0, got {folded}"

    # Half time: 40 BPM when median is 80 → fold to 80
    folded = fold_tempo(40.0, 80.0)
    assert abs(folded - 80.0) < 0.01, f"Expected 80.0, got {folded}"

    # Edge: zero BPM → return as-is
    assert fold_tempo(0.0, 80.0) == 0.0
    assert fold_tempo(80.0, 0.0) == 80.0

    # Not quite double (1.3x) → no folding
    assert fold_tempo(104.0, 80.0) == 104.0


def test_hz_to_cents():
    """Pitch deviation in cents should be musically correct."""
    # Same frequency → 0 cents
    assert hz_to_cents(440.0, 440.0) == 0.0

    # One octave up → 1200 cents
    assert abs(hz_to_cents(880.0, 440.0) - 1200.0) < 0.1

    # One semitone → ~100 cents
    # A4 (440 Hz) to A#4 (466.16 Hz)
    cents = hz_to_cents(466.16, 440.0)
    assert abs(cents - 100.0) < 1.0, f"Expected ~100 cents, got {cents}"

    # Zero frequency → 0 (safety)
    assert hz_to_cents(0.0, 440.0) == 0.0
    assert hz_to_cents(440.0, 0.0) == 0.0


def test_analyze_take():
    """analyze_take should return a dict with bpm, pitch_hz, rms_energy."""
    sr = 22050
    audio = make_tone(220.0, 4.0, sr, 0.5)

    result = analyze_take(audio, sr)

    assert "bpm" in result
    assert "pitch_hz" in result
    assert "rms_energy" in result
    assert result["rms_energy"] > 0


def test_prefilter_excludes_energy_outlier():
    """One very quiet take among normal takes should be excluded."""
    sr = 22050
    np.random.seed(42)

    # 5 normal takes
    takes = make_similar_takes(5, freq=220.0, sr=sr, amplitude=0.5)

    # Add 1 very quiet take (10% of normal amplitude)
    quiet_take = make_tone(220.0, 4.0, sr, amplitude=0.05)
    takes.append(quiet_take)

    filtered, report = prefilter_takes(
        takes, sr,
        max_bpm_deviation=50.0,    # generous to not trigger BPM filter
        max_pitch_deviation=200.0,  # generous to not trigger pitch filter
        max_energy_deviation=40.0,  # will catch the quiet take
    )

    assert report["enabled"] is True
    assert report["skipped"] is False
    assert report["excluded_count"] >= 1
    assert len(filtered) < len(takes)
    assert len(filtered) >= 2  # safety: always keep at least 2


def test_prefilter_keeps_minimum_two():
    """If all takes are different, should still keep at least 2."""
    sr = 22050

    # 3 very different takes — different frequencies and amplitudes
    takes = [
        make_tone(110.0, 4.0, sr, 0.1),  # low freq, quiet
        make_tone(440.0, 4.0, sr, 0.5),  # medium freq, normal
        make_tone(880.0, 4.0, sr, 0.9),  # high freq, loud
    ]

    filtered, report = prefilter_takes(
        takes, sr,
        max_bpm_deviation=5.0,    # very strict
        max_pitch_deviation=10.0,  # very strict
        max_energy_deviation=10.0,  # very strict
    )

    # Must keep at least 2 regardless
    assert len(filtered) >= 2
    assert report["kept_count"] >= 2


def test_prefilter_skips_with_few_takes():
    """With fewer than 3 takes, filtering should be skipped."""
    sr = 22050
    takes = [
        make_tone(220.0, 4.0, sr, 0.5),
        make_tone(440.0, 4.0, sr, 0.5),
    ]

    filtered, report = prefilter_takes(takes, sr)

    assert report["skipped"] is True
    assert report["excluded_count"] == 0
    assert len(filtered) == 2


def test_prefilter_report_structure():
    """Report dict should have all required keys."""
    sr = 22050
    np.random.seed(99)
    takes = make_similar_takes(4, freq=220.0, sr=sr)

    _, report = prefilter_takes(takes, sr)

    # Top-level keys
    assert "enabled" in report
    assert "total_takes" in report
    assert "excluded_count" in report
    assert "kept_count" in report

    # If not skipped, should have medians and per-take data
    if not report.get("skipped"):
        assert "median_bpm" in report
        assert "median_pitch_hz" in report
        assert "median_rms_energy" in report
        assert "thresholds" in report
        assert "per_take" in report
        assert len(report["per_take"]) == 4

        # Check per-take entry structure
        pt = report["per_take"][0]
        assert "take" in pt
        assert "bpm" in pt
        assert "pitch_hz" in pt
        assert "excluded" in pt
        assert "exclusion_reasons" in pt


# ── Run all tests ────────────────────────────

if __name__ == "__main__":
    tests = [
        test_compute_rms_energy,
        test_fold_tempo,
        test_hz_to_cents,
        test_analyze_take,
        test_prefilter_excludes_energy_outlier,
        test_prefilter_keeps_minimum_two,
        test_prefilter_skips_with_few_takes,
        test_prefilter_report_structure,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            print(f"  ✓ {test.__name__}")
            passed += 1
        except Exception as e:
            print(f"  ✗ {test.__name__}: {e}")
            failed += 1

    print(f"\n{'='*40}")
    print(f"  {passed} passed, {failed} failed")
    print(f"{'='*40}")

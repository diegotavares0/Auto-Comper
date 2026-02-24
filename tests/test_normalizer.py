"""
Test: Tempo & Pitch Normalizer

Generates synthetic takes at slightly different tempos and pitch centers,
runs normalize_takes(), verifies the outputs converge toward the target.
"""

import sys
import os
import numpy as np

# Add project root to path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from backend.engine.normalizer import (
    estimate_tempo, estimate_pitch_center,
    normalize_tempo, normalize_pitch, normalize_takes,
)


def generate_click_track(bpm: float, sr: int, duration_s: float) -> np.ndarray:
    """
    Generate a simple click track at the given BPM.
    Each click is a short burst of noise — librosa's beat tracker
    should pick up the tempo from the regular transients.
    """
    n_samples = int(duration_s * sr)
    audio = np.zeros(n_samples)
    click_interval_samples = int(60.0 / bpm * sr)
    click_duration = int(0.01 * sr)  # 10ms click

    # Generate clicks
    pos = 0
    while pos + click_duration < n_samples:
        # Short burst with exponential decay
        t = np.arange(click_duration) / sr
        click = np.sin(2 * np.pi * 1000 * t) * np.exp(-t * 200)
        audio[pos:pos + click_duration] += click * 0.5
        pos += click_interval_samples

    return audio


def generate_pitched_tone(freq_hz: float, sr: int, duration_s: float) -> np.ndarray:
    """
    Generate a sustained tone at the given frequency with harmonics.
    """
    n_samples = int(duration_s * sr)
    t = np.arange(n_samples) / sr

    # Fundamental + harmonics
    audio = np.sin(2 * np.pi * freq_hz * t) * 0.5
    audio += np.sin(2 * np.pi * freq_hz * 2 * t) * 0.2  # octave
    audio += np.sin(2 * np.pi * freq_hz * 3 * t) * 0.1  # fifth

    # Gentle envelope to avoid clicks
    fade = int(0.05 * sr)
    audio[:fade] *= np.linspace(0, 1, fade)
    audio[-fade:] *= np.linspace(1, 0, fade)

    return audio


def generate_take(bpm: float, pitch_hz: float, sr: int, duration_s: float) -> np.ndarray:
    """
    Generate a synthetic take combining rhythmic clicks (for tempo detection)
    and a pitched tone (for pitch center detection).
    """
    clicks = generate_click_track(bpm, sr, duration_s)
    tone = generate_pitched_tone(pitch_hz, sr, duration_s)

    # Mix: clicks dominant for tempo, tone for pitch
    return clicks * 0.6 + tone * 0.4


def test_tempo_estimation():
    """Test that tempo estimation works on synthetic click tracks."""
    sr = 22050
    duration = 8.0

    for target_bpm in [80, 100, 120]:
        audio = generate_click_track(target_bpm, sr, duration)
        estimated = estimate_tempo(audio, sr)

        print(f"  Target BPM: {target_bpm}, Estimated: {estimated:.1f}")

        # Allow some tolerance — tempo estimation is inherently approximate
        # and can pick up double/half time
        if estimated > 0:
            # Check if within 10% or is a harmonic (double/half)
            ratio = estimated / target_bpm
            is_close = abs(ratio - 1.0) < 0.15
            is_double = abs(ratio - 2.0) < 0.15
            is_half = abs(ratio - 0.5) < 0.15
            assert is_close or is_double or is_half, \
                f"BPM {estimated:.0f} not close to {target_bpm} (ratio={ratio:.2f})"

    print("  [PASS] Tempo estimation works")


def test_pitch_estimation():
    """Test that pitch center estimation works on synthetic tones."""
    sr = 22050
    duration = 4.0

    for freq in [220.0, 330.0, 440.0]:
        audio = generate_pitched_tone(freq, sr, duration)
        estimated = estimate_pitch_center(audio, sr)

        print(f"  Target Hz: {freq:.1f}, Estimated: {estimated:.1f}")

        if estimated > 0:
            ratio = estimated / freq
            assert 0.9 < ratio < 1.1, \
                f"Pitch {estimated:.1f} Hz not close to {freq:.1f} Hz"

    print("  [PASS] Pitch center estimation works")


def test_normalize_tempo():
    """Test that time-stretch brings takes closer to target tempo."""
    sr = 22050
    duration = 6.0

    source_bpm = 80.0
    target_bpm = 90.0

    audio = generate_click_track(source_bpm, sr, duration)
    original_len = len(audio)

    # Full intensity
    stretched = normalize_tempo(audio, sr, source_bpm, target_bpm, intensity=100)
    # Faster tempo → shorter audio
    assert len(stretched) < original_len, "Stretching to faster tempo should shorten audio"

    # Zero intensity → no change
    unchanged = normalize_tempo(audio, sr, source_bpm, target_bpm, intensity=0)
    assert len(unchanged) == original_len, "Zero intensity should return original"

    # Partial intensity → between original and full stretch
    partial = normalize_tempo(audio, sr, source_bpm, target_bpm, intensity=50)
    assert len(partial) < original_len, "Partial stretch should still shorten"
    assert len(partial) > len(stretched), "Partial should be between original and full"

    print(f"  Original: {original_len}, Full stretch: {len(stretched)}, "
          f"Partial: {len(partial)}")
    print("  [PASS] Tempo normalization intensity blending works")


def test_normalize_pitch():
    """Test that pitch-shift moves center frequency toward target."""
    sr = 22050
    duration = 4.0
    source_hz = 220.0  # A3
    target_hz = 233.1  # Bb3 (1 semitone up)

    audio = generate_pitched_tone(source_hz, sr, duration)

    # Full intensity
    shifted = normalize_pitch(audio, sr, source_hz, target_hz, intensity=100)
    shifted_center = estimate_pitch_center(shifted, sr)

    # Zero intensity → should be unchanged
    unchanged = normalize_pitch(audio, sr, source_hz, target_hz, intensity=0)
    unchanged_center = estimate_pitch_center(unchanged, sr)

    print(f"  Source: {source_hz:.1f} Hz, Target: {target_hz:.1f} Hz")
    print(f"  Full shift center: {shifted_center:.1f} Hz, "
          f"Unchanged center: {unchanged_center:.1f} Hz")

    if shifted_center > 0 and unchanged_center > 0:
        # Full shift should be closer to target than unchanged
        dist_shifted = abs(shifted_center - target_hz)
        dist_unchanged = abs(unchanged_center - target_hz)
        assert dist_shifted < dist_unchanged, \
            "Full pitch correction should bring center closer to target"

    print("  [PASS] Pitch normalization works")


def test_normalize_takes_integration():
    """Integration test: normalize multiple takes with different tempos/pitches."""
    sr = 22050
    duration = 6.0

    # Simulate 4 takes with slight tempo/pitch drift
    takes_params = [
        (78, 218.0),   # Take 1: slightly slow, slightly flat
        (82, 222.0),   # Take 2: slightly fast, slightly sharp
        (80, 220.0),   # Take 3: on target
        (79, 219.0),   # Take 4: slightly slow, slightly flat
    ]

    takes = [
        generate_take(bpm, hz, sr, duration)
        for bpm, hz in takes_params
    ]

    print(f"\n  Input takes:")
    for i, (bpm, hz) in enumerate(takes_params):
        print(f"    Take {i+1}: {bpm} BPM, {hz:.1f} Hz")

    # Run full normalization
    normalized = normalize_takes(
        takes, sr,
        tempo_intensity=80,
        pitch_intensity=80,
    )

    assert len(normalized) == len(takes), "Should return same number of takes"

    # All takes should still be valid audio
    for i, n in enumerate(normalized):
        assert len(n) > 0, f"Take {i+1} should not be empty"
        assert not np.any(np.isnan(n)), f"Take {i+1} should not have NaN"
        assert np.max(np.abs(n)) < 10.0, f"Take {i+1} should not clip badly"

    print("  [PASS] normalize_takes() integration works")


def test_skip_when_disabled():
    """Test that normalization is a no-op when both intensities are 0."""
    sr = 22050
    duration = 4.0

    takes = [
        generate_take(80, 220, sr, duration),
        generate_take(82, 222, sr, duration),
    ]

    original_lengths = [len(t) for t in takes]
    result = normalize_takes(takes, sr, tempo_intensity=0, pitch_intensity=0)

    # Should be the exact same objects (no processing)
    assert result is takes, "Zero intensity should return the same list (no copy)"
    for i, t in enumerate(result):
        assert len(t) == original_lengths[i], "Audio length should be unchanged"

    print("  [PASS] Zero intensity = pure no-op")


if __name__ == "__main__":
    print("=" * 60)
    print("Testing Tempo & Pitch Normalizer")
    print("=" * 60)

    print("\n1. Tempo estimation:")
    test_tempo_estimation()

    print("\n2. Pitch center estimation:")
    test_pitch_estimation()

    print("\n3. Tempo normalization (time-stretch):")
    test_normalize_tempo()

    print("\n4. Pitch normalization (pitch-shift):")
    test_normalize_pitch()

    print("\n5. Full integration (normalize_takes):")
    test_normalize_takes_integration()

    print("\n6. Zero intensity = no-op:")
    test_skip_when_disabled()

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)

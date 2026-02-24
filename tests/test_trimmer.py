"""
Test: Auto-Trimmer

Generates synthetic audio with leading/trailing silence,
runs trim_audio() and trim_takes(), verifies silence removal.
"""

import sys
import os
import numpy as np

# Add project root to path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from backend.engine.trimmer import trim_audio, trim_takes, compute_rms_db


def generate_audio_with_silence(
    sr: int, silence_start_s: float, music_s: float,
    silence_end_s: float, freq: float = 440.0,
) -> np.ndarray:
    """
    Generate audio: [silence] [tone at freq] [silence]
    """
    silence_start = np.zeros(int(silence_start_s * sr))
    silence_end = np.zeros(int(silence_end_s * sr))

    n_music = int(music_s * sr)
    t = np.arange(n_music) / sr
    music = np.sin(2 * np.pi * freq * t) * 0.5

    # Fade in/out to avoid clicks
    fade = min(int(0.01 * sr), n_music // 4)
    if fade > 0:
        music[:fade] *= np.linspace(0, 1, fade)
        music[-fade:] *= np.linspace(1, 0, fade)

    return np.concatenate([silence_start, music, silence_end])


def test_compute_rms_db():
    """Test that RMS energy is computed correctly."""
    sr = 22050

    # Silence should be very low dB
    silence = np.zeros(sr)
    rms = compute_rms_db(silence, frame_size=2048, hop_size=512)
    assert np.all(rms < -80), "Silence should be below -80 dB"

    # Loud tone should be well above threshold
    t = np.arange(sr) / sr
    loud = np.sin(2 * np.pi * 440 * t) * 0.5
    rms = compute_rms_db(loud, frame_size=2048, hop_size=512)
    assert np.mean(rms) > -20, "Loud tone should be above -20 dB"

    print("  [PASS] RMS energy computation works")


def test_trim_audio_basic():
    """Test basic silence trimming."""
    sr = 22050

    # 2s silence + 3s music + 1.5s silence = 6.5s total
    audio = generate_audio_with_silence(sr, 2.0, 3.0, 1.5)
    total_duration = len(audio) / sr

    print(f"  Input: {total_duration:.1f}s (2s silence + 3s music + 1.5s silence)")

    trimmed, info = trim_audio(audio, sr, threshold_db=-45.0)

    assert info["trimmed"], "Should detect and trim silence"
    assert info["trimmed_duration_s"] < info["original_duration_s"], \
        "Trimmed should be shorter than original"

    # Should remove most of the leading silence (minus pre-roll)
    assert info["removed_start_s"] > 1.5, \
        f"Should remove >1.5s of leading silence, got {info['removed_start_s']:.2f}s"

    # Should remove most of the trailing silence (minus post-roll)
    assert info["removed_end_s"] > 1.0, \
        f"Should remove >1.0s of trailing silence, got {info['removed_end_s']:.2f}s"

    print(f"  Trimmed: {info['trimmed_duration_s']:.1f}s "
          f"(removed {info['removed_start_s']:.2f}s start, "
          f"{info['removed_end_s']:.2f}s end)")
    print("  [PASS] Basic silence trimming works")


def test_trim_no_silence():
    """Test that audio without silence is not trimmed significantly."""
    sr = 22050

    # Music starts immediately — no leading silence
    t = np.arange(int(3.0 * sr)) / sr
    audio = np.sin(2 * np.pi * 440 * t) * 0.5

    trimmed, info = trim_audio(audio, sr, threshold_db=-45.0)

    # Should barely trim anything
    assert info["removed_start_s"] < 0.1, \
        "Should not remove significant audio from start"

    print(f"  Removed: start={info['removed_start_s']:.2f}s, "
          f"end={info['removed_end_s']:.2f}s")
    print("  [PASS] No-silence audio preserved")


def test_trim_takes_multiple():
    """Test trimming multiple takes with different silence amounts."""
    sr = 22050

    takes = [
        generate_audio_with_silence(sr, 0.5, 4.0, 0.3),   # Take 1: little silence
        generate_audio_with_silence(sr, 3.0, 4.0, 2.0),   # Take 2: lots of silence
        generate_audio_with_silence(sr, 1.0, 4.0, 0.5),   # Take 3: moderate
        generate_audio_with_silence(sr, 0.1, 4.0, 0.1),   # Take 4: almost none
    ]

    print(f"  Input takes:")
    for i, t in enumerate(takes):
        print(f"    Take {i+1}: {len(t)/sr:.1f}s")

    trimmed, infos = trim_takes(takes, sr, threshold_db=-45.0)

    assert len(trimmed) == len(takes), "Should return same number of takes"
    assert len(infos) == len(takes), "Should return info for each take"

    # Take 2 should have the most trimmed
    assert infos[1]["removed_start_s"] > infos[0]["removed_start_s"], \
        "Take 2 had more leading silence, should remove more"

    print(f"\n  Trimmed takes:")
    for i, info in enumerate(infos):
        print(f"    Take {i+1}: {info['original_duration_s']:.1f}s → "
              f"{info['trimmed_duration_s']:.1f}s "
              f"(start: -{info['removed_start_s']:.2f}s, "
              f"end: -{info['removed_end_s']:.2f}s)")

    print("  [PASS] Multi-take trimming works")


def test_trim_too_short():
    """Test that very short music region skips trimming."""
    sr = 22050

    # 5s silence + 0.3s music + 5s silence (music too short)
    audio = generate_audio_with_silence(sr, 5.0, 0.3, 5.0)

    trimmed, info = trim_audio(
        audio, sr,
        threshold_db=-45.0,
        min_music_duration_s=1.0,  # Music must be >= 1s
    )

    assert not info["trimmed"], "Should skip trimming for too-short music"
    assert len(trimmed) == len(audio), "Should return original audio unchanged"

    print("  [PASS] Too-short music detection works")


if __name__ == "__main__":
    print("=" * 60)
    print("Testing Auto-Trimmer")
    print("=" * 60)

    print("\n1. RMS energy computation:")
    test_compute_rms_db()

    print("\n2. Basic silence trimming:")
    test_trim_audio_basic()

    print("\n3. No-silence audio:")
    test_trim_no_silence()

    print("\n4. Multi-take trimming:")
    test_trim_takes_multiple()

    print("\n5. Too-short music detection:")
    test_trim_too_short()

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)

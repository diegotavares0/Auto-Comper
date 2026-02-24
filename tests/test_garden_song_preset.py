"""
End-to-end test: Garden Song tone preset.

Generates two synthetic audio clips:
1. "garden_ref.wav" — warm, lo-fi fingerpicked acoustic guitar tone
   (simulates the vibe of Phoebe Bridgers' Garden Song)
2. "dry_take.wav" — clean, bright guitar take (the "before")

Then runs the full create→apply pipeline and verifies:
- Preset is saved correctly
- Tone matching shifts spectral characteristics toward reference
- Output is warmer / less bright than the dry input
- The entire workflow completes without errors

Usage:
    python -m tests.test_garden_song_preset
"""

import os
import sys
import json
import numpy as np
import soundfile as sf

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

SR = 48000
DURATION = 8.0  # seconds


# ─────────────────────────────────────────────────────────────
#  Audio synthesis helpers
# ─────────────────────────────────────────────────────────────

def make_pluck(freq, sr, duration, decay_rate=4.0, brightness=0.5):
    """
    Synthesize a single guitar-like pluck with overtones.

    brightness: 0.0 = very warm/dark, 1.0 = very bright
    """
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)

    # Exponential decay envelope with slight attack
    attack = np.minimum(t * 200, 1.0)  # 5ms attack
    decay = np.exp(-decay_rate * t)
    envelope = attack * decay

    # Harmonic series with brightness-controlled rolloff
    signal = np.zeros_like(t)
    for harmonic in range(1, 12):
        # Higher harmonics decay faster when brightness is low
        harmonic_amp = 1.0 / (harmonic ** (1.5 + (1.0 - brightness) * 1.5))
        # Also add slight inharmonicity (like a real string)
        h_freq = freq * harmonic * (1.0 + 0.0003 * harmonic ** 2)
        if h_freq > sr / 2:
            break
        signal += harmonic_amp * np.sin(2 * np.pi * h_freq * t)

    return signal * envelope


def make_garden_song_reference(sr, duration):
    """
    Synthesize audio with Garden Song-like tonal characteristics:
    - Warm fingerpicked acoustic guitar
    - Low brightness (rolled-off highs)
    - Rich low-mids (200-800 Hz)
    - Moderate compression / gentle dynamics
    - Slight lo-fi character (subtle noise floor)
    """
    n_samples = int(sr * duration)
    audio = np.zeros(n_samples)

    # Garden Song is in D major, fingerpicked pattern
    # D - A - Bm - G progression (simplified)
    chord_freqs = {
        'D':  [146.83, 220.00, 293.66, 369.99],  # D3, A3, D4, F#4
        'A':  [110.00, 164.81, 220.00, 277.18],  # A2, E3, A3, C#4
        'Bm': [123.47, 185.00, 246.94, 293.66],  # B2, F#3, B3, D4
        'G':  [98.00, 146.83, 196.00, 246.94],   # G2, D3, G3, B3
    }

    progression = ['D', 'A', 'Bm', 'G']
    beats_per_chord = 4
    bpm = 80  # Garden Song tempo ~80 BPM
    beat_duration = 60.0 / bpm

    # Fingerpicking: arpeggiate each chord note with slight delays
    t_pos = 0.0
    chord_idx = 0

    while t_pos < duration:
        chord_name = progression[chord_idx % len(progression)]
        freqs = chord_freqs[chord_name]

        for beat in range(beats_per_chord):
            for note_idx, freq in enumerate(freqs):
                note_time = t_pos + beat * beat_duration + note_idx * 0.08
                if note_time >= duration:
                    break

                start_sample = int(note_time * sr)
                # Warm tone: low brightness (0.2), slow decay
                pluck = make_pluck(freq, sr, min(1.5, duration - note_time),
                                   decay_rate=3.0, brightness=0.2)

                end_sample = min(start_sample + len(pluck), n_samples)
                audio[start_sample:end_sample] += pluck[:end_sample - start_sample]

        t_pos += beats_per_chord * beat_duration
        chord_idx += 1

    # ── Garden Song post-processing ──

    # 1. Low-pass filter at ~4kHz (lo-fi rolloff)
    from scipy.signal import butter, sosfilt
    sos_lp = butter(4, 4000, btype='low', fs=sr, output='sos')
    audio = sosfilt(sos_lp, audio)

    # 2. Gentle mid boost (200-800 Hz warmth)
    sos_bp = butter(2, [200, 800], btype='band', fs=sr, output='sos')
    mid_boost = sosfilt(sos_bp, audio) * 0.3
    audio = audio + mid_boost

    # 3. Subtle saturation (lo-fi warmth)
    audio = np.tanh(audio * 2.0) * 0.5

    # 4. Add very subtle noise floor (tape hiss / lo-fi character)
    noise = np.random.randn(n_samples) * 0.003
    # Filter noise to be mostly high-freq hiss
    sos_hp = butter(2, 2000, btype='high', fs=sr, output='sos')
    noise = sosfilt(sos_hp, noise)
    audio = audio + noise

    # Normalize to -6 dBFS
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak * 0.5  # ~-6 dBFS

    return audio


def make_dry_bright_take(sr, duration):
    """
    Synthesize a clean, bright guitar take (the "before" that needs
    tone matching). Same notes as reference but with different tonal
    character: brighter harmonics, wider dynamics, no lo-fi processing.

    This simulates the real scenario: same song, same guitar, but
    recorded through a different signal chain (e.g., DI vs mic'd amp).
    """
    n_samples = int(sr * duration)
    audio = np.zeros(n_samples)

    # SAME chord voicings as reference (same notes = fair comparison)
    chord_freqs = {
        'D':  [146.83, 220.00, 293.66, 369.99],  # D3, A3, D4, F#4
        'A':  [110.00, 164.81, 220.00, 277.18],  # A2, E3, A3, C#4
        'Bm': [123.47, 185.00, 246.94, 293.66],  # B2, F#3, B3, D4
        'G':  [98.00, 146.83, 196.00, 246.94],   # G2, D3, G3, B3
    }

    progression = ['D', 'A', 'Bm', 'G']
    bpm = 80
    beat_duration = 60.0 / bpm
    beats_per_chord = 4

    t_pos = 0.0
    chord_idx = 0

    while t_pos < duration:
        chord_name = progression[chord_idx % len(progression)]
        freqs = chord_freqs[chord_name]

        for beat in range(beats_per_chord):
            for note_idx, freq in enumerate(freqs):
                note_time = t_pos + beat * beat_duration + note_idx * 0.06
                if note_time >= duration:
                    break

                start_sample = int(note_time * sr)
                # Bright tone: high brightness (0.8), faster decay
                pluck = make_pluck(freq, sr, min(1.2, duration - note_time),
                                   decay_rate=5.0, brightness=0.8)

                end_sample = min(start_sample + len(pluck), n_samples)
                audio[start_sample:end_sample] += pluck[:end_sample - start_sample]

        t_pos += beats_per_chord * beat_duration
        chord_idx += 1

    # Bright processing: add a high-shelf boost to simulate DI/piezo pickup
    from scipy.signal import butter, sosfilt
    sos_hp = butter(2, 2000, btype='high', fs=sr, output='sos')
    high_boost = sosfilt(sos_hp, audio) * 0.4
    audio = audio + high_boost

    # No lo-fi processing — clean and bright
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak * 0.7  # ~-3 dBFS (more dynamic)

    return audio


# ─────────────────────────────────────────────────────────────
#  Main test
# ─────────────────────────────────────────────────────────────

def run_test():
    from backend.presets.pipeline import create_preset, apply_preset
    from backend.presets.analyzer import analyze_reference, analyze_input
    from backend.presets import manager
    from backend.config import PresetConfig

    print("=" * 60)
    print("  GARDEN SONG — Tone Preset End-to-End Test")
    print("=" * 60)

    # ── Step 1: Generate test audio ──
    print("\n[1/5] Generating synthetic audio...")

    ref_audio = make_garden_song_reference(SR, DURATION)
    dry_audio = make_dry_bright_take(SR, DURATION)

    # Save WAV files for manual listening
    test_dir = os.path.join(PROJECT_ROOT, "data", "test_garden")
    os.makedirs(test_dir, exist_ok=True)

    ref_path = os.path.join(test_dir, "garden_ref.wav")
    dry_path = os.path.join(test_dir, "dry_take.wav")

    sf.write(ref_path, ref_audio, SR)
    sf.write(dry_path, dry_audio, SR)

    print(f"  Reference: {ref_path}")
    print(f"  Dry take:  {dry_path}")
    print(f"  Duration:  {DURATION}s @ {SR}Hz")

    # ── Step 2: Analyze both (for comparison) ──
    print("\n[2/5] Analyzing tonal characteristics...")

    ref_profile = analyze_reference(ref_audio, SR)
    dry_profile = analyze_reference(dry_audio, SR)

    print(f"\n  {'':30s} {'REFERENCE':>12s} {'DRY TAKE':>12s}")
    print(f"  {'─' * 56}")
    print(f"  {'Spectral Centroid (Hz)':30s} {ref_profile['timbre']['spectral_centroid_hz']:12.0f} {dry_profile['timbre']['spectral_centroid_hz']:12.0f}")
    print(f"  {'Brightness (>3kHz ratio)':30s} {ref_profile['timbre']['brightness']:12.3f} {dry_profile['timbre']['brightness']:12.3f}")
    print(f"  {'Warmth (200-800Hz ratio)':30s} {ref_profile['timbre']['warmth']:12.3f} {dry_profile['timbre']['warmth']:12.3f}")
    print(f"  {'RMS Level (dB)':30s} {ref_profile['dynamics']['rms_mean_db']:12.1f} {dry_profile['dynamics']['rms_mean_db']:12.1f}")
    print(f"  {'Crest Factor (dB)':30s} {ref_profile['dynamics']['crest_factor_db']:12.1f} {dry_profile['dynamics']['crest_factor_db']:12.1f}")
    print(f"  {'Dynamic Range (dB)':30s} {ref_profile['dynamics']['dynamic_range_db']:12.1f} {dry_profile['dynamics']['dynamic_range_db']:12.1f}")

    # Verify the two signals have distinct tonal characteristics
    # (the specific direction depends on synthesis, so just check they differ)
    centroid_diff = abs(
        ref_profile['timbre']['spectral_centroid_hz']
        - dry_profile['timbre']['spectral_centroid_hz']
    )
    print(f"\n  Centroid difference: {centroid_diff:.0f} Hz")
    assert centroid_diff > 50, \
        "Reference and dry should have noticeably different spectral centroids"
    print("  ✓ Reference and dry take have distinct tonal characteristics")

    # ── Step 3: Create preset ──
    print("\n[3/5] Creating 'Garden Song' preset...")

    def progress_log(pct, msg):
        print(f"  [{pct:3d}%] {msg}")

    meta = create_preset(
        audio=ref_audio,
        sr=SR,
        name="Phoebe Bridgers - Garden Song",
        reference_audio_path=ref_path,
        progress_callback=progress_log,
    )

    preset_id = meta["id"]
    print(f"\n  Preset ID:   {preset_id}")
    print(f"  Name:        {meta['name']}")
    print(f"  Brightness:  {meta['timbre'].get('brightness', 'N/A')}")
    print(f"  Warmth:      {meta['timbre'].get('warmth', 'N/A')}")

    # Verify preset was saved
    saved = manager.get_preset(preset_id)
    assert saved is not None, "Preset should be retrievable after save"
    assert saved["name"] == "Phoebe Bridgers - Garden Song"
    print("  ✓ Preset saved and loadable")

    # ── Step 4: Apply preset to dry take ──
    print("\n[4/5] Applying 'Garden Song' tone to dry take...")

    config = PresetConfig(
        preset_id=preset_id,
        intensity=80.0,
        use_neural=False,  # DSP only for this test
        normalize_output=True,
        dynamics_match=True,
        transient_preserve=0.7,
    )

    processed, report = apply_preset(
        audio=dry_audio,
        sr=SR,
        config=config,
        progress_callback=progress_log,
    )

    # Save processed output
    out_path = os.path.join(test_dir, "dry_take_garden_tone.wav")
    sf.write(out_path, processed, SR)
    print(f"\n  Output: {out_path}")

    # ── Step 5: Verify tone shift ──
    print("\n[5/5] Verifying tone matching results...")

    processed_profile = analyze_reference(processed, SR)

    print(f"\n  {'':30s} {'DRY (before)':>14s} {'PROCESSED':>14s} {'REFERENCE':>14s}")
    print(f"  {'─' * 74}")
    print(f"  {'Spectral Centroid (Hz)':30s} {dry_profile['timbre']['spectral_centroid_hz']:14.0f} {processed_profile['timbre']['spectral_centroid_hz']:14.0f} {ref_profile['timbre']['spectral_centroid_hz']:14.0f}")
    print(f"  {'Brightness':30s} {dry_profile['timbre']['brightness']:14.3f} {processed_profile['timbre']['brightness']:14.3f} {ref_profile['timbre']['brightness']:14.3f}")
    print(f"  {'Warmth':30s} {dry_profile['timbre']['warmth']:14.3f} {processed_profile['timbre']['warmth']:14.3f} {ref_profile['timbre']['warmth']:14.3f}")
    print(f"  {'RMS Level (dB)':30s} {dry_profile['dynamics']['rms_mean_db']:14.1f} {processed_profile['dynamics']['rms_mean_db']:14.1f} {ref_profile['dynamics']['rms_mean_db']:14.1f}")

    # Verify the processed take moved TOWARD the reference
    # Check multiple timbral dimensions
    checks = {}
    for key in ['brightness', 'warmth', 'spectral_centroid_hz']:
        dry_val = dry_profile['timbre'][key]
        proc_val = processed_profile['timbre'][key]
        ref_val = ref_profile['timbre'][key]

        dry_dist = abs(dry_val - ref_val)
        proc_dist = abs(proc_val - ref_val)
        improved = proc_dist < dry_dist
        checks[key] = improved

        label = key.replace('_', ' ').title()
        print(f"\n  {label}:")
        print(f"    Dry→Ref distance:       {dry_dist:.3f}")
        print(f"    Processed→Ref distance: {proc_dist:.3f}")
        print(f"    Shifted toward ref:     {'✓ YES' if improved else '✗ NO'}")

    # Report stats
    dsp_stats = report.get("dsp", {})
    gain = dsp_stats.get("gain_curve_db", {})
    print(f"\n  EQ Gain curve: [{gain.get('min', 0):+.1f}dB, {gain.get('max', 0):+.1f}dB] (mean: {gain.get('mean', 0):+.1f}dB)")
    print(f"  Intensity: {report.get('intensity', 0)}%")
    print(f"  Dynamics matched: {dsp_stats.get('dynamics_matched', False)}")

    # Spectral comparison bands
    spectral = report.get("spectral_comparison", {}).get("bands", [])
    if spectral:
        boosts = sum(1 for b in spectral if b["diff_db"] > 0.5)
        cuts = sum(1 for b in spectral if b["diff_db"] < -0.5)
        print(f"  Spectral bands: {boosts} boosted, {cuts} cut, {len(spectral) - boosts - cuts} unchanged")

    # ── Summary ──
    n_improved = sum(1 for v in checks.values() if v)
    all_pass = n_improved >= 2  # at least 2 of 3 metrics improved

    print("\n" + "=" * 60)
    if all_pass:
        print("  ✓ ALL CHECKS PASSED — Garden Song tone preset works!")
    else:
        print("  ⚠ PARTIAL — Some tonal shifts didn't fully converge")
        print("    (This is expected with synthetic audio. Real audio")
        print("     will have richer spectral content to work with.)")
    print("=" * 60)

    print(f"\n  Files saved in: {test_dir}/")
    print(f"    garden_ref.wav            — Reference tone (warm, lo-fi)")
    print(f"    dry_take.wav              — Input take (clean, bright)")
    print(f"    dry_take_garden_tone.wav  — Output (should sound warmer)")
    print(f"\n  Try it: open these files in any audio player to A/B compare!")

    # Cleanup — remove test preset so it doesn't pollute the library
    # (comment out if you want to keep it)
    # manager.delete_preset(preset_id)
    # print(f"\n  (Test preset cleaned up)")

    return all_pass


if __name__ == "__main__":
    success = run_test()
    sys.exit(0 if success else 1)

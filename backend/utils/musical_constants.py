"""
Musical constants — note names, scales, MIDI helpers.
"""

import numpy as np

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

SCALES = {
    "major":            [0, 2, 4, 5, 7, 9, 11],
    "minor":            [0, 2, 3, 5, 7, 8, 10],
    "harmonic_minor":   [0, 2, 3, 5, 7, 8, 11],
    "melodic_minor":    [0, 2, 3, 5, 7, 9, 11],
    "pentatonic_major": [0, 2, 4, 7, 9],
    "pentatonic_minor": [0, 3, 5, 7, 10],
    "blues":            [0, 3, 5, 6, 7, 10],
    "dorian":           [0, 2, 3, 5, 7, 9, 10],
    "mixolydian":       [0, 2, 4, 5, 7, 9, 10],
    "chromatic":        list(range(12)),
}


def hz_to_midi(hz: np.ndarray) -> np.ndarray:
    """Convert Hz to MIDI note number (fractional)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        midi = 69 + 12 * np.log2(hz / 440.0)
    midi[~np.isfinite(midi)] = 0
    return midi


def midi_to_hz(midi: np.ndarray) -> np.ndarray:
    """Convert MIDI note number to Hz."""
    return 440.0 * (2 ** ((midi - 69) / 12.0))


def midi_to_name(midi_note: int) -> str:
    """MIDI note number to name like 'C4'."""
    octave = (midi_note // 12) - 1
    note = NOTE_NAMES[midi_note % 12]
    return f"{note}{octave}"


def note_name_to_midi(name: str) -> int:
    """Note name to pitch class (0-11). E.g., 'C' -> 0, 'C#' -> 1."""
    name = name.strip().upper()
    for i, n in enumerate(NOTE_NAMES):
        if name == n:
            return i
    raise ValueError(f"Unknown note: {name}")


def get_scale_midi_notes(root_note: str, scale_type: str) -> set:
    """Get all pitch classes (0-11) in a given scale."""
    root = note_name_to_midi(root_note)
    intervals = SCALES.get(scale_type, SCALES["chromatic"])
    return {(root + i) % 12 for i in intervals}


def nearest_scale_note(midi_pitch: float, scale_notes: set) -> float:
    """Find the nearest MIDI note that belongs to the scale."""
    if len(scale_notes) == 12:
        return round(midi_pitch)

    base_note = round(midi_pitch)
    pitch_class = base_note % 12

    if pitch_class in scale_notes:
        return float(base_note)

    for offset in range(1, 7):
        if (pitch_class + offset) % 12 in scale_notes:
            return float(base_note + offset)
        if (pitch_class - offset) % 12 in scale_notes:
            return float(base_note - offset)

    return float(base_note)

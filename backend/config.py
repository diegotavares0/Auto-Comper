"""
Auto-Comper v1.0 — Configuration dataclasses.
"""

import json
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict


@dataclass
class CompRules:
    """Configurable rules for the auto-comper."""

    # Scoring weights (sum to 1.0)
    weight_pitch_stability: float = 0.30
    weight_clarity: float = 0.25
    weight_energy: float = 0.15
    weight_onset_strength: float = 0.15
    weight_noise_floor: float = 0.15

    # Segmentation
    segment_method: str = "musical"       # "musical", "fixed", "custom"
    fixed_segment_ms: float = 4000.0
    min_segment_ms: float = 2000.0
    target_segment_ms: float = 6000.0
    max_segment_ms: float = 15000.0

    # Alignment
    use_alignment: bool = True
    max_shift_ms: float = 300.0

    # Continuity
    switch_penalty: float = 0.15
    min_improvement_to_switch: float = 0.08
    max_takes_in_comp: int = 4

    # Structure (user-defined sections from Structure tab)
    custom_sections: Optional[List[Dict]] = None

    # Structure-aware comping (sections confirmed from Structure tab)
    structure_sections: Optional[List[Dict]] = None

    # Tempo/Pitch Normalization (0 = off, 100 = full correction)
    tempo_normalize_intensity: float = 0.0    # 0-100
    pitch_center_intensity: float = 0.0       # 0-100

    # Auto-trim
    auto_trim_enabled: bool = True
    trim_silence_threshold_db: float = -45.0
    trim_min_music_duration_s: float = 1.0

    # Crossfade
    crossfade_ms: float = 50.0

    # Output
    sample_rate: int = 48000
    normalize_output: bool = False
    target_lufs: float = -16.0

    def to_dict(self) -> dict:
        return asdict(self)

    def save_json(self, path: str):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_json(cls, path: str) -> "CompRules":
        with open(path) as f:
            data = json.load(f)
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class TunerConfig:
    """Configuration for the vocal tuner."""

    instrument_mode: str = "voice"      # "voice", "guitar", "auto"
    correction_amount: float = 80.0     # 0-100%
    retune_speed: float = 50.0          # 0-100
    root_note: str = "auto"             # note name or "auto"
    scale_type: str = "major"
    preserve_vibrato: bool = True
    vibrato_threshold_hz: float = 4.0
    vibrato_max_depth_cents: float = 80.0
    pitch_confidence_threshold: float = 0.6
    sample_rate: int = 48000
    hop_ms: float = 10.0
    normalize_output: bool = True
    target_lufs: float = -16.0

    def to_dict(self) -> dict:
        return asdict(self)

    def save_json(self, path: str):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_json(cls, path: str) -> "TunerConfig":
        with open(path) as f:
            data = json.load(f)
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class PresetConfig:
    """Configuration for guitar tone preset application."""

    preset_id: str = ""
    intensity: float = 80.0           # 0-100% — how strongly to apply the tone match
    use_neural: bool = False          # Whether to add neural refinement on top of DSP
    normalize_output: bool = True
    target_lufs: float = -16.0
    sample_rate: int = 48000

    # DSP internals (not user-facing)
    n_fft: int = 4096
    hop_length: int = 1024
    spectral_smoothing: int = 20      # cepstral lifter order — higher = smoother
    max_boost_db: float = 12.0        # max EQ boost at any frequency
    max_cut_db: float = 18.0          # max EQ cut at any frequency
    dynamics_match: bool = True       # match compression/dynamics
    transient_preserve: float = 0.7   # 0-1, how much to protect transients

    def to_dict(self) -> dict:
        return asdict(self)

    def save_json(self, path: str):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_json(cls, path: str) -> "PresetConfig":
        with open(path) as f:
            data = json.load(f)
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class TakeScore:
    """Score for an entire take."""
    take_idx: int
    overall: float = 0.0
    pitch_stability: float = 0.0
    clarity: float = 0.0
    energy: float = 0.0
    noise_floor: float = 0.0
    consistency: float = 0.0


@dataclass
class BlockScore:
    """Score for a block within a take."""
    take_idx: int
    block_idx: int
    start_sample: int
    end_sample: int
    pitch_stability: float = 0.0
    clarity: float = 0.0
    energy: float = 0.0
    onset_strength: float = 0.0
    noise_floor: float = 0.0
    total: float = 0.0

"""
Preset manager — CRUD operations for guitar tone presets.

Presets are stored as JSON files in a persistent directory.
Each preset optionally has an associated reference WAV file.
"""

import json
import logging
import os
import shutil
import uuid
from datetime import datetime
from typing import List, Optional

log = logging.getLogger("comper")

# Persistent storage directory (in user's home or project dir)
_PRESETS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data", "presets"
)


def _ensure_dir():
    """Ensure presets directory exists."""
    os.makedirs(_PRESETS_DIR, exist_ok=True)


def get_presets_dir() -> str:
    """Return the presets storage directory."""
    _ensure_dir()
    return _PRESETS_DIR


def save_preset(
    name: str,
    profile: dict,
    reference_audio_path: Optional[str] = None,
    preset_id: Optional[str] = None,
) -> dict:
    """
    Save a new preset to disk.

    Parameters
    ----------
    name : str
        Display name (e.g., "Phoebe Bridgers - The Garden Song").
    profile : dict
        Spectral profile from analyzer.analyze_reference().
    reference_audio_path : str, optional
        Path to the reference WAV file (will be copied to preset dir).
    preset_id : str, optional
        Override ID (for updates). Auto-generated if not provided.

    Returns
    -------
    dict : Metadata about the saved preset.
    """
    _ensure_dir()

    if not preset_id:
        preset_id = str(uuid.uuid4())[:8]

    preset_dir = os.path.join(_PRESETS_DIR, preset_id)
    os.makedirs(preset_dir, exist_ok=True)

    # Build metadata
    meta = {
        "id": preset_id,
        "name": name,
        "created_at": datetime.now().isoformat(),
        "duration_s": profile.get("duration_s", 0),
        "timbre": profile.get("timbre", {}),
        "dynamics": profile.get("dynamics", {}),
        "has_reference_audio": reference_audio_path is not None,
    }

    # Save metadata
    with open(os.path.join(preset_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # Save full profile (with spectral data)
    with open(os.path.join(preset_dir, "profile.json"), "w") as f:
        json.dump(profile, f)

    # Copy reference audio if provided
    if reference_audio_path and os.path.exists(reference_audio_path):
        ref_dest = os.path.join(preset_dir, "reference.wav")
        shutil.copy2(reference_audio_path, ref_dest)
        log.info(f"  Saved reference audio to {ref_dest}")

    log.info(f"  Preset saved: '{name}' (id={preset_id})")

    return meta


def list_presets() -> List[dict]:
    """
    List all saved presets.

    Returns list of metadata dicts sorted by creation date (newest first).
    """
    _ensure_dir()
    presets = []

    for dirname in os.listdir(_PRESETS_DIR):
        meta_path = os.path.join(_PRESETS_DIR, dirname, "meta.json")
        if os.path.exists(meta_path):
            try:
                with open(meta_path) as f:
                    meta = json.load(f)
                presets.append(meta)
            except Exception as e:
                log.warning(f"  Failed to load preset {dirname}: {e}")

    # Sort by creation date (newest first)
    presets.sort(key=lambda p: p.get("created_at", ""), reverse=True)

    return presets


def get_preset(preset_id: str) -> Optional[dict]:
    """Get preset metadata by ID."""
    meta_path = os.path.join(_PRESETS_DIR, preset_id, "meta.json")
    if not os.path.exists(meta_path):
        return None

    with open(meta_path) as f:
        return json.load(f)


def load_profile(preset_id: str) -> Optional[dict]:
    """Load the full spectral profile for a preset."""
    profile_path = os.path.join(_PRESETS_DIR, preset_id, "profile.json")
    if not os.path.exists(profile_path):
        return None

    with open(profile_path) as f:
        return json.load(f)


def get_reference_audio_path(preset_id: str) -> Optional[str]:
    """Get path to the reference WAV file if it exists."""
    ref_path = os.path.join(_PRESETS_DIR, preset_id, "reference.wav")
    return ref_path if os.path.exists(ref_path) else None


def delete_preset(preset_id: str) -> bool:
    """Delete a preset and all its files."""
    preset_dir = os.path.join(_PRESETS_DIR, preset_id)
    if not os.path.exists(preset_dir):
        return False

    shutil.rmtree(preset_dir, ignore_errors=True)
    log.info(f"  Preset deleted: {preset_id}")
    return True

"""
Audio I/O utilities — load, save, resample, mono conversion.
"""

import logging
import os
from typing import List, Tuple

import numpy as np
import soundfile as sf
import librosa

log = logging.getLogger("comper")

SUPPORTED_EXTENSIONS = ('.wav', '.flac', '.aiff', '.aif')


def load_audio_file(path: str, sr: int) -> np.ndarray:
    """Load a single audio file, convert to mono, resample if needed."""
    audio, file_sr = sf.read(path, dtype="float64")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if file_sr != sr:
        audio = librosa.resample(audio, orig_sr=file_sr, target_sr=sr)
    return audio


def load_takes_from_folder(folder: str, sr: int) -> Tuple[List[np.ndarray], int]:
    """Load all audio files from a folder."""
    wav_files = sorted([
        f for f in os.listdir(folder)
        if f.lower().endswith(SUPPORTED_EXTENSIONS)
    ])
    if not wav_files:
        raise ValueError(f"No audio files found in {folder}")

    takes = []
    for fname in wav_files:
        path = os.path.join(folder, fname)
        audio = load_audio_file(path, sr)
        takes.append(audio)
        log.info(f"  {fname}: {len(audio)/sr:.1f}s")

    return takes, sr


def load_takes_from_arrays(arrays: List[np.ndarray], sr: int) -> Tuple[List[np.ndarray], int]:
    """Use pre-loaded arrays directly, ensuring mono."""
    takes = []
    for i, audio in enumerate(arrays):
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        takes.append(audio)
        log.info(f"  Take {i+1}: {len(audio)/sr:.1f}s")
    return takes, sr


def save_audio(path: str, audio: np.ndarray, sr: int, subtype: str = "PCM_24"):
    """Save audio to file."""
    sf.write(path, audio, sr, subtype=subtype)

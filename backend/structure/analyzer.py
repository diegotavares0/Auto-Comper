"""
Structure analyzer — detect musical sections via self-similarity matrix
and novelty-based boundary detection (Foote/Serra approach).

Flow: audio → chroma features → SSM → novelty curve → boundaries → section labels
"""

import logging
from typing import List, Dict, Optional, Callable, Tuple

import numpy as np
import librosa
from scipy.ndimage import median_filter
from scipy.signal import find_peaks, argrelextrema

log = logging.getLogger("comper")

# Minimum section duration in seconds — anything shorter merges into neighbors
MIN_SECTION_S = 3.0

# Checkerboard kernel half-width in frames for novelty detection
KERNEL_SIZE = 16


def analyze_structure(audio: np.ndarray, sr: int,
                      min_section_s: float = MIN_SECTION_S,
                      progress_cb: Optional[Callable] = None) -> Dict:
    """
    Detect musical structure from a single audio take.

    Returns:
        {
            "sections": [
                {"name": "A", "start_s": 0.0, "end_s": 15.2, "label": "Intro",
                 "group": 0, "confidence": 0.85},
                ...
            ],
            "n_sections": int,
            "n_groups": int,            # distinct section types (A, B, C, ...)
            "duration_s": float,
            "novelty_curve": list,      # downsampled for frontend visualization
            "novelty_times": list,      # time axis for novelty curve
            "ssm_thumbnail": list,      # downsampled SSM for frontend heatmap
        }
    """
    def progress(pct, msg):
        if progress_cb:
            progress_cb(pct, msg)

    duration = len(audio) / sr
    progress(5, "Extraindo features harmonicas...")

    # ── Step 1: Extract chroma features ──
    hop_length = 4096  # ~85ms at 48kHz — good balance for structure
    chroma = librosa.feature.chroma_cqt(
        y=audio, sr=sr, hop_length=hop_length, n_chroma=12,
    )
    # Smooth chroma to reduce local noise (median filter over ~0.5s)
    smooth_frames = max(1, int(0.5 * sr / hop_length))
    if smooth_frames > 1:
        chroma = median_filter(chroma, size=(1, smooth_frames))

    n_frames = chroma.shape[1]
    frame_times = librosa.frames_to_time(
        np.arange(n_frames), sr=sr, hop_length=hop_length,
    )

    progress(15, "Construindo matriz de auto-similaridade...")

    # ── Step 2: Self-similarity matrix ──
    # Normalize chroma columns to unit length
    norms = np.linalg.norm(chroma, axis=0, keepdims=True) + 1e-10
    chroma_norm = chroma / norms

    # Cosine similarity matrix
    # Replace any NaN/inf from silent frames before matmul
    chroma_norm = np.nan_to_num(chroma_norm, nan=0.0, posinf=0.0, neginf=0.0)
    ssm = chroma_norm.T @ chroma_norm  # (n_frames, n_frames)
    ssm = np.nan_to_num(ssm, nan=0.0, posinf=1.0, neginf=0.0)

    # Enhance diagonal structure with path enhancement
    ssm = _enhance_ssm(ssm)

    progress(30, "Detectando mudancas harmonicas...")

    # ── Step 3: Novelty curve via checkerboard kernel ──
    novelty = _checkerboard_novelty(ssm, kernel_size=KERNEL_SIZE)

    # Smooth novelty curve
    smooth_win = max(1, int(1.0 * sr / hop_length))  # ~1 second
    if smooth_win > 1:
        novelty = np.convolve(
            novelty, np.ones(smooth_win) / smooth_win, mode='same',
        )

    progress(45, "Encontrando limites de secoes...")

    # ── Step 4: Find boundary candidates ──
    min_frames = max(1, int(min_section_s * sr / hop_length))
    boundaries = _find_boundaries(novelty, min_distance=min_frames)

    # Always include start and end
    if len(boundaries) == 0 or boundaries[0] != 0:
        boundaries = np.insert(boundaries, 0, 0)
    if boundaries[-1] != n_frames:
        boundaries = np.append(boundaries, n_frames)

    progress(60, "Agrupando secoes similares...")

    # ── Step 5: Group similar sections ──
    sections, n_groups = _group_sections(
        boundaries, ssm, frame_times, duration,
    )

    # ── Step 6: Label sections with musical names ──
    sections = _label_sections(sections, audio, sr, hop_length)

    progress(80, "Gerando dados de visualizacao...")

    # ── Step 7: Build visualization data ──
    # Downsample novelty for frontend (max ~200 points)
    max_viz_points = 200
    step = max(1, len(novelty) // max_viz_points)
    novelty_down = novelty[::step].tolist()
    times_down = frame_times[::step].tolist()

    # Downsample SSM thumbnail (max 100x100)
    ssm_thumb = _downsample_ssm(ssm, max_size=100)

    progress(95, "Estrutura detectada!")

    report = {
        "sections": sections,
        "n_sections": len(sections),
        "n_groups": n_groups,
        "duration_s": round(duration, 2),
        "novelty_curve": novelty_down,
        "novelty_times": times_down,
        "ssm_thumbnail": ssm_thumb,
    }

    log.info(f"  Estrutura: {len(sections)} secoes, {n_groups} grupos distintos")
    for s in sections:
        log.info(f"    {s['name']} ({s['label']}): "
                 f"{s['start_s']:.1f}s - {s['end_s']:.1f}s "
                 f"[grupo {s['group']}]")

    return report


# ─────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────

def _enhance_ssm(ssm: np.ndarray) -> np.ndarray:
    """Enhance SSM by adding temporal smoothing along the diagonal."""
    n = ssm.shape[0]
    enhanced = ssm.copy()

    # Apply median filter to reduce noise
    enhanced = median_filter(enhanced, size=3)

    # Symmetrize (should already be symmetric, but ensure it)
    enhanced = (enhanced + enhanced.T) / 2.0

    # Clip to [0, 1]
    enhanced = np.clip(enhanced, 0, 1)

    return enhanced


def _checkerboard_novelty(ssm: np.ndarray, kernel_size: int = 16) -> np.ndarray:
    """
    Compute novelty curve using a checkerboard kernel convolved along
    the main diagonal of the SSM. Peaks in novelty = section boundaries.
    """
    n = ssm.shape[0]
    k = min(kernel_size, n // 4)
    if k < 2:
        return np.ones(n) * 0.5

    # Build checkerboard kernel
    # [+1, -1]
    # [-1, +1]
    kernel = np.ones((2 * k, 2 * k))
    kernel[:k, :k] = 1
    kernel[k:, k:] = 1
    kernel[:k, k:] = -1
    kernel[k:, :k] = -1

    novelty = np.zeros(n)

    for i in range(k, n - k):
        # Extract local patch around diagonal position (i, i)
        r_start = max(0, i - k)
        r_end = min(n, i + k)
        c_start = max(0, i - k)
        c_end = min(n, i + k)

        patch = ssm[r_start:r_end, c_start:c_end]

        # Resize kernel to match patch if needed
        kr = min(kernel.shape[0], patch.shape[0])
        kc = min(kernel.shape[1], patch.shape[1])
        local_kernel = kernel[:kr, :kc]

        # Correlate
        if local_kernel.size > 0:
            novelty[i] = np.sum(patch[:kr, :kc] * local_kernel)

    # Normalize to [0, 1]
    nmin, nmax = novelty.min(), novelty.max()
    if nmax > nmin:
        novelty = (novelty - nmin) / (nmax - nmin)

    return novelty


def _find_boundaries(novelty: np.ndarray,
                     min_distance: int = 10) -> np.ndarray:
    """Find peaks in the novelty curve as section boundaries."""
    if len(novelty) < 3:
        return np.array([0, len(novelty)])

    # Adaptive threshold: peaks must be at least 0.3x the max novelty
    threshold = np.max(novelty) * 0.25

    # Use scipy find_peaks with constraints
    peaks, properties = find_peaks(
        novelty,
        height=threshold,
        distance=min_distance,
        prominence=threshold * 0.5,
    )

    return peaks


def _group_sections(boundaries: np.ndarray, ssm: np.ndarray,
                    frame_times: np.ndarray,
                    duration: float) -> Tuple[List[Dict], int]:
    """
    Group sections by harmonic similarity.
    Sections with similar SSM profiles get the same group label (A, B, C...).
    """
    n_sections = len(boundaries) - 1
    if n_sections <= 0:
        return [{
            "name": "A",
            "start_s": 0.0,
            "end_s": round(duration, 2),
            "group": 0,
            "confidence": 1.0,
        }], 1

    # Compute a representative vector for each section
    # (mean chroma similarity profile)
    section_profiles = []
    for i in range(n_sections):
        s = boundaries[i]
        e = boundaries[i + 1]
        if e > s:
            # Mean similarity of this section against the whole piece
            profile = np.mean(ssm[s:e, :], axis=0)
            section_profiles.append(profile)
        else:
            section_profiles.append(np.zeros(ssm.shape[1]))

    # Compute pairwise similarity between section profiles
    groups = [-1] * n_sections
    current_group = 0
    similarity_threshold = 0.75  # sections must be >75% similar to group

    for i in range(n_sections):
        if groups[i] >= 0:
            continue

        groups[i] = current_group
        ref_profile = section_profiles[i]
        ref_norm = np.linalg.norm(ref_profile) + 1e-10

        for j in range(i + 1, n_sections):
            if groups[j] >= 0:
                continue

            # Cosine similarity between section profiles
            cand_profile = section_profiles[j]
            cand_norm = np.linalg.norm(cand_profile) + 1e-10
            sim = np.dot(ref_profile, cand_profile) / (ref_norm * cand_norm)

            if sim >= similarity_threshold:
                groups[j] = current_group

        current_group += 1

    # Build section list
    group_labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    sections = []
    for i in range(n_sections):
        s_frame = boundaries[i]
        e_frame = boundaries[i + 1]

        start_s = float(frame_times[s_frame]) if s_frame < len(frame_times) else 0.0
        end_s = float(frame_times[min(e_frame, len(frame_times) - 1)])

        # For the last section, extend to actual audio end
        if i == n_sections - 1:
            end_s = round(duration, 2)

        group_idx = groups[i]
        group_name = group_labels[group_idx % len(group_labels)]

        # Count how many sections share this group
        group_count = sum(1 for g in groups if g == group_idx)

        # Confidence based on how distinct this section is from neighbors
        confidence = 1.0
        if i > 0:
            prev_profile = section_profiles[i - 1]
            curr_profile = section_profiles[i]
            pn = np.linalg.norm(prev_profile) + 1e-10
            cn = np.linalg.norm(curr_profile) + 1e-10
            diff = 1.0 - np.dot(prev_profile, curr_profile) / (pn * cn)
            confidence = min(1.0, max(0.3, diff * 2))

        sections.append({
            "name": group_name,
            "start_s": round(start_s, 2),
            "end_s": round(end_s, 2),
            "group": group_idx,
            "confidence": round(confidence, 2),
            "label": "",  # will be filled by _label_sections
        })

    n_groups = current_group
    return sections, n_groups


def _label_sections(sections: List[Dict], audio: np.ndarray,
                    sr: int, hop_length: int) -> List[Dict]:
    """
    Assign human-readable labels (Intro, Verso, Refrao, Ponte, Outro)
    based on position, energy, and group patterns.
    """
    if not sections:
        return sections

    # Compute RMS energy per section
    energies = []
    for sec in sections:
        s = int(sec["start_s"] * sr)
        e = int(sec["end_s"] * sr)
        chunk = audio[s:e]
        rms = np.sqrt(np.mean(chunk ** 2)) if len(chunk) > 0 else 0
        energies.append(rms)

    if not energies or max(energies) == 0:
        for sec in sections:
            sec["label"] = sec["name"]
        return sections

    # Normalize energies
    max_e = max(energies)
    norm_energies = [e / max_e for e in energies]

    # Heuristic labeling based on position + energy + repetition pattern
    n = len(sections)
    median_energy = np.median(norm_energies)

    # Track group occurrences
    group_first_idx = {}
    group_count = {}
    for i, sec in enumerate(sections):
        g = sec["group"]
        if g not in group_first_idx:
            group_first_idx[g] = i
        group_count[g] = group_count.get(g, 0) + 1

    # Find most repeated group (likely verse or chorus)
    most_repeated_group = max(group_count, key=group_count.get) if group_count else 0
    most_repeated_count = group_count.get(most_repeated_group, 0)

    # Find highest-energy repeated group (likely chorus)
    repeated_groups = [g for g, c in group_count.items() if c >= 2]
    chorus_group = None
    if repeated_groups:
        group_avg_energy = {}
        for i, sec in enumerate(sections):
            g = sec["group"]
            if g in repeated_groups:
                group_avg_energy.setdefault(g, []).append(norm_energies[i])
        for g in group_avg_energy:
            group_avg_energy[g] = np.mean(group_avg_energy[g])
        if group_avg_energy:
            chorus_group = max(group_avg_energy, key=group_avg_energy.get)

    # Find second most repeated group (likely verse, if chorus is found)
    verse_group = None
    if chorus_group is not None:
        other_repeated = [g for g in repeated_groups if g != chorus_group]
        if other_repeated:
            verse_group = other_repeated[0]  # first occurring
        elif most_repeated_group != chorus_group:
            verse_group = most_repeated_group

    for i, sec in enumerate(sections):
        g = sec["group"]
        e = norm_energies[i]
        pos = i / max(1, n - 1)  # 0.0 = start, 1.0 = end

        # First section with low energy = Intro
        if i == 0 and e < median_energy * 0.8 and n > 2:
            sec["label"] = "Intro"
        # Last section with lower energy = Outro
        elif i == n - 1 and e < median_energy * 0.9 and n > 2:
            sec["label"] = "Outro"
        # Chorus (high energy, repeating)
        elif g == chorus_group:
            sec["label"] = "Refrao"
        # Verse (lower energy, repeating)
        elif g == verse_group:
            sec["label"] = "Verso"
        # Single-occurrence section in the middle = Bridge/Ponte
        elif group_count.get(g, 0) == 1 and 0.3 < pos < 0.8:
            sec["label"] = "Ponte"
        # Fallback: use group letter
        else:
            sec["label"] = f"Secao {sec['name']}"

    # Add occurrence numbers for repeated labels
    label_counts = {}
    for sec in sections:
        lbl = sec["label"]
        label_counts[lbl] = label_counts.get(lbl, 0) + 1

    label_current = {}
    for sec in sections:
        lbl = sec["label"]
        if label_counts[lbl] > 1:
            label_current[lbl] = label_current.get(lbl, 0) + 1
            sec["label"] = f"{lbl} {label_current[lbl]}"

    return sections


def _downsample_ssm(ssm: np.ndarray, max_size: int = 100) -> List[List[float]]:
    """Downsample SSM to a small thumbnail for frontend heatmap."""
    n = ssm.shape[0]
    if n <= max_size:
        return [[round(float(v), 3) for v in row] for row in ssm]

    step = n / max_size
    indices = [int(i * step) for i in range(max_size)]

    thumb = ssm[np.ix_(indices, indices)]
    return [[round(float(v), 3) for v in row] for row in thumb]

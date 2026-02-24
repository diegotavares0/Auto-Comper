"""
Main auto-comping pipeline orchestrator.
"""

import logging
from typing import List, Tuple, Optional, Callable

import numpy as np

from backend.config import CompRules
from backend.engine.alignment import align_takes_xcorr
from backend.engine.segmentation import detect_blocks
from backend.engine.selection import rank_takes, select_best_blocks
from backend.engine.assembly import assemble_comp, normalize_lufs, peak_limit

log = logging.getLogger("comper")


def run_autocomp(takes: List[np.ndarray], sr: int, rules: CompRules,
                 progress_callback: Optional[Callable] = None) -> Tuple[np.ndarray, dict]:
    """
    Auto-comp pipeline — curadoria inteligente.
    Returns: (comp_audio, report_dict)
    """
    def progress(pct, msg):
        if progress_callback:
            progress_callback(pct, msg)
        log.info(f"[{pct}%] {msg}")

    n_takes = len(takes)
    progress(5, f"Carregando {n_takes} takes...")

    # Phase 0: Align
    if rules.use_alignment and n_takes > 1:
        progress(10, "Alinhando takes (cross-correlacao)...")
        takes = align_takes_xcorr(takes, sr, rules.max_shift_ms)

    # Filter outlier-short takes
    lengths = [len(t) for t in takes]
    median_len = sorted(lengths)[len(lengths) // 2]
    min_acceptable = int(median_len * 0.5)

    filtered_takes = []
    dropped = []
    for i, t in enumerate(takes):
        if len(t) >= min_acceptable:
            filtered_takes.append(t)
        else:
            dropped.append((i + 1, len(t) / sr))

    if dropped:
        for take_num, dur in dropped:
            log.info(f"  Take {take_num} descartado: {dur:.1f}s "
                     f"(muito curto vs mediana {median_len/sr:.1f}s)")

    if len(filtered_takes) < 2:
        log.warning("  Poucos takes restantes apos filtro, usando todos")
        filtered_takes = takes

    takes = filtered_takes
    n_takes = len(takes)
    min_len = min(len(t) for t in takes)
    takes = [t[:min_len] for t in takes]
    duration = min_len / sr
    progress(15, f"Takes alinhados: {duration:.1f}s, {n_takes} takes"
                 f"{f' ({len(dropped)} descartados)' if dropped else ''}")

    # Phase 1: Rank takes
    progress(20, "Ranqueando takes inteiros...")
    take_ranking = rank_takes(takes, sr, rules)
    best_take = take_ranking[0]
    progress(30, f"Melhor take: {best_take.take_idx + 1} "
                 f"(score: {best_take.overall:.3f})")

    # Phase 2: Detect musical blocks
    progress(35, "Detectando blocos musicais...")
    reference_audio = takes[best_take.take_idx]
    blocks = detect_blocks(reference_audio, sr, rules)
    progress(45, f"{len(blocks)} blocos detectados")

    # Phase 3: Score & select
    progress(50, "Analisando blocos em cada take...")
    decisions = select_best_blocks(takes, sr, blocks, take_ranking, rules,
                                   progress_callback=progress_callback)
    progress(80, "Selecao concluida")

    # Phase 4: Assemble
    progress(85, "Montando comp final...")
    comp_audio = assemble_comp(takes, decisions, sr, rules)

    # Phase 5: Normalize (optional)
    if rules.normalize_output:
        progress(90, "Normalizando...")
        comp_audio = normalize_lufs(comp_audio, sr, rules.target_lufs)

    # Phase 5b: Peak safety limiter (always)
    comp_audio = peak_limit(comp_audio)

    # Report
    take_usage = {}
    take_duration = {}
    switches = 0

    for d in decisions:
        t = d["take"]
        dur = d["duration_s"]
        take_usage[t] = take_usage.get(t, 0) + 1
        take_duration[t] = take_duration.get(t, 0) + dur
        if d.get("switched"):
            switches += 1

    total_dur = sum(take_duration.values())
    take_pct = {t: round(d / total_dur * 100, 1)
                for t, d in take_duration.items()}

    report = {
        "version": "1.0",
        "total_blocks": len(blocks),
        "duration_s": round(len(comp_audio) / sr, 2),
        "base_take": best_take.take_idx + 1,
        "base_take_score": round(best_take.overall, 3),
        "take_ranking": [
            {"take": ts.take_idx + 1, "score": round(ts.overall, 3)}
            for ts in take_ranking
        ],
        "takes_in_comp": len(take_usage),
        "take_switches": switches,
        "take_usage_blocks": take_usage,
        "take_usage_pct": take_pct,
        "decisions": decisions,
        "avg_score": round(np.mean([d["score"] for d in decisions]), 3),
    }

    progress(100, f"Comp finalizado! Base: Take {best_take.take_idx + 1}, "
                  f"{len(take_usage)} takes usados, {switches} trocas")

    return comp_audio, report

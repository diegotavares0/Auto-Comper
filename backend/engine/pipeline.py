"""
Main auto-comping pipeline orchestrator.

Supports two modes:
  - Classic: all takes truncated to same length, block-based scoring.
  - Structure-aware: detect musical sections, match across takes of different
    lengths, section-locked comping (verse vs verse, chorus vs chorus only).
"""

import logging
from typing import List, Tuple, Optional, Callable, Dict

import numpy as np

from backend.config import CompRules
from backend.engine.alignment import align_takes_xcorr
from backend.engine.segmentation import detect_blocks
from backend.engine.selection import rank_takes, select_best_blocks
from backend.engine.assembly import assemble_comp, normalize_lufs, peak_limit
from backend.engine.scoring import score_audio_chunk, compute_weighted_score

log = logging.getLogger("comper")


def run_autocomp(takes: List[np.ndarray], sr: int, rules: CompRules,
                 progress_callback: Optional[Callable] = None) -> Tuple[np.ndarray, dict]:
    """
    Auto-comp pipeline — curadoria inteligente.
    Routes to structure-aware or classic pipeline based on segment_method.
    Returns: (comp_audio, report_dict)
    """
    # Structure-aware path: handles different-length takes
    if rules.segment_method == "structure" or rules.structure_sections:
        return _run_structure_comp(takes, sr, rules, progress_callback)

    # Classic path: same-length takes, block-based scoring
    return _run_classic_comp(takes, sr, rules, progress_callback)


def _run_classic_comp(takes: List[np.ndarray], sr: int, rules: CompRules,
                      progress_callback: Optional[Callable] = None) -> Tuple[np.ndarray, dict]:
    """Original block-based comping pipeline (truncates to min length)."""

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


# ─────────────────────────────────────────────
# Structure-aware comping pipeline
# ─────────────────────────────────────────────

def _run_structure_comp(takes: List[np.ndarray], sr: int, rules: CompRules,
                        progress_callback: Optional[Callable] = None) -> Tuple[np.ndarray, dict]:
    """
    Structure-aware comping: handles takes of different lengths.
    Uses pre-analyzed sections (from Structure tab) to compare only
    matching sections across takes. Section-locked: verse 1 only competes
    with verse 1 from other takes, never with chorus from take 7.
    """
    from backend.structure.analyzer import analyze_structure
    from backend.structure.matcher import match_structure_to_takes
    from backend.engine.assembly import crossfade_join

    def progress(pct, msg):
        if progress_callback:
            progress_callback(pct, msg)
        log.info(f"[{pct}%] {msg}")

    n_takes = len(takes)
    progress(2, f"Estrutura: {n_takes} takes recebidos")

    # ── Step 1: Find the longest take as reference ──
    # (longest take most likely has the full song structure)
    take_lengths = [(i, len(t)) for i, t in enumerate(takes)]
    take_lengths.sort(key=lambda x: x[1], reverse=True)
    ref_idx = take_lengths[0][0]
    ref_audio = takes[ref_idx]
    ref_duration = len(ref_audio) / sr

    progress(5, f"Take de referencia: {ref_idx + 1} ({ref_duration:.1f}s)")

    # ── Step 2: Get or detect structure ──
    if rules.structure_sections:
        # User already confirmed structure from the Structure tab
        sections = rules.structure_sections
        log.info(f"  Usando estrutura confirmada: {len(sections)} secoes")
        progress(15, f"Estrutura confirmada: {len(sections)} secoes")
    else:
        # Auto-detect structure on the reference take
        progress(8, "Detectando estrutura automaticamente...")

        def struct_progress(pct_inner, msg):
            # Map structure detection 8% → 25%
            mapped = 8 + int(pct_inner * 0.17)
            progress(mapped, msg)

        result = analyze_structure(ref_audio, sr, progress_cb=struct_progress)
        sections = result["sections"]
        progress(25, f"Estrutura detectada: {len(sections)} secoes")

    if not sections:
        log.warning("  Nenhuma secao detectada — fallback para comp classico")
        return _run_classic_comp(takes, sr, rules, progress_callback)

    # ── Step 3: Match structure to all takes ──
    progress(28, "Mapeando estrutura para todos os takes...")

    def match_progress(pct_inner, msg):
        mapped = 28 + int(pct_inner * 0.17)
        progress(mapped, msg)

    all_take_sections = match_structure_to_takes(
        sections, ref_audio, takes, sr, progress_cb=match_progress,
    )

    progress(45, "Mapeamento concluido")

    # ── Step 4: Score each take in each section ──
    progress(48, "Avaliando qualidade por secao...")
    n_sections = len(sections)
    section_scores = []  # [section_idx][take_idx] = score

    for sec_idx, sec in enumerate(sections):
        scores_for_section = []

        for take_idx in range(n_takes):
            take_secs = all_take_sections[take_idx]

            if sec_idx >= len(take_secs) or not take_secs[sec_idx]["covered"]:
                scores_for_section.append(None)
                continue

            ts = take_secs[sec_idx]
            start = int(ts["start_s"] * sr)
            end = int(ts["end_s"] * sr)
            chunk = takes[take_idx][start:end]

            if len(chunk) < sr:  # less than 1 second
                scores_for_section.append(None)
                continue

            raw_scores = score_audio_chunk(chunk, sr)
            total = compute_weighted_score(raw_scores, rules)
            scores_for_section.append(total)

        section_scores.append(scores_for_section)

        pct = 48 + int(30 * (sec_idx + 1) / n_sections)
        progress(pct, f"Avaliando secao {sec_idx + 1}/{n_sections}: "
                      f"{sec.get('label', sec['name'])}...")

    progress(78, "Avaliacao concluida")

    # ── Step 5: Select best take per section (section-locked) ──
    progress(80, "Selecionando melhor take por secao...")
    decisions = []
    current_take = ref_idx  # start with reference take

    for sec_idx, sec in enumerate(sections):
        scores = section_scores[sec_idx]
        candidates = [
            (take_idx, score)
            for take_idx, score in enumerate(scores)
            if score is not None
        ]

        if not candidates:
            # No take covers this section — use reference if possible
            log.warning(f"  Secao {sec_idx}: nenhum take cobre esta secao")
            decisions.append({
                "block": sec_idx,
                "section_name": sec.get("label", sec["name"]),
                "take": ref_idx + 1,
                "take_idx": ref_idx,
                "start_s": sec["start_s"],
                "end_s": sec["end_s"],
                "duration_s": round(sec["end_s"] - sec["start_s"], 2),
                "score": 0.0,
                "switched": False,
                "covered": False,
            })
            continue

        # Find the best score
        candidates.sort(key=lambda x: x[1], reverse=True)
        best_take_idx, best_score = candidates[0]

        # Check if switching from current take is worth it
        current_score = scores[current_take] if scores[current_take] is not None else -1
        improvement = best_score - current_score
        switched = False

        if (best_take_idx != current_take and
                improvement >= rules.min_improvement_to_switch and
                improvement - rules.switch_penalty > 0):
            switched = True
            current_take = best_take_idx
        elif current_score < 0:
            # Current take doesn't cover this section — must switch
            switched = True
            current_take = best_take_idx

        # Get the actual time boundaries from the selected take
        selected_secs = all_take_sections[current_take]
        if sec_idx < len(selected_secs) and selected_secs[sec_idx]["covered"]:
            start_s = selected_secs[sec_idx]["start_s"]
            end_s = selected_secs[sec_idx]["end_s"]
        else:
            start_s = sec["start_s"]
            end_s = sec["end_s"]

        chosen_score = scores[current_take] if scores[current_take] is not None else best_score

        decisions.append({
            "block": sec_idx,
            "section_name": sec.get("label", sec["name"]),
            "take": current_take + 1,
            "take_idx": current_take,
            "start_s": start_s,
            "end_s": end_s,
            "duration_s": round(end_s - start_s, 2),
            "score": round(chosen_score, 3),
            "switched": switched,
            "covered": True,
        })

        status = "SWITCH" if switched else "keep"
        log.info(f"  {sec.get('label', sec['name'])}: "
                 f"Take {current_take + 1} (score={chosen_score:.3f}) {status}")

    progress(85, "Selecao por secao concluida")

    # ── Step 6: Assemble from sections ──
    progress(87, "Montando comp final por secoes...")
    crossfade_samples = int(sr * rules.crossfade_ms / 1000)
    comp_audio = np.array([], dtype=np.float64)

    for dec in decisions:
        if not dec.get("covered", True):
            # Skip uncovered sections — use silence of appropriate length
            silence_duration = dec["duration_s"]
            silence = np.zeros(int(silence_duration * sr))
            if len(comp_audio) > 0:
                comp_audio = crossfade_join(comp_audio, silence, crossfade_samples)
            else:
                comp_audio = silence
            continue

        take_idx = dec["take_idx"]
        start = int(dec["start_s"] * sr)
        end = int(dec["end_s"] * sr)
        segment = takes[take_idx][start:min(end, len(takes[take_idx]))]

        if len(segment) == 0:
            continue

        if len(comp_audio) > 0:
            comp_audio = crossfade_join(comp_audio, segment, crossfade_samples)
        else:
            comp_audio = segment

    progress(92, "Comp montado")

    # ── Step 7: Normalize and limit ──
    if rules.normalize_output:
        progress(94, "Normalizando...")
        comp_audio = normalize_lufs(comp_audio, sr, rules.target_lufs)

    comp_audio = peak_limit(comp_audio)

    # ── Report ──
    take_usage = {}
    take_duration_map = {}
    switches = 0

    for d in decisions:
        if not d.get("covered", True):
            continue
        t = d["take"]
        dur = d["duration_s"]
        take_usage[t] = take_usage.get(t, 0) + 1
        take_duration_map[t] = take_duration_map.get(t, 0) + dur
        if d.get("switched"):
            switches += 1

    total_dur = sum(take_duration_map.values()) or 1
    take_pct = {t: round(d / total_dur * 100, 1)
                for t, d in take_duration_map.items()}

    # Coverage info
    covered_sections = sum(1 for d in decisions if d.get("covered", True))
    section_names = [d.get("section_name", f"Secao {d['block']+1}") for d in decisions]

    report = {
        "version": "1.0-structure",
        "mode": "structure",
        "total_blocks": len(decisions),
        "total_sections": n_sections,
        "covered_sections": covered_sections,
        "section_names": section_names,
        "duration_s": round(len(comp_audio) / sr, 2),
        "base_take": ref_idx + 1,
        "base_take_score": 0.0,
        "take_ranking": [],
        "takes_in_comp": len(take_usage),
        "take_switches": switches,
        "take_usage_blocks": take_usage,
        "take_usage_pct": take_pct,
        "decisions": decisions,
        "avg_score": round(np.mean([d["score"] for d in decisions if d["score"] > 0]), 3) if decisions else 0.0,
        "structure": [
            {"name": s.get("label", s["name"]), "start_s": s["start_s"],
             "end_s": s["end_s"], "group": s["group"]}
            for s in sections
        ],
    }

    progress(100, f"Comp (estrutura) finalizado! "
                  f"{covered_sections}/{n_sections} secoes, "
                  f"{len(take_usage)} takes, {switches} trocas")

    return comp_audio, report

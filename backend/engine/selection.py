"""
Smart take selection — ranking + continuity-aware block selection.
"""

import logging
from typing import List, Tuple

import numpy as np

from backend.config import CompRules, TakeScore, BlockScore
from backend.engine.scoring import score_audio_chunk, compute_weighted_score

log = logging.getLogger("comper")


def rank_takes(takes: List[np.ndarray], sr: int,
               rules: CompRules) -> List[TakeScore]:
    """
    Score each take as a whole by sampling multiple chunks.
    Returns sorted list (best first).
    """
    log.info("Ranqueando takes inteiros...")
    take_scores = []

    for idx, take in enumerate(takes):
        chunk_duration = int(sr * 3)
        n_chunks = 8
        step = max(1, (len(take) - chunk_duration) // n_chunks)

        chunk_scores = []
        for i in range(n_chunks):
            start = i * step
            end = min(start + chunk_duration, len(take))
            if end - start < sr:
                continue
            chunk = take[start:end]
            scores = score_audio_chunk(chunk, sr)
            total = compute_weighted_score(scores, rules)
            chunk_scores.append({**scores, "total": total})

        if not chunk_scores:
            take_scores.append(TakeScore(take_idx=idx))
            continue

        avg_pitch = np.mean([c["pitch_stability"] for c in chunk_scores])
        avg_clarity = np.mean([c["clarity"] for c in chunk_scores])
        avg_energy = np.mean([c["energy"] for c in chunk_scores])
        avg_noise = np.mean([c["noise_floor"] for c in chunk_scores])
        avg_total = np.mean([c["total"] for c in chunk_scores])

        # Consistency bonus
        score_variance = np.std([c["total"] for c in chunk_scores])
        consistency = max(0, 1 - score_variance * 5)

        overall = avg_total * 0.85 + consistency * 0.15

        ts = TakeScore(
            take_idx=idx,
            overall=overall,
            pitch_stability=avg_pitch,
            clarity=avg_clarity,
            energy=avg_energy,
            noise_floor=avg_noise,
            consistency=consistency,
        )
        take_scores.append(ts)
        log.info(f"  Take {idx+1}: overall={overall:.3f} "
                 f"(pitch={avg_pitch:.2f} clarity={avg_clarity:.2f} "
                 f"consistency={consistency:.2f})")

    take_scores.sort(key=lambda t: t.overall, reverse=True)
    log.info(f"  Melhor take: {take_scores[0].take_idx + 1} "
             f"(score: {take_scores[0].overall:.3f})")

    return take_scores


def select_best_blocks(takes: List[np.ndarray], sr: int,
                       blocks: List[Tuple[int, int]],
                       take_ranking: List[TakeScore],
                       rules: CompRules,
                       progress_callback=None) -> List[dict]:
    """
    Select best take for each block with continuity preference.
    Starts with the best-ranked take as base and only switches
    when another take is significantly better.

    progress_callback(pct, msg) reports progress from 50→80%.
    """
    base_take_idx = take_ranking[0].take_idx
    log.info(f"Take base: {base_take_idx + 1}")

    n_blocks = len(blocks)

    # Score all blocks for all takes
    block_scores = []
    for blk_idx, (start, end) in enumerate(blocks):
        scores_for_block = []
        for take_idx, take in enumerate(takes):
            chunk = take[start:end]
            scores = score_audio_chunk(chunk, sr)
            total = compute_weighted_score(scores, rules)

            bs = BlockScore(
                take_idx=take_idx,
                block_idx=blk_idx,
                start_sample=start,
                end_sample=end,
                pitch_stability=scores.get("pitch_stability", 0),
                clarity=scores.get("clarity", 0),
                energy=scores.get("energy", 0),
                onset_strength=scores.get("onset_strength", 0),
                noise_floor=scores.get("noise_floor", 0),
                total=total,
            )
            scores_for_block.append(bs)
        block_scores.append(scores_for_block)

        # Report granular progress: 50% → 78% over the scoring loop
        if progress_callback and n_blocks > 1:
            pct = 50 + int(28 * (blk_idx + 1) / n_blocks)
            progress_callback(pct, f"Analisando bloco {blk_idx + 1}/{n_blocks}...")

    # Decision pass
    decisions = []
    current_take = base_take_idx
    takes_used = {base_take_idx}

    for blk_idx, scores_for_block in enumerate(block_scores):
        base_score = scores_for_block[current_take].total

        best_alt = None
        best_alt_score = 0
        for bs in scores_for_block:
            if bs.take_idx == current_take:
                continue
            if bs.total > best_alt_score:
                best_alt = bs
                best_alt_score = bs.total

        switch = False
        chosen = scores_for_block[current_take]

        if best_alt is not None:
            improvement = best_alt_score - base_score
            effective_improvement = improvement - rules.switch_penalty

            if (improvement >= rules.min_improvement_to_switch and
                    effective_improvement > 0 and
                    (best_alt.take_idx in takes_used or
                     len(takes_used) < rules.max_takes_in_comp)):
                switch = True
                chosen = best_alt
                takes_used.add(best_alt.take_idx)
                current_take = best_alt.take_idx

        start, end = blocks[blk_idx]
        decisions.append({
            "block": blk_idx,
            "take": chosen.take_idx + 1,
            "take_idx": chosen.take_idx,
            "start_s": start / sr,
            "end_s": end / sr,
            "duration_s": (end - start) / sr,
            "score": round(chosen.total, 3),
            "switched": switch,
            "pitch_stability": round(chosen.pitch_stability, 3),
            "clarity": round(chosen.clarity, 3),
        })

        status = "SWITCH" if switch else "keep"
        log.info(f"  Bloco {blk_idx+1}: Take {chosen.take_idx+1} "
                 f"(score={chosen.total:.3f}) {status}")

    log.info(f"  Takes usados no comp: {sorted(t+1 for t in takes_used)}")
    return decisions

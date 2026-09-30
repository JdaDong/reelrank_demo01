"""精排：GBDT 预估 pCTR，融合质量与新鲜度多目标，百级 → Top-N。"""

from __future__ import annotations

import time

import numpy as np

from reelrank.config import Settings, settings as global_settings
from reelrank.logging_utils import get_logger
from reelrank.rank.coarse import build_feature_frame
from reelrank.recall.multi_recall import Candidate
from reelrank.serving.schemas import StageTrace

logger = get_logger("reelrank.rank.fine")


def run_fine(state, candidates: list[Candidate], profile: dict, settings: Settings | None = None, topk: int | None = None) -> tuple[list[Candidate], StageTrace]:
    settings = settings or global_settings
    started = time.perf_counter()
    cfg = settings.funnel.fine
    if not candidates:
        return [], StageTrace(stage="fine", input_count=0, output_count=0, latency_ms=0.0, cut_reason="粗排结果为空")

    frame = build_feature_frame(state, candidates, profile)
    pctr = state.fine.predict(frame)
    quality = (frame["quality_bayes"].to_numpy(dtype="float32") / 10.0).clip(0.0, 1.0)
    freshness = frame["freshness"].to_numpy(dtype="float32") if "freshness" in frame else np.zeros(len(frame), dtype="float32")

    final = float(cfg.w_pctr) * pctr + float(cfg.w_quality) * quality + float(cfg.w_freshness) * freshness

    for candidate, p, score in zip(candidates, pctr, final):
        candidate.scores["pctr"] = float(p)
        candidate.scores["fine_score"] = float(score)

    candidates = sorted(candidates, key=lambda c: c.scores.get("fine_score", 0.0), reverse=True)
    topk = int(topk or cfg.topk)
    kept = candidates[:topk]
    for candidate in candidates[topk:]:
        candidate.scores["cut"] = 1.0

    limit = int(settings.serving.trace_items)
    trace = StageTrace(
        stage="fine",
        input_count=len(candidates),
        output_count=len(kept),
        latency_ms=round((time.perf_counter() - started) * 1000, 3),
        cut_reason=f"按 fine_score（{cfg.w_pctr}×pCTR + {cfg.w_quality}×质量 + {cfg.w_freshness}×新鲜度）截断至 {topk}",
        items=[c.to_item(state.title_of(c.movie_id), rank=i) for i, c in enumerate(kept[:limit])],
    )
    return kept, trace

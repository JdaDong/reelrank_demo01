"""粗排：LR 预估 + 双塔内积 + 贝叶斯质量分融合，千级 → 百级。"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from reelrank.config import Settings, settings as global_settings
from reelrank.features.registry import CROSS_FEATURES, ITEM_FEATURES, USER_FEATURES, cross_features
from reelrank.logging_utils import get_logger
from reelrank.recall.multi_recall import Candidate
from reelrank.serving.schemas import StageTrace

logger = get_logger("reelrank.rank.coarse")

RECALL_SCORE_KEYS = ("bm25_score", "vector_score", "itemcf_score", "hot_score")


def build_feature_frame(state, candidates: list[Candidate], profile: dict) -> pd.DataFrame:
    """把候选 + 用户画像拼成模型输入（召回信号 + 影片 + 用户 + 交叉特征）。

    影片侧特征用矩阵切片向量化构造，交叉特征逐候选计算（口径与训练完全一致）。
    """
    movie_ids = [int(c.movie_id) for c in candidates]
    n = len(movie_ids)
    data: dict[str, np.ndarray] = {"movie_id": np.asarray(movie_ids, dtype=np.int64)}

    for key in RECALL_SCORE_KEYS:
        data[key] = np.fromiter((float(c.scores.get(key, 0.0)) for c in candidates), dtype=np.float32, count=n)

    item_block = state.item_matrix(movie_ids)
    for column, name in enumerate(ITEM_FEATURES):
        data[name] = item_block[:, column]

    for name in USER_FEATURES:
        data[name] = np.full(n, float(profile.get(name, 0.0)), dtype=np.float32)

    cross_rows = [cross_features(profile, state.cross_inputs.get(mid, {})) for mid in movie_ids]
    for name in CROSS_FEATURES:
        data[name] = np.fromiter((float(row[name]) for row in cross_rows), dtype=np.float32, count=n)

    return pd.DataFrame(data)


def run_coarse(state, candidates: list[Candidate], profile: dict, settings: Settings | None = None, topk: int | None = None) -> tuple[list[Candidate], StageTrace]:
    settings = settings or global_settings
    started = time.perf_counter()
    cfg = settings.funnel.coarse
    if not candidates:
        return [], StageTrace(stage="coarse", input_count=0, output_count=0, latency_ms=0.0, cut_reason="召回为空")

    frame = build_feature_frame(state, candidates, profile)
    lr_score = state.coarse.predict(frame)
    tower_score = frame["vector_score"].to_numpy(dtype="float32")
    quality_score = (frame["quality_bayes"].to_numpy(dtype="float32") / 10.0).clip(0.0, 1.0)

    final = (
        float(cfg.w_lr) * lr_score
        + float(cfg.w_tower) * tower_score
        + float(cfg.w_quality) * quality_score
    )

    for candidate, lr, final_score in zip(candidates, lr_score, final):
        candidate.scores["lr_score"] = float(lr)
        candidate.scores["coarse_score"] = float(final_score)

    candidates = sorted(candidates, key=lambda c: c.scores.get("coarse_score", 0.0), reverse=True)
    topk = int(topk or cfg.topk)
    cut_reason = f"按 coarse_score 截断至 {topk}"
    kept = candidates[:topk]
    for candidate in candidates[topk:]:
        candidate.scores["cut"] = 1.0
        candidate.scores["cut_reason"] = 0.0

    limit = int(settings.serving.trace_items)
    trace = StageTrace(
        stage="coarse",
        input_count=len(candidates),
        output_count=len(kept),
        latency_ms=round((time.perf_counter() - started) * 1000, 3),
        cut_reason=cut_reason,
        items=[c.to_item(state.title_of(c.movie_id), rank=i) for i, c in enumerate(kept[:limit])],
    )
    return kept, trace

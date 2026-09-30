"""重排：MMR 多样性打散 + 同导演/同类型窗口去重 + 探索流量 + 质量门槛。"""

from __future__ import annotations

import random
import time

from reelrank.config import Settings, settings as global_settings
from reelrank.logging_utils import get_logger
from reelrank.recall.multi_recall import Candidate
from reelrank.serving.schemas import StageTrace

logger = get_logger("reelrank.rank.rerank")


def _jaccard(left: set, right: set) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def item_similarity(state, left: int, right: int) -> float:
    """重排用的内容相似度：类型 + 关键词 + 同导演。"""
    a = state.cross_inputs.get(int(left), {})
    b = state.cross_inputs.get(int(right), {})
    if not a or not b:
        return 0.0
    genre = _jaccard(set(a.get("genre_names") or []), set(b.get("genre_names") or []))
    keyword = _jaccard(set(a.get("keyword_names") or []), set(b.get("keyword_names") or []))
    same_director = 1.0 if set(a.get("director_ids") or []) & set(b.get("director_ids") or []) else 0.0
    return 0.5 * genre + 0.3 * keyword + 0.2 * same_director


def run_rerank(state, candidates: list[Candidate], settings: Settings | None = None, topk: int | None = None) -> tuple[list[Candidate], StageTrace]:
    settings = settings or global_settings
    started = time.perf_counter()
    cfg = settings.funnel.rerank
    topk = int(topk or cfg.topk)
    if not candidates:
        return [], StageTrace(stage="rerank", input_count=0, output_count=0, latency_ms=0.0, cut_reason="精排结果为空")

    # 1) 质量门槛过滤
    qualified: list[Candidate] = []
    blocked: list[Candidate] = []
    min_vote = float(cfg.min_vote_average)
    for candidate in candidates:
        vote_average = float(state.item_raw.get(int(candidate.movie_id), {}).get("vote_average", 10.0))
        if vote_average >= min_vote:
            qualified.append(candidate)
        else:
            candidate.scores["cut"] = 1.0
            blocked.append(candidate)
    if not qualified:
        qualified = candidates[:topk]

    # 2) MMR 贪心 + 窗口约束
    window = int(cfg.window_size)
    max_same_genre = int(cfg.max_same_genre)
    max_same_director = int(cfg.max_same_director)
    lam = float(cfg.mmr_lambda)

    selected: list[Candidate] = []
    pool = list(qualified)
    while pool and len(selected) < topk:
        best, best_value, best_reason = None, None, ""
        for candidate in pool:
            recent = selected[-window:]
            genres = set(state.cross_inputs.get(int(candidate.movie_id), {}).get("genre_names") or [])
            directors = set(state.cross_inputs.get(int(candidate.movie_id), {}).get("director_ids") or [])

            genre_count = sum(1 for s in recent if genres & set(state.cross_inputs.get(int(s.movie_id), {}).get("genre_names") or []))
            director_count = sum(1 for s in recent if directors & set(state.cross_inputs.get(int(s.movie_id), {}).get("director_ids") or []))
            if genre_count >= max_same_genre:
                continue
            if director_count >= max_same_director:
                continue

            redundancy = max((item_similarity(state, candidate.movie_id, s.movie_id) for s in recent), default=0.0)
            value = lam * float(candidate.scores.get("fine_score", 0.0)) - (1 - lam) * redundancy
            if best_value is None or value > best_value:
                best, best_value = candidate, value

        if best is None:  # 约束过严时退化为按分取
            best = pool[0]
        pool.remove(best)
        selected.append(best)

    # 3) 探索流量：用未入选候选替换尾部若干位，保持结果多样性
    rng = random.Random(int(cfg.explore_seed))
    explore_slots = int(topk * float(cfg.explore_ratio))
    if explore_slots > 0 and len(pool) >= explore_slots:
        for _ in range(explore_slots):
            if not pool or not selected:
                break
            pick = rng.choice(pool)
            pool.remove(pick)
            replaced = selected.pop()
            pool.append(replaced)
            pick.sources.append("explore")
            pick.scores["explore"] = 1.0
            selected.append(pick)

    for rank, candidate in enumerate(selected):
        candidate.scores["rank_before"] = float(candidates.index(candidate) if candidate in candidates else rank)
    for candidate in pool:
        candidate.scores["cut"] = 1.0

    limit = int(settings.serving.trace_items)
    trace = StageTrace(
        stage="rerank",
        input_count=len(candidates),
        output_count=len(selected),
        latency_ms=round((time.perf_counter() - started) * 1000, 3),
        cut_reason=(
            f"质量门槛过滤 {len(blocked)} 条 → MMR(λ={lam}) 打散 + 窗口去重(同类型≤{max_same_genre}, 同导演≤{max_same_director}) "
            f"→ 探索替换 {explore_slots} 条 → 输出 {len(selected)}"
        ),
        items=[
            c.to_item(state.title_of(c.movie_id), rank=i, cut_reason=None)
            for i, c in enumerate(selected[:limit])
        ],
    )
    return selected, trace

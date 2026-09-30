"""四层漏斗编排：召回 → 粗排 → 精排 → 重排 → 广告竞价混排，产出 FunnelTrace。"""

from __future__ import annotations

import time
import uuid

from reelrank.ads.bidding import run_auction
from reelrank.ads.mixer import mix_results
from reelrank.config import Settings, settings as global_settings
from reelrank.logging_utils import get_logger
from reelrank.rank.coarse import run_coarse
from reelrank.rank.fine import run_fine
from reelrank.rank.rerank import run_rerank
from reelrank.recall.multi_recall import MultiRecall
from reelrank.serving.schemas import FunnelTrace, RankResponse, ResultItem
from reelrank.serving.state import RankState

logger = get_logger("reelrank.serving.funnel")


def _context_genres(state: RankState, scene: str, query: str | None, profile: dict) -> list[str]:
    """广告定向上下文：搜索场景取文本命中影片的类型，推荐场景取用户偏好类型。"""
    if scene == "search" and query:
        genres: list[str] = []
        for movie_id, _ in state.bm25.search(query, topk=10):
            genres.extend(state.cross_inputs.get(int(movie_id), {}).get("genre_names") or [])
        if genres:
            seen, ordered = set(), []
            for genre in genres:
                if genre not in seen:
                    seen.add(genre)
                    ordered.append(genre)
            return ordered[:4]
    return list(profile.get("top_genres") or [])[:4]


def run_funnel(
    state: RankState,
    scene: str = "search",
    query: str | None = None,
    user_id: int | None = None,
    session_id: str | None = None,
    topk: int | None = None,
    ads_enabled: bool | None = None,
    recall_limit: int | None = None,
    coarse_topk: int | None = None,
    fine_topk: int | None = None,
    with_trace: bool | None = None,
    settings: Settings | None = None,
) -> RankResponse:
    """执行一次完整的排序请求，返回结果与（可选的）漏斗 trace。"""
    settings = settings or global_settings
    started = time.perf_counter()
    request_id = uuid.uuid4().hex[:12]
    trace_enabled = bool(settings.serving.trace_enabled) if with_trace is None else bool(with_trace)
    ads_on = bool(settings.ads.enabled) if ads_enabled is None else bool(ads_enabled)

    profile = state.profile(user_id)
    history = state.history(user_id)
    trace_items_limit = int(settings.serving.trace_items)

    # ---------------- 召回 ----------------
    recall = MultiRecall(state, settings)
    candidates, recall_trace = recall.recall(query=query, user_id=user_id, history=history, merge_limit=recall_limit)
    recall_trace.items = [c.to_item(state.title_of(c.movie_id), rank=i) for i, c in enumerate(candidates[:trace_items_limit])]
    stages = [recall_trace]

    # ---------------- 粗排 ----------------
    candidates, coarse_trace = run_coarse(state, candidates, profile, settings, topk=coarse_topk)
    stages.append(coarse_trace)

    # ---------------- 精排 ----------------
    candidates, fine_trace = run_fine(state, candidates, profile, settings, topk=fine_topk)
    stages.append(fine_trace)

    # ---------------- 重排 ----------------
    final_candidates, rerank_trace = run_rerank(state, candidates, settings, topk=topk)
    stages.append(rerank_trace)

    # ---------------- 广告竞价 ----------------
    ad_winners, ad_blocked = [], []
    if ads_on:
        slots = [int(s) for s in settings.ads.slots]
        eligible = state.ad_index.eligible(_context_genres(state, scene, query, profile), limit=60)
        session_counts = state.session_counts(session_id or request_id)
        ad_winners, ad_blocked, ads_trace = run_auction(state, eligible, profile, session_counts, slots, settings)
        stages.append(ads_trace)

    # ---------------- 混排 ----------------
    rows, removed = mix_results(final_candidates, ad_winners, [int(s) for s in settings.ads.slots])

    items: list[ResultItem] = []
    for position, row in enumerate(rows):
        card = state.movie_card(row.movie_id)
        if row.is_ad and row.ad is not None:
            items.append(
                ResultItem(
                    **card, score=float(row.ad.ecpm), pctr=float(row.ad.pctr),
                    sources=["ads"], is_ad=True, ad=row.ad, rank_before=None,
                )
            )
        else:
            candidate = row.candidate
            scores = candidate.scores if candidate else {}
            items.append(
                ResultItem(
                    **card,
                    score=round(float(scores.get("fine_score", 0.0)), 5),
                    pctr=round(float(scores.get("pctr", 0.0)), 5),
                    sources=list(candidate.sources) if candidate else [],
                    rank_before=int(scores.get("rank_before", position)) if candidate else None,
                )
            )

    total_latency = round((time.perf_counter() - started) * 1000, 3)
    trace = None
    if trace_enabled:
        trace = FunnelTrace(
            request_id=request_id,
            scene=scene,
            query=query,
            user_id=user_id,
            stages=stages,
            ads=ad_winners + ad_blocked,
            total_latency_ms=total_latency,
            config={
                "recall_limit": int(recall_limit or settings.funnel.recall.merge_limit),
                "coarse_topk": int(coarse_topk or settings.funnel.coarse.topk),
                "fine_topk": int(fine_topk or settings.funnel.fine.topk),
                "rerank_topk": int(topk or settings.funnel.rerank.topk),
                "ads_enabled": int(ads_on),
                "removed_by_ad": len(removed),
            },
        )

    logger.info(
        "请求 %s | scene=%s | 召回 %s → 粗排 %s → 精排 %s → 重排 %s | 广告 %s | 耗时 %.1fms",
        request_id, scene, stages[0].output_count, stages[1].output_count,
        stages[2].output_count, stages[3].output_count, len(ad_winners), total_latency,
    )
    return RankResponse(
        request_id=request_id, scene=scene, query=query, user_id=user_id,
        items=items, trace=trace, total_latency_ms=total_latency,
    )

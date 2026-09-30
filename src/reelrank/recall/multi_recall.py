"""多路召回：BM25 / 向量 / ItemCF / 热门规则，融合去重并标记来源。"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from reelrank.config import Settings, settings as global_settings
from reelrank.logging_utils import get_logger
from reelrank.serving.schemas import StageTrace

logger = get_logger("reelrank.recall")


@dataclass
class Candidate:
    movie_id: int
    scores: dict[str, float] = field(default_factory=dict)
    sources: list[str] = field(default_factory=list)

    @property
    def merge_score(self) -> float:
        return max(self.scores.values()) if self.scores else 0.0

    def to_item(self, title: str, rank: int | None = None, cut_reason: str | None = None):
        from reelrank.serving.schemas import CandidateItem

        return CandidateItem(
            movie_id=self.movie_id,
            title=title,
            scores={k: round(float(v), 5) for k, v in self.scores.items()},
            sources=list(self.sources),
            rank=rank,
            cut_reason=cut_reason,
        )


class MultiRecall:
    """统一召回入口，输出千级候选并记录每一路的命中量。"""

    def __init__(self, state, settings: Settings | None = None):
        self.state = state
        self.settings = settings or global_settings
        self.hot_ranking = state.hot_ranking()

    # ---------------- 单路召回 ----------------
    def _route_bm25(self, query: str | None, topk: int) -> list[tuple[int, float]]:
        if not query:
            return []
        return self.state.bm25.search(query, topk=topk)

    def _route_vector(self, query: str | None, history: list[int], exclude: set[int], topk: int) -> list[tuple[int, float]]:
        if query:
            vector = self.state.vector.query_vector(query, self.state.bm25, topn=20)
        elif history:
            vector = self.state.vector.user_vector([(mid, 1.0) for mid in history[-30:]])
        else:
            vector = self.state.vector.global_vector
        return self.state.vector.search(vector, topk=topk, exclude=exclude)

    def _route_itemcf(self, query: str | None, history: list[int], exclude: set[int], topk: int) -> list[tuple[int, float]]:
        if query:
            triggers = [(mid, score) for mid, score in self.state.bm25.search(query, topk=10)]
        elif history:
            triggers = [(mid, 1.0 - 0.4 * i / max(1, len(history))) for i, mid in enumerate(history[-20:][::-1])]
        else:
            return []
        return self.state.itemcf.recall(triggers, topk=topk, exclude=exclude)

    def _route_hot(self, query: str | None, topk: int) -> list[tuple[int, float]]:
        return self.hot_ranking[:topk]

    # ---------------- 融合 ----------------
    def recall(
        self,
        query: str | None = None,
        user_id: int | None = None,
        history: list[int] | None = None,
        merge_limit: int | None = None,
    ) -> tuple[list[Candidate], StageTrace]:
        started = time.perf_counter()
        cfg = self.settings.funnel.recall
        history = history or []
        exclude = set(history) if not query else set()

        routes = {
            "bm25": self._route_bm25(query, int(cfg.bm25_topk)),
            "vector": self._route_vector(query, history, exclude, int(cfg.vector_topk)),
            "itemcf": self._route_itemcf(query, history, exclude, int(cfg.itemcf_topk)),
            "hot": self._route_hot(query, int(cfg.hot_topk)),
        }

        merged: dict[int, Candidate] = {}
        sources_count: dict[str, int] = {}
        for route, hits in routes.items():
            sources_count[route] = len(hits)
            for movie_id, score in hits:
                candidate = merged.get(movie_id)
                if candidate is None:
                    candidate = Candidate(movie_id=movie_id)
                    merged[movie_id] = candidate
                candidate.scores[f"{route}_score"] = max(candidate.scores.get(f"{route}_score", 0.0), float(score))
                if route not in candidate.sources:
                    candidate.sources.append(route)

        candidates = sorted(merged.values(), key=lambda c: c.merge_score, reverse=True)
        limit = int(merge_limit or cfg.merge_limit)
        cut_reason = f"多路融合去重后按 merge_score 截断至 {limit}"
        if len(candidates) > limit:
            dropped = candidates[limit:]
            candidates = candidates[:limit]
            for candidate in dropped:
                candidate.scores["cut"] = 1.0
        else:
            cut_reason = "候选未超过 merge_limit，无需截断"

        trace = StageTrace(
            stage="recall",
            input_count=sum(sources_count.values()),
            output_count=len(candidates),
            latency_ms=round((time.perf_counter() - started) * 1000, 3),
            cut_reason=cut_reason,
            sources=sources_count,
        )
        return candidates, trace

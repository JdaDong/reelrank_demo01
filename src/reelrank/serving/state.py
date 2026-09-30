"""进程内状态：索引、模型、特征、广告库的一次性加载与缓存。"""

from __future__ import annotations

import json
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd

from reelrank.config import Settings, settings as global_settings
from reelrank.features.item_features import build_item_features
from reelrank.features.registry import ITEM_FEATURES, USER_FEATURES
from reelrank.features.user_features import build_user_features, default_profile
from reelrank.logging_utils import get_logger
from reelrank.models.bm25_index import BM25Index
from reelrank.models.coarse_ranker import CoarseRanker
from reelrank.models.fine_ranker import FineRanker
from reelrank.models.item_cf import ItemCF
from reelrank.models.vector_index import VectorIndex

logger = get_logger("reelrank.serving.state")


def _as_list(value) -> list:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return list(value)


class RankState:
    """服务启动时加载全部产物，请求路径上零 IO。"""

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or global_settings
        self.items: pd.DataFrame | None = None
        self.users: pd.DataFrame | None = None
        self.item_raw: dict[int, dict] = {}
        self.item_features: dict[int, dict] = {}
        self.cross_inputs: dict[int, dict] = {}
        self._item_matrix: np.ndarray | None = None
        self._item_pos: dict[int, int] = {}
        self.profiles: dict[int, dict] = {}
        self.user_history: dict[int, list[int]] = {}
        self._titles: dict[int, str] = {}
        self._hot: list[tuple[int, float]] | None = None
        self._query_cache: OrderedDict = OrderedDict()
        self.sessions: dict[str, dict[int, int]] = {}

    # ---------------- 加载 ----------------
    @classmethod
    def load(cls, settings: Settings | None = None) -> "RankState":
        settings = settings or global_settings
        started = time.perf_counter()
        artifacts = settings.path("artifacts")
        state = cls(settings)

        items_path = artifacts / "item_features.parquet"
        state.items = pd.read_parquet(items_path) if items_path.exists() else build_item_features(settings, persist=True)
        users_path = artifacts / "user_features.parquet"
        state.users = pd.read_parquet(users_path) if users_path.exists() else build_user_features(settings, state.items)

        for row in state.items.to_dict("records"):
            movie_id = int(row["movie_id"])
            state.item_raw[movie_id] = row
            state.item_features[movie_id] = {name: float(row[name]) for name in ITEM_FEATURES}
            state.cross_inputs[movie_id] = {
                "genre_names": _as_list(row.get("genre_names")),
                "keyword_names": _as_list(row.get("keyword_names")),
                "director_ids": [int(x) for x in _as_list(row.get("director_ids"))],
                "cast_ids": [int(x) for x in _as_list(row.get("cast_ids"))],
                "year": int(row.get("year") or 2000),
                "quality_bayes": float(row.get("quality_bayes") or 0.0),
            }
            state._titles[movie_id] = str(row.get("title") or "")
            state._item_pos[movie_id] = len(state._item_pos)

        for row in state.users.to_dict("records"):
            user_id = int(row["user_id"])
            state.profiles[user_id] = {
                **{name: float(row[name]) for name in USER_FEATURES},
                "genre_affinity": json.loads(row["genre_affinity"] or "{}"),
                "keyword_affinity": json.loads(row["keyword_affinity"] or "{}"),
                "director_affinity": json.loads(row["director_affinity"] or "{}"),
                "cast_affinity": json.loads(row["cast_affinity"] or "{}"),
                "top_genres": _as_list(row.get("top_genres")),
            }
            state.user_history[user_id] = [int(x) for x in _as_list(row.get("history_positive"))]

        state.bm25 = BM25Index.load(artifacts / "bm25_index.joblib")
        state.vector = VectorIndex.load(artifacts / "vector_index.npz")
        state.itemcf = ItemCF.load(artifacts / "item_cf.joblib")
        state.coarse = CoarseRanker.load(artifacts / "coarse_ranker.joblib")
        state.fine = FineRanker.load(artifacts / "fine_ranker.joblib")

        from reelrank.ads.ad_index import AdIndex

        ads_path = artifacts / "ads.json"
        state.ad_index = AdIndex.load(ads_path, settings) if ads_path.exists() else AdIndex.build(settings, state.items)

        state.default_profile = default_profile(state.users)
        logger.info(
            "状态加载完成：%s 部影片 / %s 个用户 / %s 个广告，耗时 %.2fs",
            len(state.item_raw), len(state.profiles), len(state.ad_index.ads), time.perf_counter() - started,
        )
        return state

    # ---------------- 查询工具 ----------------
    def title_of(self, movie_id: int) -> str:
        return self._titles.get(int(movie_id), "")

    def profile(self, user_id: int | None) -> dict:
        if user_id is None:
            return self.default_profile
        return self.profiles.get(int(user_id), self.default_profile)

    def history(self, user_id: int | None) -> list[int]:
        if user_id is None:
            return []
        return self.user_history.get(int(user_id), [])

    def hot_ranking(self) -> list[tuple[int, float]]:
        """热门召回：热度 × 质量 × 新鲜度的加权排名（归一化 0~1）。"""
        if self._hot is not None:
            return self._hot
        frame = self.items.copy()
        popularity = frame["popularity_log"].astype(float)
        quality = (frame["quality_bayes"].astype(float) / 10.0).clip(0, 1)
        freshness = frame["freshness"].astype(float)
        score = 0.55 * popularity + 0.30 * quality + 0.15 * freshness
        ordered = sorted(zip(frame["movie_id"].astype(int).tolist(), score.tolist()), key=lambda kv: kv[1], reverse=True)
        max_score = ordered[0][1] or 1.0
        self._hot = [(int(mid), float(score / max_score)) for mid, score in ordered]
        return self._hot

    def item_matrix(self, movie_ids: list[int]) -> np.ndarray:
        """按 movie_id 顺序取出影片数值特征矩阵（向量化，避免逐行构造 DataFrame）。"""
        if self._item_matrix is None:
            self._item_matrix = np.asarray(
                [[self.item_features[mid].get(name, 0.0) for name in ITEM_FEATURES] for mid in self._item_pos],
                dtype=np.float32,
            )
        idx = np.asarray([self._item_pos.get(int(m), -1) for m in movie_ids], dtype=np.int64)
        result = np.zeros((len(movie_ids), len(ITEM_FEATURES)), dtype=np.float32)
        valid = idx >= 0
        if valid.any():
            result[valid] = self._item_matrix[idx[valid]]
        return result

    def movie_card(self, movie_id: int) -> dict:
        row = self.item_raw.get(int(movie_id), {})
        return {
            "movie_id": int(movie_id),
            "title": str(row.get("title") or ""),
            "year": int(row.get("year") or 0),
            "genres": _as_list(row.get("genre_names")),
            "vote_average": float(row.get("vote_average") or 0.0),
            "popularity": float(row.get("popularity") or 0.0),
            "poster_path": str(row.get("poster_path") or ""),
            "overview": str(row.get("overview") or "")[:220],
        }

    # ---------------- 缓存 ----------------
    def cache_get(self, key: str):
        return self._query_cache.get(key)

    def cache_put(self, key: str, value) -> None:
        limit = int(self.settings.serving.query_cache_size)
        self._query_cache[key] = value
        self._query_cache.move_to_end(key)
        while len(self._query_cache) > limit:
            self._query_cache.popitem(last=False)

    def session_counts(self, session_id: str) -> dict[int, int]:
        return self.sessions.setdefault(session_id or "default", {})

    def reload_artifacts(self) -> None:
        self._hot = None
        artifacts = self.settings.path("artifacts")
        if (artifacts / "fine_ranker.joblib").exists():
            self.fine = FineRanker.load(artifacts / "fine_ranker.joblib")
        if (artifacts / "coarse_ranker.joblib").exists():
            self.coarse = CoarseRanker.load(artifacts / "coarse_ranker.joblib")
        logger.info("模型产物已热更新")


_STATE: RankState | None = None


def get_state(settings: Settings | None = None, reload: bool = False) -> RankState:
    """进程级单例状态。"""
    global _STATE
    if _STATE is None or reload:
        _STATE = RankState.load(settings)
    return _STATE

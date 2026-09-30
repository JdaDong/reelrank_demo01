"""特征注册中心：统一特征口径，避免粗排/精排特征漂移。

所有排序模型必须从这里取特征名与矩阵构造函数，禁止在模型内部各自拼装。
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import duckdb
import pandas as pd

from reelrank.config import Settings, settings as global_settings

# 召回层产出的信号（粗排/精排共同使用）
RECALL_FEATURES = ["bm25_score", "vector_score", "itemcf_score", "hot_score", "recall_routes"]

# 影片侧特征
ITEM_FEATURES = [
    "quality_bayes",
    "popularity_log",
    "vote_count_log",
    "freshness",
    "runtime_norm",
    "revenue_log",
    "roi_log",
    "n_genres",
    "n_keywords",
    "n_cast",
]

# 用户侧特征
USER_FEATURES = [
    "user_activity",
    "user_click_rate",
    "user_avg_rating",
    "user_era_mean",
    "user_era_std",
    "user_genre_entropy",
]

# 用户 × 影片 交叉特征
CROSS_FEATURES = [
    "genre_match",
    "keyword_match",
    "director_match",
    "cast_match",
    "era_gap",
    "quality_gap",
]

COARSE_FEATURES = RECALL_FEATURES[:4] + ITEM_FEATURES
FINE_FEATURES = RECALL_FEATURES[:4] + ITEM_FEATURES + USER_FEATURES + CROSS_FEATURES

FEATURE_DOCS = {
    "bm25_score": "查询与影片文本的 BM25 相关性（归一化到 0~1）",
    "vector_score": "用户/查询向量与影片向量的余弦相似度（0~1）",
    "itemcf_score": "ItemCF 共现 + TMDB 相似图融合得分（0~1）",
    "hot_score": "热度/趋势召回分（0~1）",
    "quality_bayes": "贝叶斯平滑后的评分质量分",
    "popularity_log": "log1p(popularity)",
    "vote_count_log": "log1p(vote_count)",
    "freshness": "按半衰期衰减的新鲜度（0~1）",
    "runtime_norm": "片长归一化（0~1）",
    "revenue_log": "log1p(revenue)",
    "roi_log": "log1p(revenue/(budget+1))",
    "n_genres": "类型数量",
    "n_keywords": "关键词数量",
    "n_cast": "主演数量",
    "user_activity": "用户历史行为数（对数）",
    "user_click_rate": "用户历史点击率",
    "user_avg_rating": "用户历史平均评分",
    "user_era_mean": "用户偏好年代均值（归一化）",
    "user_era_std": "用户偏好年代标准差（归一化）",
    "user_genre_entropy": "用户类型偏好分布的熵（越大越发散）",
    "genre_match": "用户类型偏好与影片类型的加权匹配度",
    "keyword_match": "用户关键词偏好与影片关键词的加权匹配度",
    "director_match": "是否命中用户偏好导演",
    "cast_match": "影片主演命中用户偏好的比例",
    "era_gap": "影片年代与用户偏好年代的差距（越小越好，已取负）",
    "quality_gap": "影片质量分与用户平均评分的匹配度（越接近越好，已取负）",
}


def load_warehouse(settings: Settings | None = None, tables: Iterable[str] | None = None) -> dict[str, pd.DataFrame]:
    """从 DuckDB 读取数仓表为 DataFrame。"""
    settings = settings or global_settings
    wanted = list(tables) if tables else [
        "dim_movie",
        "bridge_movie_genre",
        "bridge_movie_keyword",
        "bridge_movie_cast",
        "bridge_movie_director",
        "item_sim_edge",
        "fact_user_event",
    ]
    db_path = settings.path(str(settings.warehouse.duckdb_path))
    con = duckdb.connect(str(db_path), read_only=True)
    frames: dict[str, pd.DataFrame] = {}
    try:
        existing = {row[0] for row in con.execute("SHOW TABLES").fetchall()}
        for table in wanted:
            if table in existing:
                frames[table] = con.execute(f"SELECT * FROM {table}").df()
    finally:
        con.close()
    return frames


def cross_features(user: dict, item: dict) -> dict[str, float]:
    """用户 × 影片 交叉特征（统一口径，粗排/精排共用同一实现）。"""
    genre_aff: dict[str, float] = user.get("genre_affinity") or {}
    keyword_aff: dict[str, float] = user.get("keyword_affinity") or {}
    director_aff = {int(k): float(v) for k, v in (user.get("director_affinity") or {}).items()}
    cast_aff = {int(k): float(v) for k, v in (user.get("cast_affinity") or {}).items()}

    def _match(affinity: dict[str, float], values: list[str]) -> float:
        if not affinity or not values:
            return 0.0
        scale = max(affinity.values()) or 1.0
        return float(sum(affinity.get(v, 0.0) for v in values) / (len(values) * scale))

    genres = list(item.get("genre_names") or [])
    keywords = list(item.get("keyword_names") or [])
    director_ids = [int(x) for x in (item.get("director_ids") or [])]
    cast_ids = [int(x) for x in (item.get("cast_ids") or [])]

    director_hit = 1.0 if any(d in director_aff for d in director_ids) else 0.0
    cast_hit = (sum(1 for c in cast_ids if c in cast_aff) / len(cast_ids)) if cast_ids else 0.0

    user_year = float(user.get("user_era_mean", 0.0)) * 40.0 + 1990.0
    item_year = float(item.get("year") or 2000)
    era_gap = -abs(item_year - user_year) / 40.0

    user_rating = float(user.get("user_avg_rating", 3.5))
    item_quality = float(item.get("quality_bayes", 6.5)) / 2.0
    quality_gap = -abs(item_quality - user_rating) / 5.0

    return {
        "genre_match": _match(genre_aff, genres),
        "keyword_match": _match(keyword_aff, keywords),
        "director_match": director_hit,
        "cast_match": float(cast_hit),
        "era_gap": era_gap,
        "quality_gap": quality_gap,
    }


def check_columns(frame: pd.DataFrame, features: list[str]) -> None:
    """校验特征矩阵列名与顺序，缺失时明确报错。"""
    missing = [name for name in features if name not in frame.columns]
    if missing:
        raise KeyError(f"特征矩阵缺少列: {missing}")


def build_matrix(frame: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    check_columns(frame, features)
    return frame.loc[:, features].astype("float32").fillna(0.0)


def artifacts_path(settings: Settings | None = None, name: str = "") -> Path:
    settings = settings or global_settings
    return settings.path(f"artifacts/{name}" if name else "artifacts")

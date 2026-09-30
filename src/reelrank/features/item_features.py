"""影片特征工程：质量分、新鲜度、热度、内容文本与结构化标签。

产物：artifacts/item_features.parquet（主键 movie_id）
"""

from __future__ import annotations

import math
from datetime import date, datetime, timezone
from typing import Any

import pandas as pd

from reelrank.config import Settings, settings as global_settings
from reelrank.features.registry import ITEM_FEATURES, load_warehouse
from reelrank.logging_utils import get_logger

logger = get_logger("reelrank.features.item")


def bayesian_quality(vote_average: pd.Series, vote_count: pd.Series, prior_votes: float, prior_mean: float) -> pd.Series:
    """贝叶斯平滑评分：(v*R + m*C) / (v + m)，抑制低票数影片的高分噪声。"""
    v = vote_count.fillna(0).astype(float)
    r = vote_average.fillna(prior_mean).astype(float)
    return (v * r + prior_votes * prior_mean) / (v + prior_votes)


def freshness_score(release_year: pd.Series, half_life_days: float, today: date | None = None) -> pd.Series:
    """按半衰期衰减的新鲜度，越新越接近 1。"""
    today = today or datetime.now(timezone.utc).date()
    years = release_year.fillna(2000).astype(float)
    age_days = (today.year - years) * 365.25
    return (0.5 ** (age_days / float(half_life_days))).clip(0.0, 1.0)


def _to_list(value: Any) -> list:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    try:
        return list(value)
    except TypeError:
        return []


def _group_lists(frame: pd.DataFrame, key: str, value: str, alias: str) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame({"movie_id": pd.Series(dtype="int64"), alias: pd.Series(dtype="object")})
    grouped = frame.groupby("movie_id")[value].apply(lambda s: [x for x in s if x]).reset_index()
    grouped.columns = ["movie_id", alias]
    return grouped


def build_item_features(settings: Settings | None = None, persist: bool = True) -> pd.DataFrame:
    """构建影片特征表并落盘 artifacts/item_features.parquet。"""
    settings = settings or global_settings
    tables = load_warehouse(settings, tables=["dim_movie", "bridge_movie_cast", "bridge_movie_director", "bridge_movie_company"])
    dim = tables.get("dim_movie")
    if dim is None or dim.empty:
        raise RuntimeError("数仓 dim_movie 为空，请先运行采集与 ETL")

    dim = dim.copy()
    dim["genre_names"] = dim["genre_names"].apply(_to_list)
    dim["keyword_names"] = dim["keyword_names"].apply(_to_list)
    dim["year"] = dim["year"].fillna(2000).astype(int)
    dim["vote_count"] = dim["vote_count"].fillna(0).astype(float)
    dim["vote_average"] = dim["vote_average"].fillna(0.0).astype(float)
    dim["popularity"] = dim["popularity"].fillna(0.0).astype(float)

    cast = tables.get("bridge_movie_cast", pd.DataFrame())
    director = tables.get("bridge_movie_director", pd.DataFrame())
    company = tables.get("bridge_movie_company", pd.DataFrame())

    cast_ids = _group_lists(cast, "movie_id", "person_id", "cast_ids") if not cast.empty else pd.DataFrame()
    cast_names = _group_lists(cast, "movie_id", "person_name", "cast_names") if not cast.empty else pd.DataFrame()
    director_ids = _group_lists(director, "movie_id", "person_id", "director_ids") if not director.empty else pd.DataFrame()
    director_names = _group_lists(director, "movie_id", "person_name", "director_names") if not director.empty else pd.DataFrame()
    company_names = _group_lists(company, "movie_id", "company_name", "company_names") if not company.empty else pd.DataFrame()

    items = dim[["movie_id", "title", "original_title", "overview", "tagline", "release_date", "year",
                 "runtime", "budget", "revenue", "popularity", "vote_average", "vote_count",
                 "original_language", "poster_path", "backdrop_path", "genre_names", "keyword_names"]].copy()

    for extra in (cast_ids, cast_names, director_ids, director_names, company_names):
        if not extra.empty:
            items = items.merge(extra, on="movie_id", how="left")
    for column in ("cast_ids", "cast_names", "director_ids", "director_names", "company_names"):
        if column not in items.columns:
            items[column] = [[] for _ in range(len(items))]
        items[column] = items[column].apply(_to_list)

    # ---------------- 数值特征 ----------------
    prior_votes = float(settings.features.bayesian_prior_votes)
    prior_mean = float(settings.features.bayesian_prior_mean)
    half_life = float(settings.features.freshness_half_life_days)

    items["quality_bayes"] = bayesian_quality(items["vote_average"], items["vote_count"], prior_votes, prior_mean)
    items["popularity_log"] = items["popularity"].clip(lower=0).apply(lambda v: math.log1p(float(v)) / 10.0)
    items["vote_count_log"] = items["vote_count"].clip(lower=0).apply(lambda v: math.log1p(float(v)) / 12.0)
    items["freshness"] = freshness_score(items["year"], half_life)
    items["runtime_norm"] = (items["runtime"].fillna(0).clip(0, 240).astype(float) / 240.0)
    items["revenue_log"] = items["revenue"].fillna(0).clip(lower=0).apply(lambda v: math.log1p(float(v)) / 22.0)
    items["roi_log"] = (items["revenue"].fillna(0).astype(float) / (items["budget"].fillna(0).astype(float) + 1.0)).clip(0, 50).apply(lambda v: math.log1p(v) / 5.0)
    items["n_genres"] = items["genre_names"].apply(len).astype(float)
    items["n_keywords"] = items["keyword_names"].apply(len).astype(float)
    items["n_cast"] = items["cast_ids"].apply(len).astype(float)

    # ---------------- 检索文本（BM25 语料） ----------------
    items["text"] = [
        " ".join(
            part for part in [
                title,
                original_title,
                tagline,
                overview,
                " ".join(genres),
                " ".join(keywords),
                " ".join(cast_names[:6]),
                " ".join(director_names),
                " ".join(companies),
                str(year),
            ] if part
        )
        for title, original_title, tagline, overview, genres, keywords, cast_names, director_names, companies, year in zip(
            items["title"], items["original_title"], items["tagline"], items["overview"],
            items["genre_names"], items["keyword_names"], items["cast_names"], items["director_names"],
            items["company_names"], items["year"]
        )
    ]

    for name in ITEM_FEATURES:
        items[name] = items[name].astype("float32")

    if persist:
        target = settings.path("artifacts/item_features.parquet")
        items.to_parquet(target, index=False)
        logger.info("影片特征落盘：%s 部 → %s", len(items), target)
    return items


if __name__ == "__main__":
    frame = build_item_features()
    print(frame[["movie_id", "title", "year", "quality_bayes", "freshness", "popularity_log"]].head())

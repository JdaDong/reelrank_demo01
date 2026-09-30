"""用户行为数据源。

优先级：
1. MovieLens `ml-latest-small`（`links.csv` 含 `tmdbId`，可与 TMDB 影片精确对齐）→ 真实交互
2. 固定种子合成行为日志 → 下载失败或对齐覆盖率过低时回退

产物：
- data/raw/events.jsonl          行为明细
- DuckDB `fact_user_event`       行为事实表
- artifacts/user_history.parquet 训练/召回用的用户历史（含 label）
"""

from __future__ import annotations

import io
import json
import math
import random
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from reelrank.config import Settings, settings as global_settings
from reelrank.logging_utils import get_logger

logger = get_logger("reelrank.data.behavior")

MIN_REAL_EVENTS = 2000      # 真实行为可用门槛
MIN_REAL_USERS = 50


# ---------------------------------------------------------------- MovieLens
def download_movielens(settings: Settings | None = None) -> Path | None:
    """下载并解压 MovieLens，返回数据目录；失败返回 None。"""
    settings = settings or global_settings
    dest = settings.path("data/raw/movielens")
    ratings, links = dest / "ratings.csv", dest / "links.csv"
    if ratings.exists() and links.exists():
        return dest

    import httpx

    url = str(settings.behavior.movielens_url)
    try:
        logger.info("下载 MovieLens：%s", url)
        resp = httpx.get(url, timeout=90.0, follow_redirects=True)
        resp.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            for member in zf.namelist():
                if member.endswith("ratings.csv"):
                    ratings.write_bytes(zf.read(member))
                elif member.endswith("links.csv"):
                    links.write_bytes(zf.read(member))
    except Exception as exc:  # 网络不可用时静默回退
        logger.warning("MovieLens 下载失败：%s", exc)
        return None

    if not (ratings.exists() and links.exists()):
        return None
    logger.info("MovieLens 就绪：%s", dest)
    return dest


def build_events_from_movielens(settings: Settings, movie_ids: set[int]) -> pd.DataFrame | None:
    """把 MovieLens 评分按 tmdbId 对齐到本仓库影片，产出行为明细。"""
    directory = download_movielens(settings)
    if directory is None:
        return None

    ratings = pd.read_csv(directory / "ratings.csv")
    links = pd.read_csv(directory / "links.csv")
    links = links.dropna(subset=["tmdbId"])
    links["tmdbId"] = links["tmdbId"].astype("int64")
    links = links[links["tmdbId"].isin(movie_ids)]
    if links.empty:
        logger.warning("MovieLens 与本仓库影片无 tmdbId 交集")
        return None

    joined = ratings.merge(links[["movieId", "tmdbId"]], on="movieId", how="inner")
    if joined.empty:
        return None

    threshold = float(settings.behavior.movielens_min_rating)
    frame = pd.DataFrame(
        {
            "user_id": joined["userId"].astype("int64"),
            "movie_id": joined["tmdbId"].astype("int64"),
            "rating": joined["rating"].astype(float),
            "event_time": pd.to_datetime(joined["timestamp"], unit="s"),
        }
    )
    frame["label"] = (frame["rating"] >= threshold).astype(int)
    frame["event_type"] = frame["label"].map({1: "click", 0: "impression"})
    frame["source"] = "movielens"

    if len(frame) < MIN_REAL_EVENTS or frame["user_id"].nunique() < MIN_REAL_USERS:
        logger.warning(
            "MovieLens 对齐后样本不足（%s 条 / %s 用户），回退合成行为",
            len(frame),
            frame["user_id"].nunique(),
        )
        return None

    logger.info(
        "MovieLens 对齐完成：%s 条行为 / %s 用户 / %s 部影片（正样本率 %.2f）",
        len(frame),
        frame["user_id"].nunique(),
        frame["movie_id"].nunique(),
        frame["label"].mean(),
    )
    return frame


def _weighted_sample_without_replacement(rng: random.Random, pool: list, weights: list[float], k: int) -> list:
    """按权重无放回抽样（random.sample 的 counts 只接受整数，故自行实现）。"""
    remaining = list(range(len(pool)))
    picked: list = []
    for _ in range(min(k, len(pool))):
        total = sum(weights[i] for i in remaining)
        if total <= 0:
            break
        threshold = rng.random() * total
        acc = 0.0
        chosen = remaining[-1]
        for index in remaining:
            acc += weights[index]
            if acc >= threshold:
                chosen = index
                break
        picked.append(pool[chosen])
        remaining.remove(chosen)
    return picked


# ---------------------------------------------------------------- 合成行为
def synthesize_events(settings: Settings, items: pd.DataFrame) -> pd.DataFrame:
    """基于影片真实特征生成固定种子的合成行为日志。

    用户偏好 = 类型偏好 + 年代偏好 + 质量敏感度；候选打分后按 softmax 采样，
    因此行为分布与影片内容结构一致，可用于训练与评估。
    """
    cfg = settings.behavior.synthetic
    rng = random.Random(int(cfg.seed))
    genres = sorted({g for row in items["genre_names"] for g in row})
    if not genres:
        genres = ["剧情"]

    movie_rows = items.to_dict("records")
    n_users = int(cfg.users)
    now = datetime.now(timezone.utc)

    records: list[dict[str, Any]] = []
    for uid in range(1, n_users + 1):
        # 稀疏偏好：2~4 个强偏好类型 + 其余弱偏好，使画像具备区分度
        favorites = set(rng.sample(genres, k=rng.randint(2, 4)))
        affinity = {g: (rng.uniform(1.0, 2.0) if g in favorites else rng.uniform(0.02, 0.25)) for g in genres}
        era_center = int(rng.triangular(1995, 2026, 2016))
        era_width = rng.uniform(4.0, 14.0)
        quality_bias = rng.gauss(0.0, 0.35)

        def score(movie: dict[str, Any]) -> float:
            genre_score = sum(affinity.get(g, 0.0) for g in movie["genre_names"]) / max(1, len(movie["genre_names"]))
            era_score = math.exp(-((movie["year"] - era_center) ** 2) / (2 * era_width**2))
            quality = (movie["quality_bayes"] - 6.0) / 4.0
            return 1.5 * genre_score + 1.0 * era_score + quality_bias * quality + 0.35 * movie["popularity_log"]

        pool = rng.sample(movie_rows, min(len(movie_rows), 320))
        weights = [math.exp(score(m)) for m in pool]
        n_events = rng.randint(int(cfg.events_per_user_min), int(cfg.events_per_user_max))

        def _emit(movie: dict[str, Any], label: int, rating: float) -> None:
            records.append(
                {
                    "user_id": uid,
                    "movie_id": int(movie["movie_id"]),
                    "rating": round(float(rating), 2),
                    "label": label,
                    "event_type": "click" if label else "impression",
                    "event_time": now - pd.Timedelta(days=rng.randint(0, 180), hours=rng.randint(0, 23)),
                    "source": "synthetic",
                }
            )

        # 正例：按偏好强度采样，点击概率由 sigmoid(匹配分) 决定，正样本率约 30%~40%
        for movie in _weighted_sample_without_replacement(rng, pool, weights, n_events):
            p_click = 1.0 / (1.0 + math.exp(-(score(movie) - 3.05) * 1.55))
            if rng.random() < p_click:
                _emit(movie, 1, min(5.0, max(3.0, rng.gauss(4.1, 0.55))))
            else:
                _emit(movie, 0, min(3.4, max(0.5, rng.gauss(2.5, 0.7))))

        # 探索行为：与画像无关的随机点击（真实用户总会被热点/好奇心驱动），
        # 保证历史无法完全解释未来行为，避免排序模型 AUC 虚高
        exploration_pool = rng.sample(movie_rows, min(len(movie_rows), 150))
        for movie in rng.sample(exploration_pool, rng.randint(2, 7)):
            if rng.random() < 0.38:
                _emit(movie, 1, min(5.0, max(3.0, rng.gauss(3.9, 0.5))))
            else:
                _emit(movie, 0, min(3.4, max(0.5, rng.gauss(2.4, 0.8))))

        # 负例：曝光未点击，模拟真实流量中的低相关曝光
        inverse_pool = rng.sample(movie_rows, min(len(movie_rows), 120))
        inverse_weights = [math.exp(-score(m)) for m in inverse_pool]
        for movie in _weighted_sample_without_replacement(rng, inverse_pool, inverse_weights, rng.randint(6, 16)):
            _emit(movie, 0, min(3.4, max(0.5, rng.gauss(2.3, 0.8))))

    frame = pd.DataFrame(records)
    logger.info(
        "合成行为生成完成：%s 条 / %s 用户 / %s 部影片（正样本率 %.2f）",
        len(frame),
        frame["user_id"].nunique(),
        frame["movie_id"].nunique(),
        frame["label"].mean(),
    )
    return frame


# ---------------------------------------------------------------- 编排
def _load_movie_frame(settings: Settings) -> pd.DataFrame:
    import duckdb

    db_path = settings.path(str(settings.warehouse.duckdb_path))
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        frame = con.execute(
            "SELECT movie_id, year, popularity, vote_average, vote_count, genre_names FROM dim_movie"
        ).df()
    finally:
        con.close()
    prior_votes = float(settings.features.bayesian_prior_votes)
    prior_mean = float(settings.features.bayesian_prior_mean)
    frame["quality_bayes"] = (
        frame["vote_count"] * frame["vote_average"] + prior_votes * prior_mean
    ) / (frame["vote_count"] + prior_votes)
    frame["popularity_log"] = frame["popularity"].clip(lower=0).apply(lambda v: math.log1p(v))
    frame["year"] = frame["year"].fillna(2000).astype(int)
    return frame


def ensure_events(settings: Settings | None = None, refresh: bool = False) -> dict[str, Any]:
    """保证行为数据就绪：写 data/raw/events.jsonl + 数仓 fact_user_event。"""
    settings = settings or global_settings
    events_path = settings.path("data/raw/events.jsonl")
    meta_path = settings.path("data/raw/behavior_meta.json")

    if events_path.exists() and meta_path.exists() and not refresh:
        logger.info("复用已有行为数据：%s", events_path)
        return json.loads(meta_path.read_text(encoding="utf-8"))

    items = _load_movie_frame(settings)
    movie_ids = set(items["movie_id"].tolist())
    source = str(settings.behavior.source)

    frame: pd.DataFrame | None = None
    if source in ("auto", "movielens"):
        frame = build_events_from_movielens(settings, movie_ids)
    if frame is None:
        if source == "movielens":
            logger.warning("behavior.source=movielens 但真实数据不可用，回退合成行为")
        frame = synthesize_events(settings, items)

    frame = frame.sort_values(["user_id", "event_time"])
    events_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False, default=str) for row in frame.to_dict("records")) + "\n",
        encoding="utf-8",
    )

    parquet_dir = settings.path(str(settings.warehouse.parquet_dir))
    fact_path = parquet_dir / "fact_user_event.parquet"
    fact = frame[["user_id", "movie_id", "event_type", "rating", "event_time", "source"]].copy()
    fact.to_parquet(fact_path, index=False)

    import duckdb

    db_path = settings.path(str(settings.warehouse.duckdb_path))
    con = duckdb.connect(str(db_path))
    try:
        con.execute(Path(__file__).parent.parent.joinpath("warehouse/schema.sql").read_text(encoding="utf-8"))
        con.execute(
            f"CREATE OR REPLACE TABLE fact_user_event AS "
            f"SELECT user_id, movie_id, event_type, rating, event_time, source "
            f"FROM read_parquet('{fact_path.as_posix()}')"
        )
        con.execute("CHECKPOINT")
    finally:
        con.close()

    history_path = settings.path("artifacts/user_history.parquet")
    frame[["user_id", "movie_id", "label", "rating", "event_time"]].to_parquet(history_path, index=False)

    meta = {
        "source": frame["source"].iloc[0],
        "event_count": int(len(frame)),
        "user_count": int(frame["user_id"].nunique()),
        "movie_count": int(frame["movie_id"].nunique()),
        "positive_rate": round(float(frame["label"].mean()), 4),
        "built_at": datetime.now(timezone.utc).isoformat(),
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("行为数据落盘：%s", meta)
    return meta


if __name__ == "__main__":
    print(ensure_events())

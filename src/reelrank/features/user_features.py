"""用户画像特征：类型/关键词/导演/演员偏好、年代偏好、活跃度与行为历史。

产物：artifacts/user_features.parquet（主键 user_id）
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from typing import Any

import pandas as pd

from reelrank.config import Settings, settings as global_settings
from reelrank.features.item_features import build_item_features
from reelrank.features.registry import USER_FEATURES
from reelrank.logging_utils import get_logger

logger = get_logger("reelrank.features.user")

ERA_BASE, ERA_SPAN = 1990.0, 40.0


def _entropy(weights: dict[str, float]) -> float:
    total = sum(weights.values())
    if total <= 0:
        return 0.0
    return float(-sum((w / total) * math.log(w / total + 1e-12) for w in weights.values()) / math.log(max(2, len(weights))))


def _top(weights: dict[Any, float], k: int) -> list[Any]:
    return [key for key, _ in sorted(weights.items(), key=lambda kv: kv[1], reverse=True)[:k]]


def _load_history(settings: Settings) -> pd.DataFrame:
    history_path = settings.path("artifacts/user_history.parquet")
    if history_path.exists():
        return pd.read_parquet(history_path)
    frames = __import__("reelrank.features.registry", fromlist=["load_warehouse"]).load_warehouse(settings, tables=["fact_user_event"])
    events = frames.get("fact_user_event")
    if events is None or events.empty:
        raise RuntimeError("缺少行为数据，请先运行 ensure_events()")
    events = events.copy()
    events["label"] = (events["rating"] >= 3.5).astype(int)
    return events[["user_id", "movie_id", "label", "rating", "event_time"]]


def build_user_features(
    settings: Settings | None = None,
    items: pd.DataFrame | None = None,
    persist: bool = True,
    history: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """构建用户画像表并落盘 artifacts/user_features.parquet。

    传入 history 时可基于指定行为切片构建画像（训练时只用训练段，避免标签泄漏）。
    """
    settings = settings or global_settings
    if items is None:
        items = build_item_features(settings, persist=False)

    item_index: dict[int, dict[str, Any]] = {
        int(row["movie_id"]): row for row in items.to_dict("records")
    }
    history = history if history is not None else _load_history(settings)
    history = history[history["movie_id"].isin(item_index.keys())]
    if history.empty:
        raise RuntimeError("行为数据与影片特征无交集，请检查数仓与行为对齐结果")

    grouped: dict[int, list[tuple[int, float, float]]] = defaultdict(list)
    for user_id, movie_id, label, rating in zip(history["user_id"], history["movie_id"], history["label"], history["rating"]):
        grouped[int(user_id)].append((int(movie_id), float(label), float(rating)))

    rows: list[dict[str, Any]] = []
    for user_id, interactions in sorted(grouped.items()):
        genre_w: dict[str, float] = defaultdict(float)
        keyword_w: dict[str, float] = defaultdict(float)
        director_w: dict[int, float] = defaultdict(float)
        cast_w: dict[int, float] = defaultdict(float)
        year_values: list[float] = []
        positives: list[int] = []
        all_items: list[int] = []
        label_sum = 0.0
        rating_sum = 0.0

        for movie_id, label, rating in interactions:
            item = item_index.get(movie_id)
            if item is None:
                continue
            weight = 0.4 + 0.6 * label + 0.12 * max(0.0, rating - 3.0)
            for genre in item["genre_names"]:
                genre_w[genre] += weight
            for keyword in item["keyword_names"][:8]:
                keyword_w[keyword] += weight
            for director_id in item["director_ids"][:2]:
                director_w[int(director_id)] += weight
            for cast_id in item["cast_ids"][:5]:
                cast_w[int(cast_id)] += weight * 0.7
            year_values.append(float(item["year"]))
            all_items.append(movie_id)
            if label >= 1:
                positives.append(movie_id)
            label_sum += label
            rating_sum += rating

        n = max(1, len(interactions))
        era_mean = sum(year_values) / len(year_values) if year_values else 2005.0
        era_std = (sum((y - era_mean) ** 2 for y in year_values) / len(year_values)) ** 0.5 if year_values else 5.0

        rows.append(
            {
                "user_id": user_id,
                "user_activity": math.log1p(n) / 8.0,
                "user_click_rate": label_sum / n,
                "user_avg_rating": rating_sum / n,
                "user_era_mean": (era_mean - ERA_BASE) / ERA_SPAN,
                "user_era_std": min(1.0, era_std / 20.0),
                "user_genre_entropy": _entropy(dict(genre_w)),
                "genre_affinity": json.dumps(dict(genre_w), ensure_ascii=False),
                "keyword_affinity": json.dumps(dict(keyword_w), ensure_ascii=False),
                "director_affinity": json.dumps({str(k): v for k, v in director_w.items()}),
                "cast_affinity": json.dumps({str(k): v for k, v in cast_w.items()}),
                "top_genres": _top(genre_w, 3),
                "top_keywords": _top(keyword_w, 5),
                "top_directors": _top(director_w, 3),
                "top_actors": _top(cast_w, 5),
                "history": all_items[-100:],
                "history_positive": positives[-50:],
            }
        )

    users = pd.DataFrame(rows)
    for name in USER_FEATURES:
        users[name] = users[name].astype("float32")

    if persist:
        target = settings.path("artifacts/user_features.parquet")
        users.to_parquet(target, index=False)
        logger.info("用户画像落盘：%s 个用户 → %s", len(users), target)
    return users


def default_profile(users: pd.DataFrame) -> dict[str, Any]:
    """冷启动用户：取全量用户的统计均值画像。"""
    numeric = {name: float(users[name].mean()) for name in USER_FEATURES}
    genre_counter: dict[str, float] = defaultdict(float)
    for payload in users["genre_affinity"]:
        for name, weight in json.loads(payload).items():
            genre_counter[name] += weight
    return {
        **numeric,
        "genre_affinity": dict(genre_counter),
        "keyword_affinity": {},
        "director_affinity": {},
        "cast_affinity": {},
        "top_genres": _top(genre_counter, 3),
        "top_keywords": [],
        "top_directors": [],
        "top_actors": [],
        "history": [],
        "history_positive": [],
    }


if __name__ == "__main__":
    frame = build_user_features()
    print(frame[["user_id", "user_click_rate", "user_avg_rating", "top_genres"]].head())

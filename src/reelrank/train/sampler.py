"""训练样本构造：模拟线上「召回 → 粗排 → 精排」分布生成级联训练样本。

- 推荐场景：以用户历史（训练段）为 trigger，跑 itemcf / 向量 / 热门召回，
  label = 该影片是否出现在用户未来的正样本（时间切分的测试段）
- 搜索场景：从影片标题/关键词合成 query，跑 bm25 / 向量(query) / i2i 召回，
  label = 相关性代理（query 源影片及其 TMDB 相似邻居）

产物列：sample_id / scene / user_id / movie_id / label + 召回分 + 影片/用户/交叉特征
"""

from __future__ import annotations

import json
import random

import numpy as np
import pandas as pd

from reelrank.config import Settings, settings as global_settings
from reelrank.features.registry import (
    CROSS_FEATURES,
    ITEM_FEATURES,
    RECALL_FEATURES,
    USER_FEATURES,
    cross_features,
)
from reelrank.logging_utils import get_logger

logger = get_logger("reelrank.train.sampler")


def _as_list(value) -> list:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return list(value)


def build_item_dicts(items: pd.DataFrame) -> dict[int, dict]:
    return {
        int(row["movie_id"]): {
            "genre_names": _as_list(row["genre_names"]),
            "keyword_names": _as_list(row["keyword_names"]),
            "director_ids": [int(x) for x in _as_list(row["director_ids"])],
            "cast_ids": [int(x) for x in _as_list(row["cast_ids"])],
            "year": int(row["year"]),
            "quality_bayes": float(row["quality_bayes"]),
        }
        for row in items.to_dict("records")
    }


def build_item_partials(items: pd.DataFrame) -> dict[int, dict]:
    return {
        int(row["movie_id"]): {name: float(row[name]) for name in ITEM_FEATURES}
        for row in items.to_dict("records")
    }


def build_user_profiles(users: pd.DataFrame) -> dict[int, dict]:
    profiles: dict[int, dict] = {}
    for row in users.to_dict("records"):
        profiles[int(row["user_id"])] = {
            **{name: float(row[name]) for name in USER_FEATURES},
            "genre_affinity": json.loads(row["genre_affinity"] or "{}"),
            "keyword_affinity": json.loads(row["keyword_affinity"] or "{}"),
            "director_affinity": json.loads(row["director_affinity"] or "{}"),
            "cast_affinity": json.loads(row["cast_affinity"] or "{}"),
        }
    return profiles


def _hot_scores(items: pd.DataFrame) -> dict[int, float]:
    ranked = items["popularity_log"].rank(pct=True)
    return {int(mid): float(score) for mid, score in zip(items["movie_id"], ranked)}


def _make_rows(
    sample_id: int,
    scene: str,
    user_id: int,
    profile: dict,
    movie_ids: list[int],
    score_maps: dict[str, dict[int, float]],
    labels: dict[int, int],
    item_dicts: dict[int, dict],
    item_partials: dict[int, dict],
) -> list[dict]:
    rows = []
    for movie_id in movie_ids:
        item = item_dicts.get(int(movie_id))
        partial = item_partials.get(int(movie_id))
        if item is None or partial is None:
            continue
        row = {
            "sample_id": sample_id,
            "scene": scene,
            "user_id": user_id,
            "movie_id": int(movie_id),
            "label": int(labels.get(int(movie_id), 0)),
            "bm25_score": float(score_maps.get("bm25", {}).get(int(movie_id), 0.0)),
            "vector_score": float(score_maps.get("vector", {}).get(int(movie_id), 0.0)),
            "itemcf_score": float(score_maps.get("itemcf", {}).get(int(movie_id), 0.0)),
            "hot_score": float(score_maps.get("hot", {}).get(int(movie_id), 0.0)),
        }
        row.update(partial)
        row.update({name: profile.get(name, 0.0) for name in USER_FEATURES})
        row.update(cross_features(profile, item))
        rows.append(row)
    return rows


def split_history_frames(history: pd.DataFrame, test_ratio: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """按时间切分行为序列：训练段（画像/trigger）与测试段（label）。"""
    history = history.sort_values(["user_id", "event_time"], kind="mergesort")
    train_parts, test_parts = [], []
    for _, group in history.groupby("user_id", sort=False):
        cut = max(1, int(len(group) * (1 - test_ratio)))
        train_parts.append(group.iloc[:cut])
        if cut < len(group):
            test_parts.append(group.iloc[cut:])
    train = pd.concat(train_parts) if train_parts else history.iloc[0:0]
    test = pd.concat(test_parts) if test_parts else history.iloc[0:0]
    return train, test


def _split_maps(train: pd.DataFrame, test: pd.DataFrame) -> tuple[dict[int, list[int]], dict[int, set[int]], dict[int, list[int]]]:
    train_pos: dict[int, list[int]] = {}
    test_pos: dict[int, set[int]] = {}
    train_all: dict[int, list[int]] = {}
    for user_id, group in train.groupby("user_id", sort=False):
        positives = group[group["label"] >= 1]["movie_id"].tolist()
        train_pos[int(user_id)] = [int(x) for x in positives]
        train_all[int(user_id)] = [int(x) for x in group["movie_id"].tolist()]
    for user_id, group in test.groupby("user_id", sort=False):
        positives = {int(x) for x in group[group["label"] >= 1]["movie_id"].tolist()}
        if positives:
            test_pos[int(user_id)] = positives
    return train_pos, test_pos, train_all


def build_training_frames(
    settings: Settings | None = None,
    items: pd.DataFrame | None = None,
    users: pd.DataFrame | None = None,
    history: pd.DataFrame | None = None,
    bm25=None,
    vector=None,
    itemcf=None,
    similar_map: dict[int, list[int]] | None = None,
) -> pd.DataFrame:
    """构造级联训练样本（粗排/精排共用同一份特征，精排取粗排 Top-K 子集）。"""
    settings = settings or global_settings
    cfg = settings.train
    rng = random.Random(int(cfg.random_state))

    item_ids = [int(x) for x in items["movie_id"]]
    item_dicts = build_item_dicts(items)
    item_partials = build_item_partials(items)
    profiles = build_user_profiles(users)
    hot = _hot_scores(items)

    train_history, test_history = split_history_frames(history, float(settings.eval.test_ratio))
    train_pos, test_pos, train_all = _split_maps(train_history, test_history)
    rows: list[dict] = []
    sample_id = 0

    # ---------------- 推荐场景 ----------------
    eligible = [u for u in profiles if len(train_pos.get(u, [])) >= 5 and len(test_pos.get(u, set())) >= 1]
    sampled_users = rng.sample(eligible, min(int(cfg.sample_users), len(eligible))) if eligible else []
    for user_id in sampled_users:
        sample_id += 1
        positives = train_pos[user_id]
        triggers = [(mid, 0.6 + 0.4 * (i / max(1, len(positives) - 1))) for i, mid in enumerate(positives[-20:])]
        exclude = set(train_all[user_id])

        itemcf_scores = dict(itemcf.recall(triggers, topk=int(settings.funnel.recall.itemcf_topk), exclude=exclude))
        user_vector = vector.user_vector([(mid, 1.0) for mid in positives[-30:]])
        vector_scores = dict(vector.search(user_vector, topk=int(settings.funnel.recall.vector_topk), exclude=exclude))

        candidates = set(itemcf_scores) | set(vector_scores) | set(
            sorted(hot, key=lambda m: hot[m], reverse=True)[: int(settings.funnel.recall.hot_topk)]
        )
        ranked = sorted(candidates, key=lambda m: itemcf_scores.get(m, 0.0) + vector_scores.get(m, 0.0), reverse=True)
        selected = ranked[: int(cfg.candidates_per_sample)]

        # 未被召回的正样本同样入样（召回分为 0），避免模型只见过"容易"的正例
        missing = [mid for mid in test_pos[user_id] if mid not in candidates][:10]
        selected = selected + missing

        labels = {mid: 1 for mid in test_pos[user_id]}
        rows.extend(
            _make_rows(
                sample_id, "recommend", user_id, profiles[user_id], selected,
                {"itemcf": itemcf_scores, "vector": vector_scores, "hot": hot},
                labels, item_dicts, item_partials,
            )
        )

    # ---------------- 搜索场景 ----------------
    # label 来自用户真实行为，但只统计「与 query 相关」的命中（bm25 召回集内），
    # 避免把与 query 无关的历史正例标成正样本而弱化文本相关性。
    movie_to_users: dict[int, list[int]] = {}
    for user_id, positives in test_pos.items():
        for movie_id in positives:
            movie_to_users.setdefault(int(movie_id), []).append(int(user_id))

    pool = items.sample(n=min(int(cfg.sample_queries), len(items)), random_state=int(cfg.random_state))
    user_ids = list(profiles.keys())
    for row in pool.to_dict("records"):
        sample_id += 1
        movie_id = int(row["movie_id"])
        keywords = _as_list(row["keyword_names"])[:2]
        query = " ".join([str(row["title"]), *keywords]).strip()

        bm25_scores = dict(bm25.search(query, topk=int(settings.funnel.recall.bm25_topk)))
        query_vector = vector.query_vector(query, bm25, topn=20)
        vector_scores = dict(vector.search(query_vector, topk=int(settings.funnel.recall.vector_topk)))
        itemcf_scores = dict(
            itemcf.recall([(mid, score) for mid, score in list(bm25_scores.items())[:10]],
                          topk=int(settings.funnel.recall.itemcf_topk), exclude={movie_id})
        )

        candidates = set(bm25_scores) | set(vector_scores) | set(
            sorted(hot, key=lambda m: hot[m], reverse=True)[: int(settings.funnel.recall.hot_topk)]
        )
        ranked = sorted(candidates, key=lambda m: bm25_scores.get(m, 0.0) + vector_scores.get(m, 0.0), reverse=True)
        selected = ranked[: int(cfg.candidates_per_sample)]

        # 选出「对这条 query 有过正反馈」的用户，使搜索标签既来自行为又与 query 相关
        relevant_users: list[int] = []
        for hit in list(bm25_scores)[:50]:
            relevant_users.extend(movie_to_users.get(int(hit), []))
        user_id = rng.choice(relevant_users) if relevant_users else (rng.choice(user_ids) if user_ids else 0)
        relevant = set(list(bm25_scores)[:50])
        positives = {mid for mid in test_pos.get(user_id, set()) if mid in relevant}
        positives.add(movie_id)  # query 精确命中的主需结果

        missing = [mid for mid in positives if mid not in candidates]
        selected = selected + missing
        labels = {mid: 1 for mid in positives}
        rows.extend(
            _make_rows(
                sample_id, "search", user_id, profiles.get(user_id, {}), selected,
                {"bm25": bm25_scores, "vector": vector_scores, "itemcf": itemcf_scores, "hot": hot},
                labels, item_dicts, item_partials,
            )
        )

    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError("训练样本为空，请检查行为数据与召回索引")

    feature_columns = RECALL_FEATURES[:4] + ITEM_FEATURES + USER_FEATURES + CROSS_FEATURES
    for column in feature_columns:
        if column not in frame.columns:
            frame[column] = 0.0
        frame[column] = frame[column].astype("float32")

    logger.info(
        "训练样本构造完成：%s 行 / %s 个样本 / 正样本率 %.3f（推荐 %s / 搜索 %s）",
        len(frame), frame["sample_id"].nunique(), frame["label"].mean(),
        int((frame["scene"] == "recommend").sum()), int((frame["scene"] == "search").sum()),
    )
    return frame

"""离线训练编排：构建召回索引 → 构造级联样本 → 训练粗排 LR 与精排 GBDT → 落盘。

产物（artifacts/）：
- bm25_index.joblib / vector_index.npz / item_cf.joblib
- coarse_ranker.joblib / fine_ranker.joblib
- train_samples.parquet（含粗排分，供评估与复现）
- train_meta.json（样本量、AUC、特征重要性）
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from reelrank.config import Settings, settings as global_settings
from reelrank.features.item_features import build_item_features
from reelrank.features.registry import load_warehouse
from reelrank.features.user_features import build_user_features
from reelrank.logging_utils import get_logger
from reelrank.models.bm25_index import BM25Index
from reelrank.models.coarse_ranker import CoarseRanker
from reelrank.models.fine_ranker import FineRanker
from reelrank.models.item_cf import ItemCF
from reelrank.models.vector_index import VectorIndex
from reelrank.train.sampler import build_training_frames, split_history_frames

logger = get_logger("reelrank.train.offline")


def _load_edges(settings: Settings) -> tuple[pd.DataFrame, dict[int, list[int]]]:
    frames = load_warehouse(settings, tables=["item_sim_edge"])
    edges = frames.get("item_sim_edge", pd.DataFrame())
    similar_map: dict[int, list[int]] = {}
    if not edges.empty and "relation" in edges.columns:
        sim = edges[edges["relation"].isin(["similar", "recommendation", "recommended"])]
        for src, group in sim.groupby("src_movie_id"):
            similar_map[int(src)] = [int(x) for x in group.sort_values("rank_pos")["dst_movie_id"].tolist()]
    return edges, similar_map


def _load_history(settings: Settings) -> pd.DataFrame:
    path = settings.path("artifacts/user_history.parquet")
    if not path.exists():
        raise RuntimeError("缺少 artifacts/user_history.parquet，请先运行特征流水线")
    return pd.read_parquet(path)


def train_all(settings: Settings | None = None, refresh: bool = False) -> dict:
    settings = settings or global_settings
    artifacts = settings.path("artifacts")

    items_path = artifacts / "item_features.parquet"
    users_path = artifacts / "user_features.parquet"
    items = pd.read_parquet(items_path) if items_path.exists() and not refresh else build_item_features(settings)
    users = pd.read_parquet(users_path) if users_path.exists() and not refresh else build_user_features(settings, items)
    history = _load_history(settings)
    edges, similar_map = _load_edges(settings)

    # 训练用画像只基于训练段行为，避免测试段标签泄漏进用户/交叉特征
    train_history, _ = split_history_frames(history, float(settings.eval.test_ratio))
    users_train = build_user_features(settings, items, persist=False, history=train_history)

    item_ids = [int(x) for x in items["movie_id"].tolist()]

    # ---------------- 召回索引 ----------------
    bm25 = BM25Index().build(item_ids, items["text"].tolist())
    bm25.save(artifacts / "bm25_index.joblib")

    vector = VectorIndex(dim=int(settings.features.svd_dim)).build(item_ids, history)
    vector.save(artifacts / "vector_index.npz")

    itemcf = ItemCF(max_neighbors=int(settings.train.max_neighbors)).build(item_ids, history, edges)
    itemcf.save(artifacts / "item_cf.joblib")

    # ---------------- 级联样本 ----------------
    frame = build_training_frames(settings, items, users_train, history, bm25, vector, itemcf, similar_map)

    # ---------------- 粗排 ----------------
    coarse = CoarseRanker(random_state=int(settings.train.random_state))
    coarse_meta = coarse.train(frame)
    coarse.save(artifacts / "coarse_ranker.joblib")

    frame = frame.copy()
    frame["coarse_score"] = coarse.predict(frame)

    # ---------------- 精排（粗排 Top-K 子集） ----------------
    ordered = frame.sort_values(["sample_id", "coarse_score"], ascending=[True, False])
    fine_frame = ordered.groupby("sample_id", group_keys=False).head(int(settings.train.fine_candidates_per_sample))
    fine = FineRanker(settings=settings)
    fine_meta = fine.train(fine_frame)
    fine.save(artifacts / "fine_ranker.joblib")

    # 全量样本补上 pctr，便于离线评估与重训对比
    frame["pctr"] = fine.predict(frame)
    frame.to_parquet(artifacts / "train_samples.parquet", index=False)

    meta = {
        "built_at": datetime.now(timezone.utc).isoformat(),
        "items": len(item_ids),
        "samples": int(frame["sample_id"].nunique()),
        "rows": int(len(frame)),
        "positive_rate": round(float(frame["label"].mean()), 4),
        "coarse": coarse_meta,
        "fine": fine_meta,
        "vector_explained_variance": round(vector.explained_variance, 4),
        "bm25_terms": len(bm25.doc_freq),
        "itemcf_items": len(itemcf.neighbor_ids),
        "artifacts": {
            "bm25": "artifacts/bm25_index.joblib",
            "vector": "artifacts/vector_index.npz",
            "itemcf": "artifacts/item_cf.joblib",
            "coarse": "artifacts/coarse_ranker.joblib",
            "fine": "artifacts/fine_ranker.joblib",
            "samples": "artifacts/train_samples.parquet",
        },
    }
    (artifacts / "train_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(
        "离线训练完成：粗排 AUC=%.4f / 精排 train AUC=%.4f valid AUC=%.4f",
        coarse_meta["train_auc"], fine_meta["train_auc"], fine_meta.get("valid_auc") or 0.0,
    )
    return meta


if __name__ == "__main__":
    print(json.dumps(train_all(), ensure_ascii=False, indent=2))

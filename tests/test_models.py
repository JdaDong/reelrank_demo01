from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from reelrank.features.registry import COARSE_FEATURES, FINE_FEATURES
from reelrank.models.bm25_index import BM25Index, tokenize
from reelrank.models.coarse_ranker import CoarseRanker
from reelrank.models.fine_ranker import FineRanker
from reelrank.models.item_cf import ItemCF
from reelrank.models.vector_index import VectorIndex
from reelrank.train.sampler import split_history_frames

ARTIFACTS = Path("artifacts")


def test_tokenize_handles_mixed_languages():
    tokens = tokenize("星际 Inception 2020")
    assert "inception" in tokens and "2020" in tokens
    assert "星" in tokens and "际" in tokens and "星际" in tokens


def test_bm25_ranks_relevant_document_first():
    docs = ["科幻 太空 飞船 太空", "爱情 家庭 婚礼", "科幻 机器人"]
    index = BM25Index().build([1, 2, 3], docs)
    hits = index.search("科幻 太空", topk=3)
    assert hits[0][0] == 1
    assert 0.0 < hits[0][1] <= 1.0
    assert index.score_movies("科幻", [1, 2, 3])[0] > index.score_movies("科幻", [1, 2, 3])[1]


def _tiny_history() -> pd.DataFrame:
    rows = []
    for user in range(1, 31):
        for movie, label in [(1, 1), (2, 1), (3, 0), (4, 1)]:
            rows.append({"user_id": user, "movie_id": movie + (user % 3), "label": label, "rating": 4.0 if label else 2.0})
    return pd.DataFrame(rows)


def test_vector_index_build_search_and_roundtrip(tmp_path):
    items = list(range(1, 7))
    index = VectorIndex(dim=8).build(items, _tiny_history())
    assert index.item_vectors.shape[0] == len(items)

    vector = index.user_vector([(1, 1.0), (2, 1.0)])
    hits = index.search(vector, topk=3, exclude={1, 2})
    assert hits and all(mid not in {1, 2} for mid, _ in hits)
    assert all(0.0 <= score <= 1.0 for _, score in hits)

    target = tmp_path / "vec.npz"
    index.save(target)
    reloaded = VectorIndex.load(target)
    assert np.allclose(reloaded.item_vectors, index.item_vectors)
    assert reloaded.item_vector(3).shape == (index.item_vectors.shape[1],)


def test_cold_start_vector_shares_embedding_space():
    index = VectorIndex(dim=8).build(list(range(1, 7)), _tiny_history())
    assert index.global_vector.shape == (index.item_vectors.shape[1],)
    hits = index.search(index.global_vector, topk=3)
    assert hits and all(0.0 <= score <= 1.0 for _, score in hits)


def test_itemcf_recall_and_roundtrip(tmp_path):
    itemcf = ItemCF(max_neighbors=5).build(list(range(1, 7)), _tiny_history())
    assert itemcf.neighbor_ids, "ItemCF 应构建出邻居表"
    hits = itemcf.recall([(1, 1.0)], topk=5, exclude={1})
    assert hits and hits[0][1] == pytest.approx(1.0, abs=1e-6)  # 最高分归一化为 1

    target = tmp_path / "itemcf.joblib"
    itemcf.save(target)
    reloaded = ItemCF.load(target)
    assert reloaded.neighbors_of(1, topk=2) == itemcf.neighbors_of(1, topk=2)


def test_split_history_frames_is_temporal():
    history = pd.DataFrame(
        {
            "user_id": [1] * 10,
            "movie_id": list(range(10)),
            "label": [1] * 10,
            "rating": [4.0] * 10,
            "event_time": pd.date_range("2024-01-01", periods=10, freq="D"),
        }
    )
    train, test = split_history_frames(history, test_ratio=0.2)
    assert len(train) == 8 and len(test) == 2
    assert train["event_time"].max() < test["event_time"].min()


def _random_frame(features: list[str], n: int = 900, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame(rng.random((n, len(features))), columns=features)
    score = frame[features[:4]].sum(axis=1)
    frame["label"] = (score > score.median()).astype(int)
    return frame


def test_coarse_ranker_trains_and_persists(tmp_path):
    frame = _random_frame(COARSE_FEATURES)
    ranker = CoarseRanker()
    meta = ranker.train(frame)
    assert meta["train_auc"] > 0.5
    proba = ranker.predict(frame.head(10))
    assert ((proba >= 0) & (proba <= 1)).all()

    target = tmp_path / "coarse.joblib"
    ranker.save(target)
    assert CoarseRanker.load(target).features == COARSE_FEATURES


def test_fine_ranker_trains_and_reports_importance(tmp_path):
    frame = _random_frame(FINE_FEATURES, n=1200)
    ranker = FineRanker()
    meta = ranker.train(frame)
    assert meta["valid_auc"] is not None and meta["valid_auc"] > 0.5
    assert len(meta["top_features"]) > 0
    proba = ranker.predict(frame.head(5))
    assert ((proba >= 0) & (proba <= 1)).all()

    target = tmp_path / "fine.joblib"
    ranker.save(target)
    assert FineRanker.load(target).features == FINE_FEATURES


@pytest.mark.skipif(not (ARTIFACTS / "bm25_index.joblib").exists(), reason="尚未训练")
def test_trained_artifacts_are_usable():
    bm25 = BM25Index.load(ARTIFACTS / "bm25_index.joblib")
    vector = VectorIndex.load(ARTIFACTS / "vector_index.npz")
    itemcf = ItemCF.load(ARTIFACTS / "item_cf.joblib")

    hits = bm25.search("科幻 太空", topk=10)
    assert hits, "BM25 在真实语料上应有召回"

    any_movie = int(itemcf.item_ids[0])
    neighbors = itemcf.neighbors_of(any_movie, topk=5)
    assert neighbors and all(score > 0 for _, score in neighbors)

    vec_hits = vector.search(vector.user_vector([(any_movie, 1.0)]), topk=10)
    assert vec_hits

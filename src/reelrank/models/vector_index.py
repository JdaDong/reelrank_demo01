"""向量召回索引：行为矩阵 → TruncatedSVD 隐语义 → 归一化暴力 ANN。

- 影片向量 = SVD 的 item 因子（行为协同信号）
- 用户向量 = 历史影片向量的加权平均（冷启动时退化为全局热度向量）
- query 向量 = BM25 命中影片向量的加权平均（伪 query embedding，搜索场景）

产物：artifacts/vector_index.npz
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD

from reelrank.logging_utils import get_logger

logger = get_logger("reelrank.models.vector")


def _normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


class VectorIndex:
    """SVD 语义向量索引，检索使用归一化内积（万级规模毫秒级）。"""

    def __init__(self, dim: int = 64, random_state: int = 42):
        self.dim = dim
        self.random_state = random_state
        self.item_ids: np.ndarray = np.empty(0, dtype=np.int64)
        self.item_vectors: np.ndarray = np.empty((0, dim), dtype=np.float32)
        self.global_vector: np.ndarray = np.zeros(dim, dtype=np.float32)
        self.explained_variance: float = 0.0
        self._position: dict[int, int] = {}

    # ---------------- 构建 ----------------
    def build(self, item_ids: list[int], history: pd.DataFrame) -> "VectorIndex":
        item_ids = [int(i) for i in item_ids]
        self.item_ids = np.asarray(item_ids, dtype=np.int64)
        self._position = {mid: i for i, mid in enumerate(item_ids)}

        positive = history[history["label"] >= 1]
        rows, cols, values = [], [], []
        for user_id, movie_id, rating in zip(positive["user_id"], positive["movie_id"], positive["rating"]):
            position = self._position.get(int(movie_id))
            if position is None:
                continue
            rows.append(int(user_id))
            cols.append(position)
            values.append(float(rating) / 5.0 if pd.notna(rating) else 1.0)

        n_users = int(history["user_id"].max()) + 1 if len(history) else 1
        matrix = sparse.csr_matrix(
            (values, (rows, cols)), shape=(max(n_users, 1), len(item_ids)), dtype=np.float32
        )

        components = min(self.dim, min(matrix.shape) - 1)
        components = max(2, components)
        svd = TruncatedSVD(n_components=components, random_state=self.random_state)
        svd.fit(matrix)
        self.item_vectors = _normalize(svd.components_.T.astype(np.float32))
        self.explained_variance = float(svd.explained_variance_ratio_.sum())

        # 冷启动向量：按热度加权平均的影片向量（必须在 embedding 空间内，否则无法与 item 向量做内积）
        popularity = np.asarray(matrix.sum(axis=0)).ravel()
        if popularity.sum() <= 0:
            self.global_vector = _normalize(self.item_vectors.mean(axis=0, keepdims=True))[0]
        else:
            weights = popularity / popularity.sum()
            self.global_vector = _normalize((self.item_vectors * weights[:, None]).sum(axis=0, keepdims=True))[0]

        logger.info(
            "向量索引构建完成：%s 部影片 / 维度 %s / 累计解释方差 %.3f",
            len(item_ids), components, self.explained_variance,
        )
        return self

    # ---------------- 取向量 ----------------
    def item_vector(self, movie_id: int) -> np.ndarray:
        position = self._position.get(int(movie_id))
        if position is None:
            return self.global_vector
        return self.item_vectors[position]

    def user_vector(self, history: list[tuple[int, float]] | None) -> np.ndarray:
        """history = [(movie_id, weight)]，无历史时返回全局热度向量。"""
        if not history:
            return self.global_vector
        vectors, weights = [], []
        for movie_id, weight in history:
            position = self._position.get(int(movie_id))
            if position is None:
                continue
            vectors.append(self.item_vectors[position])
            weights.append(float(weight))
        if not vectors:
            return self.global_vector
        stacked = np.vstack(vectors)
        weights_arr = np.asarray(weights, dtype=np.float32).reshape(-1, 1)
        blended = (stacked * weights_arr).sum(axis=0) / weights_arr.sum()
        return _normalize(blended.reshape(1, -1))[0]

    def query_vector(self, query: str, bm25_index, topn: int = 20) -> np.ndarray:
        """搜索场景：用 BM25 命中影片的向量加权平均构造 query 向量。"""
        hits = bm25_index.search(query, topk=topn)
        if not hits:
            return self.global_vector
        return self.user_vector([(mid, score) for mid, score in hits])

    # ---------------- 检索 ----------------
    def search(self, vector: np.ndarray, topk: int = 200, exclude: set[int] | None = None) -> list[tuple[int, float]]:
        if self.item_vectors.size == 0:
            return []
        vector = np.asarray(vector, dtype=np.float32).ravel()
        if vector.shape[0] != self.item_vectors.shape[1]:  # 维度异常时退化为冷启动向量
            vector = self.global_vector
        scores = self.item_vectors @ vector
        if exclude:
            for movie_id in exclude:
                position = self._position.get(int(movie_id))
                if position is not None:
                    scores[position] = -1.0
        if topk < len(scores):
            candidate_idx = np.argpartition(-scores, topk)[:topk]
        else:
            candidate_idx = np.arange(len(scores))
        ordered = candidate_idx[np.argsort(-scores[candidate_idx])]
        return [(int(self.item_ids[i]), float(max(0.0, scores[i]))) for i in ordered[:topk] if scores[i] > 0]

    def score_movies(self, vector: np.ndarray, movie_ids: list[int]) -> np.ndarray:
        scores = self.item_vectors @ vector.astype(np.float32)
        return np.asarray([max(0.0, scores[self._position[mid]]) if mid in self._position else 0.0 for mid in movie_ids], dtype=np.float32)

    # ---------------- 持久化 ----------------
    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            item_ids=self.item_ids,
            item_vectors=self.item_vectors,
            global_vector=self.global_vector,
            meta=np.asarray([self.dim, self.explained_variance], dtype=np.float64),
        )

    @classmethod
    def load(cls, path: str | Path) -> "VectorIndex":
        data = np.load(path, allow_pickle=False)
        meta = data["meta"]
        index = cls(dim=int(meta[0]))
        index.item_ids = data["item_ids"]
        index.item_vectors = data["item_vectors"]
        index.global_vector = data["global_vector"]
        index.explained_variance = float(meta[1])
        index._position = {int(mid): i for i, mid in enumerate(index.item_ids)}
        return index

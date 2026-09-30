"""ItemCF 召回：行为共现余弦 + TMDB 相似/推荐关系图融合。

相似度 = 0.62 × 行为共现余弦 + 0.38 × 关系图分数（similar 权重 1.0，recommendation 0.75，按位次衰减）

产物：artifacts/item_cf.joblib
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from reelrank.logging_utils import get_logger

logger = get_logger("reelrank.models.itemcf")

RELATION_WEIGHT = {"similar": 1.0, "recommendation": 0.75, "recommended": 0.75}


class ItemCF:
    """Item-Item 协同过滤，保留每个 item 的 Top-N 邻居。"""

    def __init__(self, max_neighbors: int = 100, cooc_weight: float = 0.62, graph_weight: float = 0.38):
        self.max_neighbors = max_neighbors
        self.cooc_weight = cooc_weight
        self.graph_weight = graph_weight
        self.item_ids: np.ndarray = np.empty(0, dtype=np.int64)
        self.neighbor_ids: dict[int, np.ndarray] = {}
        self.neighbor_scores: dict[int, np.ndarray] = {}

    # ---------------- 构建 ----------------
    def build(
        self,
        item_ids: list[int],
        history: pd.DataFrame,
        edges: pd.DataFrame | None = None,
        block_size: int = 256,
    ) -> "ItemCF":
        item_ids = [int(i) for i in item_ids]
        self.item_ids = np.asarray(item_ids, dtype=np.int64)
        position = {mid: i for i, mid in enumerate(item_ids)}
        n_items = len(item_ids)

        positive = history[history["label"] >= 1]
        rows, cols = [], []
        for user_id, movie_id in zip(positive["user_id"], positive["movie_id"]):
            pos = position.get(int(movie_id))
            if pos is not None:
                rows.append(int(user_id))
                cols.append(pos)
        n_users = int(history["user_id"].max()) + 1 if len(history) else 1
        matrix = sparse.csr_matrix(
            (np.ones(len(rows), dtype=np.float32), (rows, cols)),
            shape=(max(n_users, 1), n_items),
            dtype=np.float32,
        )
        degree = np.asarray(matrix.sum(axis=0)).ravel()

        graph = self._build_graph(edges, position, n_items)
        neighbor_ids: dict[int, np.ndarray] = {}
        neighbor_scores: dict[int, np.ndarray] = {}

        for start in range(0, n_items, block_size):
            end = min(start + block_size, n_items)
            block = matrix[:, start:end].T @ matrix          # (block, n_items)
            block = block.toarray() if sparse.issparse(block) else np.asarray(block)
            for offset in range(end - start):                 # 仅去掉自身相似度（对角线）
                block[offset, start + offset] = 0.0

            denom = np.sqrt(np.outer(degree[start:end], degree))
            denom[denom == 0] = 1.0
            cooc = block / denom

            fused = self.cooc_weight * np.clip(cooc, 0.0, 1.0)
            if graph is not None:
                fused = fused + self.graph_weight * graph[start:end]

            k = min(self.max_neighbors, n_items - 1)
            if k <= 0:
                continue
            top_idx = np.argpartition(-fused, k - 1, axis=1)[:, :k]
            for offset, row_idx in enumerate(range(start, end)):
                candidates = top_idx[offset]
                values = fused[offset, candidates]
                order = np.argsort(-values)
                candidates = candidates[order]
                values = values[order]
                keep = values > 1e-6
                neighbor_ids[int(item_ids[row_idx])] = self.item_ids[candidates[keep]]
                neighbor_scores[int(item_ids[row_idx])] = values[keep].astype(np.float32)

        self.neighbor_ids = neighbor_ids
        self.neighbor_scores = neighbor_scores
        avg_degree = float(np.mean([len(v) for v in neighbor_ids.values()])) if neighbor_ids else 0.0
        logger.info("ItemCF 构建完成：%s 个 item，平均邻居数 %.1f", len(neighbor_ids), avg_degree)
        return self

    @staticmethod
    def _build_graph(edges: pd.DataFrame | None, position: dict[int, int], n_items: int) -> np.ndarray | None:
        if edges is None or edges.empty:
            return None
        graph = np.zeros((n_items, n_items), dtype=np.float32)
        for src, dst, relation, rank_pos in zip(edges["src_movie_id"], edges["dst_movie_id"], edges["relation"], edges["rank_pos"]):
            src_pos, dst_pos = position.get(int(src)), position.get(int(dst))
            if src_pos is None or dst_pos is None:
                continue
            weight = RELATION_WEIGHT.get(str(relation), 0.5) * (0.5 ** (float(rank_pos) / 6.0))
            graph[src_pos, dst_pos] = max(graph[src_pos, dst_pos], weight)
        return graph

    # ---------------- 召回 / 查询 ----------------
    def recall(self, triggers: list[tuple[int, float]], topk: int = 200, exclude: set[int] | None = None) -> list[tuple[int, float]]:
        """triggers = [(movie_id, weight)]（用户历史或 query 命中），返回聚合后的邻居分。"""
        if not triggers:
            return []
        aggregated: dict[int, float] = {}
        for movie_id, weight in triggers:
            neighbors = self.neighbor_ids.get(int(movie_id))
            scores = self.neighbor_scores.get(int(movie_id))
            if neighbors is None:
                continue
            for neighbor, score in zip(neighbors, scores):
                aggregated[int(neighbor)] = aggregated.get(int(neighbor), 0.0) + float(score) * float(weight)

        if exclude:
            for movie_id in exclude:
                aggregated.pop(int(movie_id), None)
        if not aggregated:
            return []
        ordered = sorted(aggregated.items(), key=lambda kv: kv[1], reverse=True)[:topk]
        max_score = ordered[0][1] or 1.0
        return [(mid, float(score / max_score)) for mid, score in ordered]

    def score_movies(self, triggers: list[tuple[int, float]], movie_ids: list[int]) -> np.ndarray:
        scores = dict(self.recall(triggers, topk=len(self.item_ids) + 1))
        return np.asarray([scores.get(int(mid), 0.0) for mid in movie_ids], dtype=np.float32)

    def neighbors_of(self, movie_id: int, topk: int = 10) -> list[tuple[int, float]]:
        ids = self.neighbor_ids.get(int(movie_id))
        scores = self.neighbor_scores.get(int(movie_id))
        if ids is None:
            return []
        pairs = list(zip((int(i) for i in ids[:topk]), (float(s) for s in scores[:topk])))
        return pairs

    # ---------------- 持久化 ----------------
    def save(self, path: str | Path) -> None:
        import joblib

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"item_ids": self.item_ids, "neighbors": self.neighbor_ids, "scores": self.neighbor_scores}, path)

    @classmethod
    def load(cls, path: str | Path) -> "ItemCF":
        import joblib

        payload = joblib.load(path)
        index = cls()
        index.item_ids = payload["item_ids"]
        index.neighbor_ids = payload["neighbors"]
        index.neighbor_scores = payload["scores"]
        return index

"""BM25 倒排索引：中文按「字 + 二元切分」，英文按词切分，无需外部分词器。

产物：artifacts/bm25_index.joblib
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from reelrank.logging_utils import get_logger

logger = get_logger("reelrank.models.bm25")

_LATIN = re.compile(r"[a-zA-Z0-9]+")
_CJK = re.compile(r"[一-鿿]+")


def tokenize(text: str) -> list[str]:
    """中英文混合切分：英文/数字按词，中文按字 + 相邻二字组。"""
    if not text:
        return []
    tokens: list[str] = []
    for match in _LATIN.finditer(text):
        tokens.append(match.group().lower())
    for match in _CJK.finditer(text):
        run = match.group()
        tokens.extend(run)
        tokens.extend(run[i : i + 2] for i in range(len(run) - 1))
    return tokens


class BM25Index:
    """标准 BM25：idf = ln(1 + (N - df + 0.5) / (df + 0.5))，长度归一化参数 b=0.75。"""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_ids: np.ndarray = np.empty(0, dtype=np.int64)
        self.doc_len: np.ndarray = np.empty(0, dtype=np.float32)
        self.avgdl: float = 0.0
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self.doc_freq: dict[str, int] = {}

    # ---------------- 构建 ----------------
    def build(self, movie_ids: list[int], texts: list[str]) -> "BM25Index":
        self.doc_ids = np.asarray(movie_ids, dtype=np.int64)
        self.doc_len = np.zeros(len(texts), dtype=np.float32)
        self.postings = defaultdict(list)
        doc_freq: dict[str, int] = defaultdict(int)

        for idx, text in enumerate(texts):
            counter: dict[str, int] = defaultdict(int)
            for token in tokenize(text or ""):
                counter[token] += 1
            self.doc_len[idx] = float(sum(counter.values()))
            for token, tf in counter.items():
                self.postings[token].append((idx, tf))
                doc_freq[token] += 1

        self.doc_freq = dict(doc_freq)
        self.avgdl = float(self.doc_len.mean()) if len(self.doc_len) else 1.0
        logger.info("BM25 索引构建完成：%s 篇文档 / %s 个词项", len(self.doc_ids), len(self.doc_freq))
        return self

    # ---------------- 检索 ----------------
    def _idf(self, token: str) -> float:
        df = self.doc_freq.get(token, 0)
        if df == 0:
            return 0.0
        return math.log(1.0 + (len(self.doc_ids) - df + 0.5) / (df + 0.5))

    def raw_scores(self, query: str) -> dict[int, float]:
        """返回 {文档下标: BM25 原始分}，未命中返回空字典。"""
        if not query or not len(self.doc_ids):
            return {}
        counter: dict[str, int] = defaultdict(int)
        for token in tokenize(query):
            counter[token] += 1

        scores: dict[int, float] = defaultdict(float)
        for token, qtf in counter.items():
            postings = self.postings.get(token)
            if not postings:
                continue
            idf = self._idf(token)
            for idx, tf in postings:
                denom = tf + self.k1 * (1 - self.b + self.b * self.doc_len[idx] / self.avgdl)
                scores[idx] += idf * (tf * (self.k1 + 1)) / denom * qtf
        return dict(scores)

    def search(self, query: str, topk: int = 200) -> list[tuple[int, float]]:
        """检索并返回 [(movie_id, 归一化分数 0~1)]，按分数降序。"""
        scores = self.raw_scores(query)
        if not scores:
            return []
        ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:topk]
        max_score = ordered[0][1] or 1.0
        return [(int(self.doc_ids[idx]), float(score / max_score)) for idx, score in ordered]

    def score_movies(self, query: str, movie_ids: list[int]) -> np.ndarray:
        """对指定影片集合打分（未命中的文档为 0），用于训练样本构造。"""
        scores = self.raw_scores(query)
        if not scores:
            return np.zeros(len(movie_ids), dtype=np.float32)
        max_score = max(scores.values()) or 1.0
        index_of = {int(mid): i for i, mid in enumerate(movie_ids)}
        result = np.zeros(len(movie_ids), dtype=np.float32)
        for idx, score in scores.items():
            position = index_of.get(int(self.doc_ids[idx]))
            if position is not None:
                result[position] = score / max_score
        return result

    # ---------------- 持久化 ----------------
    def save(self, path: str | Path) -> None:
        import joblib

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str | Path) -> "BM25Index":
        import joblib

        return joblib.load(path)

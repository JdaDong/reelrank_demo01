"""评估指标：Recall@K、NDCG@K、AUC/GAUC、eCPM/RPM。"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def recall_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    """召回率：前 K 个结果命中相关集合的比例。"""
    if not relevant:
        return 0.0
    hits = sum(1 for item in list(recommended)[:k] if int(item) in relevant)
    return hits / len(relevant)


def precision_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    if k <= 0:
        return 0.0
    hits = sum(1 for item in list(recommended)[:k] if int(item) in relevant)
    return hits / k


def dcg(gains: list[float]) -> float:
    return float(sum(gain / math.log2(idx + 2) for idx, gain in enumerate(gains)))


def ndcg_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    """NDCG@K：二元相关性，位置折扣 log2(i+2)。"""
    if not relevant:
        return 0.0
    gains = [1.0 if int(item) in relevant else 0.0 for item in list(recommended)[:k]]
    ideal = [1.0] * min(len(relevant), k)
    ideal_dcg = dcg(ideal)
    if ideal_dcg <= 0:
        return 0.0
    return dcg(gains) / ideal_dcg


def auc(labels, scores) -> float:
    labels = np.asarray(labels)
    scores = np.asarray(scores, dtype=float)
    if len(np.unique(labels)) < 2:
        return 0.5
    return float(roc_auc_score(labels, scores))


def gauc(frame: pd.DataFrame, group_col: str, label_col: str, score_col: str, min_group: int = 2) -> float:
    """GAUC：按 group（用户/请求）计算 AUC 后按样本量加权，衡量个性化排序能力。"""
    total, weight_sum = 0.0, 0.0
    for _, group in frame.groupby(group_col, sort=False):
        if len(group) < min_group or group[label_col].nunique() < 2:
            continue
        value = auc(group[label_col], group[score_col])
        total += value * len(group)
        weight_sum += len(group)
    return total / weight_sum if weight_sum else 0.5


def rpm(revenue: float, impressions: int) -> float:
    """RPM：每千次曝光收入（元）。"""
    if impressions <= 0:
        return 0.0
    return float(revenue) * 1000.0 / float(impressions)


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=float), q))

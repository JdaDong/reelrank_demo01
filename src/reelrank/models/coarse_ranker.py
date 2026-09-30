"""粗排模型：LogisticRegression（标准化 + 类别均衡），输入为召回信号 + 影片特征。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from reelrank.features.registry import COARSE_FEATURES, build_matrix
from reelrank.logging_utils import get_logger

logger = get_logger("reelrank.models.coarse")


class CoarseRanker:
    """轻量线性模型，用于千级候选的快速打分。"""

    def __init__(self, features: list[str] | None = None, random_state: int = 42):
        self.features = list(features or COARSE_FEATURES)
        self.random_state = random_state
        self.model: Pipeline | None = None
        self.train_auc: float | None = None

    def train(self, frame: pd.DataFrame, label_col: str = "label") -> dict:
        matrix = build_matrix(frame, self.features)
        labels = frame[label_col].astype(int).to_numpy()
        self.model = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("lr", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=self.random_state)),
            ]
        )
        self.model.fit(matrix, labels)
        proba = self.model.predict_proba(matrix)[:, 1]
        self.train_auc = float(roc_auc_score(labels, proba)) if len(np.unique(labels)) > 1 else 0.5
        logger.info("粗排 LR 训练完成：%s 样本 / AUC=%.4f", len(frame), self.train_auc)
        return {"samples": int(len(frame)), "train_auc": round(self.train_auc, 4), "features": self.features}

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("粗排模型未训练")
        return self.model.predict_proba(build_matrix(frame, self.features))[:, 1].astype(np.float32)

    def weights(self) -> dict[str, float]:
        if self.model is None:
            return {}
        coef = self.model.named_steps["lr"].coef_[0]
        return {name: float(value) for name, value in zip(self.features, coef)}

    def save(self, path: str | Path) -> None:
        import joblib

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": self.model, "features": self.features, "train_auc": self.train_auc}, path)

    @classmethod
    def load(cls, path: str | Path) -> "CoarseRanker":
        import joblib

        payload = joblib.load(path)
        ranker = cls(features=payload["features"])
        ranker.model = payload["model"]
        ranker.train_auc = payload["train_auc"]
        return ranker

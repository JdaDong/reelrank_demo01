"""精排模型：GBDT（HistGradientBoostingClassifier）预估 pCTR，融合质量与新鲜度多目标。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from reelrank.config import Settings, settings as global_settings
from reelrank.features.registry import FINE_FEATURES, build_matrix
from reelrank.logging_utils import get_logger

logger = get_logger("reelrank.models.fine")

try:  # scikit-learn >= 1.4 提供原生实现
    from sklearn.ensemble import HistGradientBoostingClassifier as _HGB
except ImportError:  # pragma: no cover
    _HGB = None


class FineRanker:
    """精排 GBDT：输入召回信号 + 影片 + 用户 + 交叉特征，输出 pCTR。"""

    def __init__(self, features: list[str] | None = None, settings: Settings | None = None):
        self.settings = settings or global_settings
        self.features = list(features or FINE_FEATURES)
        self.model = None
        self.train_auc: float | None = None
        self.valid_auc: float | None = None
        self.importance: dict[str, float] = {}

    def train(self, frame: pd.DataFrame, label_col: str = "label") -> dict:
        cfg = self.settings.funnel.fine.gbdt
        matrix = build_matrix(frame, self.features)
        labels = frame[label_col].astype(int).to_numpy()

        self.model = _HGB(
            max_iter=int(cfg.max_iter),
            learning_rate=float(cfg.learning_rate),
            max_depth=int(cfg.max_depth),
            random_state=int(self.settings.train.random_state),
        )
        self.model.fit(matrix, labels)

        proba = self.model.predict_proba(matrix)[:, 1]
        self.train_auc = float(roc_auc_score(labels, proba)) if len(np.unique(labels)) > 1 else 0.5

        # 留出集评估 + 排列重要性（采样控制耗时）
        if len(np.unique(labels)) > 1 and len(matrix) > 500:
            x_train, x_valid, y_train, y_valid = train_test_split(
                matrix, labels, test_size=0.2, random_state=int(self.settings.train.random_state), stratify=labels
            )
            valid_model = _HGB(
                max_iter=int(cfg.max_iter),
                learning_rate=float(cfg.learning_rate),
                max_depth=int(cfg.max_depth),
                random_state=int(self.settings.train.random_state),
            )
            valid_model.fit(x_train, y_train)
            self.valid_auc = float(roc_auc_score(y_valid, valid_model.predict_proba(x_valid)[:, 1]))

            sample = x_valid.iloc[: min(4000, len(x_valid))]
            sample_y = y_valid[: len(sample)]
            result = permutation_importance(
                valid_model, sample, sample_y, n_repeats=3,
                random_state=int(self.settings.train.random_state), scoring="roc_auc",
            )
            self.importance = {
                name: round(float(value), 5)
                for name, value in sorted(
                    zip(self.features, result.importances_mean), key=lambda kv: kv[1], reverse=True
                )
            }

        logger.info(
            "精排 GBDT 训练完成：%s 样本 / train AUC=%.4f / valid AUC=%.4f",
            len(frame), self.train_auc, self.valid_auc if self.valid_auc else float("nan"),
        )
        return {
            "samples": int(len(frame)),
            "train_auc": round(self.train_auc, 4),
            "valid_auc": round(self.valid_auc, 4) if self.valid_auc else None,
            "top_features": list(self.importance.items())[:8],
        }

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("精排模型未训练")
        return self.model.predict_proba(build_matrix(frame, self.features))[:, 1].astype(np.float32)

    def predict_with_breakdown(self, frame: pd.DataFrame) -> tuple[np.ndarray, dict[str, float]]:
        """返回 pCTR 与关键特征贡献（用于调试面板的分数归因）。"""
        pctr = self.predict(frame)
        breakdown = {
            "pctr_mean": float(np.mean(pctr)),
            "quality_mean": float(frame["quality_bayes"].mean()) if "quality_bayes" in frame else 0.0,
            "freshness_mean": float(frame["freshness"].mean()) if "freshness" in frame else 0.0,
        }
        return pctr, breakdown

    def save(self, path: str | Path) -> None:
        import joblib

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "model": self.model,
                "features": self.features,
                "train_auc": self.train_auc,
                "valid_auc": self.valid_auc,
                "importance": self.importance,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> "FineRanker":
        import joblib

        payload = joblib.load(path)
        ranker = cls(features=payload["features"])
        ranker.model = payload["model"]
        ranker.train_auc = payload["train_auc"]
        ranker.valid_auc = payload["valid_auc"]
        ranker.importance = payload.get("importance", {})
        return ranker

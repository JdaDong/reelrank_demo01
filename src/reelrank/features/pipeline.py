"""特征流水线编排：行为对齐 → 影片特征 → 用户画像 → 元信息落盘。"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pandas as pd

from reelrank.config import Settings, settings as global_settings
from reelrank.data.behavior_source import ensure_events
from reelrank.features.item_features import build_item_features
from reelrank.features.user_features import build_user_features
from reelrank.logging_utils import get_logger

logger = get_logger("reelrank.features.pipeline")


def run_features(settings: Settings | None = None, refresh: bool = False) -> dict:
    settings = settings or global_settings
    behavior_meta = ensure_events(settings, refresh=refresh)
    items: pd.DataFrame = build_item_features(settings)
    users: pd.DataFrame = build_user_features(settings, items)

    meta = {
        "behavior": behavior_meta,
        "item_count": int(len(items)),
        "user_count": int(len(users)),
        "item_features_path": "artifacts/item_features.parquet",
        "user_features_path": "artifacts/user_features.parquet",
        "built_at": datetime.now(timezone.utc).isoformat(),
    }
    target = settings.path("artifacts/features_meta.json")
    target.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("特征流水线完成：%s 部影片 / %s 个用户", meta["item_count"], meta["user_count"])
    return meta


if __name__ == "__main__":
    print(json.dumps(run_features(), ensure_ascii=False, indent=2))

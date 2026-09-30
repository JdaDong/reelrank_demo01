"""广告库：以 TMDB 影片作为被推广的宣发内容，广告主 = 片方/流媒体。

构造信息（bid、日预算、频次上限、定向）由固定种子生成，预算消耗落盘持久化。
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

from reelrank.config import Settings, settings as global_settings
from reelrank.logging_utils import get_logger

logger = get_logger("reelrank.ads.index")

ADVERTISER_NAMES = [
    "北岸影业", "灯塔宣发", "银幕工场", "蓝鲸传媒", "拾光影业", "黑石影业",
    "远山制片", "恒星影业", "潮汐影业", "第七映画", "星野传媒", "长夜影业",
    "橙光映画", "白鲸影业", "南风传媒", "拓野影业",
]


def _as_list(value) -> list:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return list(value)


@dataclass
class Ad:
    ad_id: int
    advertiser_id: int
    advertiser_name: str
    movie_id: int
    title: str
    bid: float
    daily_budget: float
    freq_cap: int
    targeting: list[str] = field(default_factory=list)
    status: str = "active"


class AdIndex:
    """广告库与预算状态（spent 落盘，重启不丢）。"""

    def __init__(self, ads: list[Ad], budget_path: Path, spent: dict[str, float] | None = None, budget_date: str | None = None):
        self.ads = ads
        self.by_id: dict[int, Ad] = {ad.ad_id: ad for ad in ads}
        self.budget_path = budget_path
        self.budget_date = budget_date or date.today().isoformat()
        self.spent: dict[str, float] = spent or {str(ad.ad_id): 0.0 for ad in ads}
        self.reset_if_new_day()

    # ---------------- 构建 ----------------
    @classmethod
    def build(cls, settings: Settings | None = None, items=None, persist: bool = True) -> "AdIndex":
        settings = settings or global_settings
        cfg = settings.ads
        rng = random.Random(int(settings.behavior.synthetic.seed))

        ranked = items.sort_values("popularity", ascending=False)
        pool_size = int(cfg.advertiser_count) * int(cfg.creatives_per_advertiser)
        pool = ranked.head(max(pool_size, int(cfg.advertiser_count))).to_dict("records")

        ads: list[Ad] = []
        ad_id = 1
        for advertiser_index in range(int(cfg.advertiser_count)):
            advertiser_id = 1000 + advertiser_index
            advertiser_name = ADVERTISER_NAMES[advertiser_index % len(ADVERTISER_NAMES)]
            bid = round(rng.uniform(float(cfg.bid_min), float(cfg.bid_max)), 2)
            daily_budget = round(float(cfg.default_daily_budget) * rng.uniform(0.5, 1.5), 2)

            start = advertiser_index * int(cfg.creatives_per_advertiser)
            creatives = pool[start : start + int(cfg.creatives_per_advertiser)] or [pool[rng.randrange(len(pool))]]
            for row in creatives:
                genres = [str(g) for g in _as_list(row.get("genre_names"))][:2]
                ads.append(
                    Ad(
                        ad_id=ad_id,
                        advertiser_id=advertiser_id,
                        advertiser_name=advertiser_name,
                        movie_id=int(row["movie_id"]),
                        title=str(row["title"]),
                        bid=bid,
                        daily_budget=daily_budget,
                        freq_cap=int(cfg.freq_cap_per_session),
                        targeting=genres,
                        status="active",
                    )
                )
                ad_id += 1

        budget_path = settings.path(str(cfg.budget_state_path))
        index = cls(ads, budget_path)
        if persist:
            index.save(settings.path("artifacts/ads.json"))
            index.persist_budget()
        logger.info("广告库构建完成：%s 个广告 / %s 个广告主", len(ads), int(cfg.advertiser_count))
        return index

    # ---------------- 定向与预算 ----------------
    def eligible(self, context_genres: list[str] | None, limit: int = 200) -> list[Ad]:
        context = set(context_genres or [])
        result = []
        for ad in self.ads:
            if ad.status != "active":
                continue
            if context and ad.targeting and not context & set(ad.targeting):
                continue
            result.append(ad)
        return result[:limit] if limit else result

    def remaining_budget(self, ad: Ad) -> float:
        return max(0.0, float(ad.daily_budget) - float(self.spent.get(str(ad.ad_id), 0.0)))

    def charge(self, ad: Ad, amount: float) -> None:
        self.spent[str(ad.ad_id)] = float(self.spent.get(str(ad.ad_id), 0.0)) + float(amount)

    def reset_if_new_day(self) -> None:
        today = date.today().isoformat()
        if self.budget_date != today:
            logger.info("跨日重置广告预算：%s → %s", self.budget_date, today)
            self.budget_date = today
            self.spent = {str(ad.ad_id): 0.0 for ad in self.ads}
            self.persist_budget()

    def persist_budget(self) -> None:
        self.budget_path.parent.mkdir(parents=True, exist_ok=True)
        self.budget_path.write_text(
            json.dumps({"date": self.budget_date, "spent": self.spent}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def advertiser_board(self) -> list[dict]:
        """广告看板数据：广告主的出价、预算与消耗。"""
        board: dict[int, dict] = {}
        for ad in self.ads:
            entry = board.setdefault(
                ad.advertiser_id,
                {
                    "advertiser_id": ad.advertiser_id,
                    "advertiser_name": ad.advertiser_name,
                    "bid": ad.bid,
                    "daily_budget": ad.daily_budget,
                    "spent": 0.0,
                    "freq_cap": ad.freq_cap,
                    "creatives": 0,
                    "targeting": set(),
                },
            )
            entry["spent"] += float(self.spent.get(str(ad.ad_id), 0.0))
            entry["creatives"] += 1
            entry["targeting"].update(ad.targeting)
        result = []
        for entry in board.values():
            entry["targeting"] = sorted(entry["targeting"])
            entry["budget_usage"] = round(entry["spent"] / entry["daily_budget"], 4) if entry["daily_budget"] else 0.0
            entry["spent"] = round(entry["spent"], 2)
            result.append(entry)
        return sorted(result, key=lambda x: x["advertiser_id"])

    # ---------------- 持久化 ----------------
    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        payload = [ad.__dict__ for ad in self.ads]
        Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path, settings: Settings | None = None) -> "AdIndex":
        settings = settings or global_settings
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        ads = [Ad(**item) for item in payload]
        budget_path = settings.path(str(settings.ads.budget_state_path))
        spent, budget_date = None, None
        if budget_path.exists():
            state = json.loads(budget_path.read_text(encoding="utf-8"))
            spent = state.get("spent", {})
            budget_date = state.get("date")
        return cls(ads, budget_path, spent=spent, budget_date=budget_date)

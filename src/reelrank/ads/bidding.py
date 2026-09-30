"""广告竞价：eCPM = pCTR × bid × 质量因子 × scale，GSP 二价扣费 + 频次/预算控制。"""

from __future__ import annotations

import time

from reelrank.config import Settings, settings as global_settings
from reelrank.ads.ad_index import Ad
from reelrank.logging_utils import get_logger
from reelrank.recall.multi_recall import Candidate
from reelrank.rank.coarse import build_feature_frame
from reelrank.serving.schemas import AdItem, StageTrace

logger = get_logger("reelrank.ads.bidding")


def quality_factor(settings: Settings, vote_average: float, vote_count: float) -> float:
    """质量因子：评分越高、票数越可信，因子越大（裁剪到配置区间）。"""
    cfg = settings.ads
    confidence = min(1.0, (vote_count or 0.0) / 2000.0)
    raw = 0.75 + 0.5 * (float(vote_average or 6.0) / 10.0) * (0.6 + 0.4 * confidence)
    return float(min(float(cfg.quality_factor_max), max(float(cfg.quality_factor_min), raw)))


def compute_ecpm(pctr: float, bid: float, quality: float, scale: float = 1000.0) -> float:
    """eCPM（元/千次曝光）= pCTR × bid(元/点击) × 质量因子 × 1000。

    例：pCTR=5%、bid=2 元、质量因子=1 → eCPM = 0.05 × 2 × 1 × 1000 = 100 元/千次曝光。
    """
    return float(pctr) * float(bid) * float(quality) * float(scale)


def gsp_price(next_ecpm: float, pctr: float, quality: float, bid: float, settings: Settings) -> float:
    """GSP 二价：把下一名的 eCPM 折算成本 CPC，并夹在 [底价折算 CPC, 自身出价] 之间。"""
    scale = float(settings.ads.ecpm_scale)
    reserve_cpc = float(settings.ads.reserve_price) / scale  # 底价（元/千次）→ CPC（元/点击）
    if pctr <= 0 or quality <= 0:
        return round(min(float(bid), reserve_cpc), 4)
    price = next_ecpm / (pctr * quality * scale) + 0.01
    return float(min(float(bid), max(reserve_cpc, price)))


def run_auction(
    state,
    ads: list[Ad],
    profile: dict,
    session_counts: dict[int, int],
    slots: list[int],
    settings: Settings | None = None,
) -> tuple[list[AdItem], list[AdItem], StageTrace]:
    """执行广告竞价，返回 (中标广告, 被拦广告, 广告层 trace)。"""
    settings = settings or global_settings
    started = time.perf_counter()
    cfg = settings.ads
    winners: list[AdItem] = []
    blocked: list[AdItem] = []
    if not ads:
        return winners, blocked, StageTrace(stage="ads", input_count=0, output_count=0, latency_ms=0.0, cut_reason="无候选广告")

    # pCTR 复用精排模型（同一套特征口径）
    candidates = [Candidate(movie_id=ad.movie_id, sources=["ad"]) for ad in ads]
    frame = build_feature_frame(state, candidates, profile)
    pctrs = state.fine.predict(frame)

    scored: list[tuple[Ad, float, float, float]] = []  # (ad, pctr, quality, ecpm)
    for ad, pctr in zip(ads, pctrs):
        raw = state.item_raw.get(int(ad.movie_id), {})
        quality = quality_factor(settings, float(raw.get("vote_average", 6.0)), float(raw.get("vote_count", 0)))
        ecpm = compute_ecpm(pctr, ad.bid, quality, float(cfg.ecpm_scale))

        advertiser_seen = int(session_counts.get(int(ad.advertiser_id), 0))
        if advertiser_seen >= int(ad.freq_cap):
            blocked.append(AdItem(ad_id=ad.ad_id, advertiser_id=ad.advertiser_id, advertiser_name=ad.advertiser_name,
                                  movie_id=ad.movie_id, title=ad.title, pctr=float(pctr), bid=ad.bid,
                                  quality_factor=quality, ecpm=ecpm, blocked_reason=f"频次超限({advertiser_seen}/{ad.freq_cap})"))
            continue
        if ecpm < float(cfg.min_ecpm):
            blocked.append(AdItem(ad_id=ad.ad_id, advertiser_id=ad.advertiser_id, advertiser_name=ad.advertiser_name,
                                  movie_id=ad.movie_id, title=ad.title, pctr=float(pctr), bid=ad.bid,
                                  quality_factor=quality, ecpm=ecpm, blocked_reason=f"eCPM 低于门槛({ecpm:.2f}<{cfg.min_ecpm})"))
            continue
        if state.ad_index.remaining_budget(ad) <= 0:
            blocked.append(AdItem(ad_id=ad.ad_id, advertiser_id=ad.advertiser_id, advertiser_name=ad.advertiser_name,
                                  movie_id=ad.movie_id, title=ad.title, pctr=float(pctr), bid=ad.bid,
                                  quality_factor=quality, ecpm=ecpm, blocked_reason="日预算耗尽"))
            continue
        scored.append((ad, float(pctr), quality, ecpm))

    scored.sort(key=lambda item: item[3], reverse=True)
    need = len(slots)
    for position, (ad, pctr, quality, ecpm) in enumerate(scored[:need]):
        next_ecpm = scored[position + 1][3] if position + 1 < len(scored) else ecpm * 0.85
        price = gsp_price(next_ecpm, pctr, quality, ad.bid, settings)
        cost = price * pctr  # 按期望扣费（eCPM/scale）
        if cost > state.ad_index.remaining_budget(ad):
            blocked.append(AdItem(ad_id=ad.ad_id, advertiser_id=ad.advertiser_id, advertiser_name=ad.advertiser_name,
                                  movie_id=ad.movie_id, title=ad.title, pctr=pctr, bid=ad.bid,
                                  quality_factor=quality, ecpm=ecpm, blocked_reason="剩余预算不足"))
            continue
        state.ad_index.charge(ad, cost)
        session_counts[int(ad.advertiser_id)] = int(session_counts.get(int(ad.advertiser_id), 0)) + 1
        winners.append(
            AdItem(
                ad_id=ad.ad_id, advertiser_id=ad.advertiser_id, advertiser_name=ad.advertiser_name,
                movie_id=ad.movie_id, title=ad.title, slot=int(slots[position]) if position < len(slots) else position,
                pctr=round(pctr, 5), bid=ad.bid, quality_factor=round(quality, 4),
                ecpm=round(ecpm, 3), price=round(price, 4),
            )
        )

    if winners:
        state.ad_index.persist_budget()

    trace = StageTrace(
        stage="ads",
        input_count=len(ads),
        output_count=len(winners),
        latency_ms=round((time.perf_counter() - started) * 1000, 3),
        cut_reason=f"eCPM 排序取前 {need} 个广告位，GSP 二价扣费；拦截 {len(blocked)} 条（频次/eCPM/预算）",
    )
    return winners, blocked, trace
